from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

if __package__:
    from .admission import evaluate_candidate
    from .extractor import ExtractorSettings, extract_candidates
    from .projection import rebuild_current_view
    from .reconciliation import reconcile_candidate
    from .redaction import contains_sensitive_material
else:
    _PLUGIN_DIR = Path(__file__).resolve().parent
    _PKG_ROOT = "hermes_plugins"
    _PKG_NAME = "hermes_plugins.truth_ledger"

    if _PKG_ROOT not in sys.modules:
        root_pkg = type(sys)(_PKG_ROOT)
        root_pkg.__path__ = []
        sys.modules[_PKG_ROOT] = root_pkg
    if _PKG_NAME not in sys.modules:
        pkg = type(sys)(_PKG_NAME)
        pkg.__path__ = [str(_PLUGIN_DIR)]
        sys.modules[_PKG_NAME] = pkg

    if not hasattr(sys.modules[_PKG_NAME], "on_post_llm_call"):
        init_spec = importlib.util.spec_from_file_location(
            _PKG_NAME,
            _PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(_PLUGIN_DIR)],
        )
        if init_spec is None or init_spec.loader is None:
            raise ImportError("Unable to load truth-ledger runtime package")
        init_mod = importlib.util.module_from_spec(init_spec)
        init_mod.__package__ = _PKG_NAME
        sys.modules[_PKG_NAME] = init_mod
        init_spec.loader.exec_module(init_mod)

    def _load_local(name: str):
        module_name = f"{_PKG_NAME}.{name}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            _PLUGIN_DIR / f"{name}.py",
            submodule_search_locations=[str(_PLUGIN_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load local truth-ledger module: {name}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG_NAME
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    evaluate_candidate = _load_local("admission").evaluate_candidate
    _extractor_mod = _load_local("extractor")
    ExtractorSettings = _extractor_mod.ExtractorSettings
    extract_candidates = _extractor_mod.extract_candidates
    rebuild_current_view = _load_local("projection").rebuild_current_view
    reconcile_candidate = _load_local("reconciliation").reconcile_candidate
    contains_sensitive_material = _load_local("redaction").contains_sensitive_material

_ALLOWED_CLASSES = {"assert", "confirm", "supersede", "retract", "none"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_corpus_jsonl(path: str | Path) -> list[dict[str, Any]]:
    corpus_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(payload)
    return rows


def write_corpus_jsonl(path: str | Path, fixtures: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in fixtures:
            fh.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def _expected_class(row: Mapping[str, Any]) -> str:
    expected = row.get("expected")
    if isinstance(expected, Mapping):
        cls = str(expected.get("class") or "none").strip().lower()
    else:
        cls = "none"
    if cls not in _ALLOWED_CLASSES:
        return "none"
    return cls


def _prediction_matches_expected(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    expected = row.get("expected")
    if not isinstance(expected, Mapping):
        expected = {"class": "none"}

    expected_class = _expected_class(row)
    predicted_class = str(prediction.get("class") or "none")
    if predicted_class != expected_class:
        return False

    if predicted_class == "none":
        return True

    predicted_event = prediction.get("event")
    if not isinstance(predicted_event, Mapping):
        return False

    predicted_fact = predicted_event.get("fact")
    if not isinstance(predicted_fact, Mapping):
        return False

    for key in ("scope", "kind", "subject", "key"):
        if key in expected and expected.get(key) != predicted_fact.get(key):
            return False

    if "value" in expected and expected.get("value") != predicted_fact.get("value"):
        return False

    return True


def _structured_diagnostics(
    row: Mapping[str, Any], prediction: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    raw_expected = row.get("expected")
    expected = raw_expected if isinstance(raw_expected, Mapping) else {"class": "none"}
    expected_fact = {
        key: expected.get(key)
        for key in ("scope", "kind", "subject", "key", "value")
        if key in expected
    }

    predicted_fact: dict[str, Any] = {}
    predicted_event = prediction.get("event")
    if isinstance(predicted_event, Mapping):
        raw_fact = predicted_event.get("fact")
        if isinstance(raw_fact, Mapping):
            predicted_fact = {
                key: raw_fact.get(key)
                for key in ("scope", "kind", "subject", "key", "value")
                if key in raw_fact
            }

    mismatch_fields: list[str] = []
    if str(prediction.get("class") or "none") != _expected_class(row):
        mismatch_fields.append("class")
    for key, value in expected_fact.items():
        if predicted_fact.get(key) != value:
            mismatch_fields.append(key)
    return expected_fact, predicted_fact, mismatch_fields


def _safe_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        base = dict(metadata)
    else:
        base = {}

    base.setdefault("profile", "default")
    base.setdefault("platform", "cli")
    base.setdefault("session_id", f"sess-{row.get('fixture_id', 'unknown')}")
    base.setdefault("turn_id", f"turn-{row.get('fixture_id', 'unknown')}")
    return base


def _turn_text(row: Mapping[str, Any]) -> str:
    text = row.get("turn_text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _assistant_text(row: Mapping[str, Any]) -> str:
    value = row.get("assistant_text")
    if isinstance(value, str):
        return value.strip()
    return ""


def _extractor_envelope(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _safe_metadata(row)
    return {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": str(row.get("captured_at") or _now_iso()),
        "profile": metadata.get("profile"),
        "session_id": metadata.get("session_id"),
        "turn_id": metadata.get("turn_id"),
        "origin": {
            "platform": metadata.get("platform"),
            "conversation_id": metadata.get("conversation_id"),
            "thread_id": metadata.get("thread_id"),
            "speaker_id": metadata.get("speaker_id"),
        },
        "input": {"user_message": _turn_text(row)},
        "output": {"assistant_response": _assistant_text(row)},
        "attempt_count": 0,
    }


def _run_extractor_for_row(
    *,
    row: Mapping[str, Any],
    extractor_fn: Any,
    extractor_ctx: Any,
    extractor_settings: Any,
    attempt_count: int = 0,
) -> dict[str, Any]:
    envelope = _extractor_envelope(row)
    envelope["attempt_count"] = attempt_count
    if extractor_fn is not None:
        result = extractor_fn(row=row, envelope=envelope)
        return dict(result) if isinstance(result, Mapping) else {"status": "unavailable", "reason": "invalid_extractor_result"}

    if extractor_ctx is None:
        return {"status": "unavailable", "reason": "extractor_unavailable"}

    try:
        return cast(
            dict[str, Any],
            asyncio.run(
                extract_candidates(
                    ctx=extractor_ctx,
                    envelope=envelope,
                    settings=extractor_settings or ExtractorSettings(),
                )
            ),
        )
    except Exception as exc:
        return {"status": "unavailable", "reason": f"extractor_error:{type(exc).__name__}"}


def _predict_one(
    *,
    row: Mapping[str, Any],
    history_by_stream: dict[str, list[dict[str, Any]]],
    extraction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _safe_metadata(row)
    candidate_obj = row.get("candidate")
    if not isinstance(candidate_obj, Mapping):
        return {
            "class": "none",
            "reason": "no_candidate",
            "admitted": False,
            "duplicate": False,
            "gate_skipped": True,
            "event": None,
        }

    candidate = dict(candidate_obj)
    gate = evaluate_candidate(candidate, metadata)
    if not gate.get("admit"):
        return {
            "class": "none",
            "reason": str(gate.get("reason") or "rejected"),
            "admitted": False,
            "duplicate": False,
            "gate_skipped": True,
            "event": None,
        }

    stream = str(row.get("stream") or row.get("fixture_id") or "default")
    history = history_by_stream[stream]
    occurred_at = str(row.get("occurred_at") or _now_iso())

    reconcile_candidate_payload = {
        "scope": candidate.get("scope"),
        "kind": candidate.get("kind"),
        "subject": candidate.get("subject"),
        "key": candidate.get("key"),
        "value": candidate.get("value"),
        "proposed_operation": candidate.get("operation") or "assert",
    }

    recon = reconcile_candidate(
        history=history,
        observation={
            "profile": metadata.get("profile"),
            "platform": metadata.get("platform"),
            "session_id": metadata.get("session_id"),
            "turn_id": metadata.get("turn_id"),
            "task_id": metadata.get("task_id"),
            "speaker_id": metadata.get("speaker_id"),
            "conversation_id": metadata.get("conversation_id"),
            "thread_id": metadata.get("thread_id"),
        },
        candidate=reconcile_candidate_payload,
        occurred_at=occurred_at,
        extraction=extraction,
    )

    if recon.get("decision") == "append":
        event_raw = recon.get("event")
        event = cast(dict[str, Any], dict(event_raw)) if isinstance(event_raw, Mapping) else {}
        history.append(event)
        return {
            "class": str(event.get("operation") or "none"),
            "reason": None,
            "admitted": True,
            "duplicate": False,
            "gate_skipped": False,
            "event": event,
        }

    if recon.get("decision") == "duplicate":
        return {
            "class": "none",
            "reason": "duplicate",
            "admitted": False,
            "duplicate": True,
            "gate_skipped": False,
            "event": recon.get("event") if isinstance(recon.get("event"), Mapping) else None,
        }

    return {
        "class": "none",
        "reason": str(recon.get("reason") or "none"),
        "admitted": False,
        "duplicate": False,
        "gate_skipped": False,
        "event": None,
    }


def _scan_leakage(payload: Any) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return contains_sensitive_material(serialized)


def evaluate_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    extractor_fn: Any = None,
    extractor_ctx: Any = None,
    extractor_settings: Any = None,
    expected_extraction: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    history_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class = Counter()
    expected_by_class = Counter()

    evaluated: list[dict[str, Any]] = []
    admitted = 0
    expected_admissible = 0
    true_positive = 0
    no_fact_expected = 0
    no_fact_correct = 0
    correct_total = 0
    duplicate_count = 0
    gate_skipped = 0
    leakage_findings = 0

    extraction_status = Counter()
    observed_routes = Counter()
    provenance_mismatches = 0
    extracted_turns = 0
    retry_limit = max(0, int(getattr(extractor_settings, "max_attempts", 6)))

    for row in fixtures:
        extraction: dict[str, Any] = {"status": "unavailable", "reason": "not_attempted"}
        turn_text = _turn_text(row)
        if not turn_text:
            extraction = {"status": "unavailable", "reason": "missing_turn_text"}
        else:
            for attempt_count in range(retry_limit + 1):
                extraction = _run_extractor_for_row(
                    row=row,
                    extractor_fn=extractor_fn,
                    extractor_ctx=extractor_ctx,
                    extractor_settings=extractor_settings,
                    attempt_count=attempt_count,
                )
                if str(extraction.get("status") or "unavailable") != "retry":
                    break
                if attempt_count < retry_limit:
                    delay_seconds = min(max(float(extraction.get("retry_delay_ms") or 0) / 1000.0, 0.0), 1.0)
                    if delay_seconds:
                        time.sleep(delay_seconds)
        status = str(extraction.get("status") or "unavailable")
        extraction_status[status] += 1

        candidates: list[dict[str, Any]] = []
        facts = extraction.get("facts")
        if status == "ok" and isinstance(facts, list):
            candidates = [dict(fact) for fact in facts if isinstance(fact, Mapping)]

        extraction_meta_raw = extraction.get("extraction")
        extraction_meta = dict(extraction_meta_raw) if isinstance(extraction_meta_raw, Mapping) else {}
        if status in {"ok", "none"} and extraction_meta:
            provider = str(extraction_meta.get("provider") or "unknown")
            model = str(extraction_meta.get("model") or "unknown")
            observed_routes[f"{provider}/{model}"] += 1
            if expected_extraction is not None and (
                provider != str(expected_extraction.get("provider") or "")
                or model != str(expected_extraction.get("model") or "")
            ):
                provenance_mismatches += 1
        elif status in {"ok", "none"} and expected_extraction is not None:
            provenance_mismatches += 1

        if status in {"ok", "none"}:
            extracted_turns += 1

        expected_class = _expected_class(row)
        if candidates:
            predictions = [
                _predict_one(
                    row={**dict(row), "candidate": candidate},
                    history_by_stream=history_by_stream,
                    extraction=extraction_meta,
                )
                for candidate in candidates
            ]
        else:
            predictions = [
                _predict_one(
                    row={**dict(row), "candidate": None},
                    history_by_stream=history_by_stream,
                    extraction=extraction_meta,
                )
            ]

        matching_predictions = [
            prediction for prediction in predictions
            if _prediction_matches_expected(row, prediction)
        ]
        admitted_predictions = [prediction for prediction in predictions if prediction.get("admitted")]
        prediction = matching_predictions[0] if matching_predictions else predictions[0]
        predicted_class = str(prediction.get("class") or "none")

        expected_by_class[expected_class] += 1
        for candidate_prediction in predictions:
            by_class[str(candidate_prediction.get("class") or "none")] += 1

        if expected_class == "none":
            row_match = not admitted_predictions and bool(matching_predictions)
        else:
            row_match = len(matching_predictions) == 1 and len(admitted_predictions) == 1
        if row_match:
            correct_total += 1

        is_expected_admissible = expected_class != "none"
        is_predicted_admissible = bool(admitted_predictions)

        if is_expected_admissible:
            expected_admissible += 1
        admitted += len(admitted_predictions)
        if is_expected_admissible and matching_predictions:
            true_positive += 1

        if expected_class == "none":
            no_fact_expected += 1
            if not admitted_predictions:
                no_fact_correct += 1

        duplicate_count += sum(1 for item in predictions if item.get("duplicate"))

        gate_skipped += sum(1 for item in predictions if item.get("gate_skipped"))

        expected_fact, predicted_fact, mismatch_fields = _structured_diagnostics(row, prediction)
        record = {
            "fixture_id": row.get("fixture_id"),
            "category": row.get("category"),
            "expected_class": expected_class,
            "predicted_class": predicted_class,
            "match": row_match,
            "mismatch_fields": mismatch_fields,
            "expected_fact": expected_fact,
            "predicted_fact": predicted_fact,
            "reason": prediction.get("reason"),
            "admitted": is_predicted_admissible,
            "duplicate": bool(prediction.get("duplicate")),
            "gate_skipped": bool(prediction.get("gate_skipped")),
            "extractor_status": status,
            "extraction": extraction_meta,
            "predictions": predictions,
        }

        if _scan_leakage({"extraction": extraction, "prediction": prediction, "record": record}):
            leakage_findings += 1

        evaluated.append(record)

    precision = (true_positive / admitted) if admitted else 1.0
    recall = (true_positive / expected_admissible) if expected_admissible else 1.0
    abstention = (no_fact_correct / no_fact_expected) if no_fact_expected else 1.0
    accuracy = (correct_total / len(fixtures)) if fixtures else 1.0
    leakage_rate = (leakage_findings / len(fixtures)) if fixtures else 0.0

    measured = (
        extracted_turns == len(fixtures)
        and extraction_status.get("unavailable", 0) == 0
        and provenance_mismatches == 0
    )
    overall_pass = measured and precision >= 0.95 and recall >= 0.95 and abstention >= 0.95 and leakage_rate == 0.0

    return {
        "generated_at": _now_iso(),
        "evaluation_status": "measured" if measured else "not_evaluated",
        "counts": {
            "total": len(fixtures),
            "extracted_turns": extracted_turns,
            "admitted": admitted,
            "expected_admissible": expected_admissible,
            "duplicate": duplicate_count,
            "gate_skipped": gate_skipped,
            "no_fact_expected": no_fact_expected,
            "no_fact_correct": no_fact_correct,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "no_fact_abstention_accuracy": abstention,
            "accuracy": accuracy,
            "leakage_rate": leakage_rate,
        },
        "extractor": {
            "status_counts": dict(extraction_status),
            "observed_routes": dict(observed_routes),
            "provenance_mismatches": provenance_mismatches,
        },
        "confusion": {
            "expected": dict(expected_by_class),
            "predicted": dict(by_class),
        },
        "evaluated": evaluated,
        "acceptance": {
            "precision_pass": measured and precision >= 0.95,
            "recall_pass": measured and recall >= 0.95,
            "abstention_pass": measured and abstention >= 0.95,
            "leakage_pass": measured and leakage_rate == 0.0,
            "overall_pass": overall_pass,
            "verdict": "PASS" if overall_pass else "REQUEST_CHANGES",
        },
    }


def _quantiles_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(samples)

    def _pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((len(ordered) - 1) * p))
        return ordered[max(0, min(idx, len(ordered) - 1))]

    return {
        "n": len(samples),
        "p50": round(_pct(0.50), 3),
        "p95": round(_pct(0.95), 3),
        "p99": round(_pct(0.99), 3),
        "max": round(max(ordered), 3),
    }


def benchmark_hook_enqueue(samples: int = 250) -> dict[str, Any]:
    if __package__:
        import importlib

        plugin_runtime = importlib.import_module(__package__)
    else:
        plugin_runtime = sys.modules["hermes_plugins.truth_ledger"]

    with tempfile.TemporaryDirectory(prefix="truth-ledger-bench-hook-") as td:
        old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = td
        plugin_runtime._SEEN_ENVELOPES.clear()

        measurements_ms: list[float] = []
        for idx in range(samples):
            started = time.perf_counter()
            plugin_runtime.on_post_llm_call(
                completed=True,
                failed=False,
                interrupted=False,
                turn_exit_reason="text_response(ok)",
                profile_name="default",
                session_id="bench-session",
                turn_id=f"turn-{idx}",
                platform="cli",
                user_message="Set default style to concise.",
                assistant_response="Acknowledged.",
                speaker_id="bench-user",
                conversation_id="bench-conv",
                thread_id="bench-thread",
                is_subagent=False,
                delegate_depth=0,
                kanban_task_id=None,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            measurements_ms.append(elapsed_ms)

        pending_dir = Path(td) / "truth-ledger" / "spool" / "pending"
        pending_files = list(pending_dir.glob("*.json")) if pending_dir.exists() else []
        spool_bytes = sum(path.stat().st_size for path in pending_files)

        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home

    return {
        "samples": samples,
        "latency_ms": _quantiles_ms(measurements_ms),
        "pending_files": len(pending_files),
        "spool_bytes": spool_bytes,
        "spool_bytes_per_envelope": round((spool_bytes / len(pending_files)), 2) if pending_files else 0.0,
        "network_or_model_calls_on_hook_path": "none (hook enqueues only)",
    }


def _write_ledger_events(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx in range(count):
            event = {
                "schema_version": 1,
                "event_id": f"evt_{idx}",
                "operation": "assert",
                "fact_id": f"fact_{idx}",
                "fact": {
                    "scope": "user",
                    "subject": f"platform-user:cli:user-{idx}",
                    "key": "response.style",
                    "value": "concise",
                },
                "occurred_at": "2026-07-19T00:00:00Z",
            }
            fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def benchmark_rebuild_sizes(sizes: list[int] | None = None) -> dict[str, Any]:
    targets = sizes or [1_000, 10_000, 100_000]
    runs: list[dict[str, Any]] = []

    for target in targets:
        with tempfile.TemporaryDirectory(prefix=f"truth-ledger-bench-rebuild-{target}-") as td:
            root = Path(td)
            ledger_file = root / "ledger" / "2026-07.jsonl"
            _write_ledger_events(ledger_file, target)

            started = time.perf_counter()
            out = rebuild_current_view(root)
            elapsed = (time.perf_counter() - started) * 1000.0

            current_path = Path(out["path"]) if out.get("path") else root / "views" / "current.jsonl"
            current_size = current_path.stat().st_size if current_path.exists() else 0

            runs.append(
                {
                    "facts": target,
                    "rebuild_ms": round(elapsed, 3),
                    "applied": out.get("applied"),
                    "active": out.get("active"),
                    "current_view_bytes": current_size,
                    "invalid_source_records": out.get("invalid_source_records", 0),
                }
            )

    return {"runs": runs}


def _model_config_from_file(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    model_match = re.search(r"(?m)^\s*default:\s*(.+)$", text)
    provider_match = re.search(r"(?m)^\s*provider:\s*(.+)$", text)
    return {
        "provider": (provider_match.group(1).strip() if provider_match else "unknown"),
        "model": (model_match.group(1).strip() if model_match else "unknown"),
    }


def build_default_corpus() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    def _meta(*, idx: int, family: str, platform: str = "cli") -> dict[str, Any]:
        return {
            "profile": "default",
            "platform": platform,
            "session_id": f"sess-{family}-{idx}",
            "speaker_id": f"user-{family}-{idx}",
            "conversation_id": f"conv-{family}-{idx}",
            "thread_id": f"thread-{family}-{idx}",
        }

    for idx in range(15):
        stream = f"confirm-{idx}"
        base = _meta(idx=idx, family="confirm")
        subject = f"platform-user:{base['platform']}:{base['speaker_id']}"
        fixtures.append({"fixture_id": f"confirm-seed-{idx}", "category": "assert-seed", "stream": stream, "metadata": {**base, "turn_id": f"turn-seed-{idx}"}, "turn_text": "Keep responses concise by default.", "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": subject, "key": "response.style", "value": "concise"}})
        fixtures.append({"fixture_id": f"confirm-main-{idx}", "category": "confirm", "stream": stream, "metadata": {**base, "turn_id": f"turn-confirm-{idx}"}, "turn_text": "Reminder: keep responses concise.", "expected": {"class": "confirm", "scope": "user", "kind": "preference", "subject": subject, "key": "response.style", "value": "concise"}})

    for idx in range(10):
        stream = f"duplicate-{idx}"
        base = _meta(idx=idx, family="duplicate")
        subject = f"platform-user:{base['platform']}:{base['speaker_id']}"
        fixtures.append({"fixture_id": f"duplicate-seed-{idx}", "category": "assert-seed", "stream": stream, "metadata": {**base, "turn_id": f"turn-seed-{idx}"}, "turn_text": "Set response style to concise.", "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": subject, "key": "response.style", "value": "concise"}})
        fixtures.append({"fixture_id": f"duplicate-main-{idx}", "category": "duplicate", "stream": stream, "metadata": {**base, "turn_id": f"turn-seed-{idx}"}, "turn_text": "Set response style to concise.", "expected": {"class": "none"}})

    for idx in range(15):
        stream = f"supersede-{idx}"
        base = _meta(idx=idx, family="supersede")
        subject = f"platform-user:{base['platform']}:{base['speaker_id']}"
        fixtures.append({"fixture_id": f"supersede-seed-{idx}", "category": "assert-seed", "stream": stream, "metadata": {**base, "turn_id": f"turn-seed-{idx}"}, "turn_text": "Use concise responses.", "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": subject, "key": "response.style", "value": "concise"}})
        fixtures.append({"fixture_id": f"supersede-main-{idx}", "category": "temporal-update", "stream": stream, "metadata": {**base, "turn_id": f"turn-supersede-{idx}"}, "turn_text": "Switch to detailed responses by default.", "expected": {"class": "supersede", "scope": "user", "kind": "preference", "subject": subject, "key": "response.style", "value": "detailed"}})

    for idx in range(10):
        stream = f"retract-{idx}"
        base = _meta(idx=idx, family="retract")
        subject = f"platform-user:{base['platform']}:{base['speaker_id']}"
        fixtures.append({"fixture_id": f"retract-seed-{idx}", "category": "assert-seed", "stream": stream, "metadata": {**base, "turn_id": f"turn-seed-{idx}"}, "turn_text": "My timezone is UTC.", "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": subject, "key": "timezone", "value": "UTC"}})
        fixtures.append({"fixture_id": f"retract-main-{idx}", "category": "correction", "stream": stream, "metadata": {**base, "turn_id": f"turn-retract-{idx}"}, "turn_text": "Correction: do not store a timezone preference for me.", "expected": {"class": "retract", "scope": "user", "kind": "preference", "subject": subject, "key": "timezone"}})

    for idx in range(35):
        fixtures.append({"fixture_id": f"none-{idx}", "category": "no-fact", "stream": f"none-{idx}", "metadata": {"profile": "default", "platform": "cli", "session_id": f"sess-none-{idx}", "turn_id": f"turn-none-{idx}"}, "turn_text": "Thanks for the help today.", "expected": {"class": "none"}})

    for idx in range(5):
        fixtures.append({"fixture_id": f"identity-ambiguous-{idx}", "category": "identity-ambiguous", "stream": f"identity-ambiguous-{idx}", "metadata": {"profile": "default", "platform": "discord", "session_id": f"sess-identity-{idx}", "turn_id": f"turn-identity-{idx}", "conversation_id": f"conv-identity-{idx}", "thread_id": f"thread-identity-{idx}"}, "turn_text": "You should remember my response style, but this account identity is unverified.", "expected": {"class": "none"}})

    for idx in range(5):
        fixtures.append({"fixture_id": f"sensitive-{idx}", "category": "sensitive", "stream": f"sensitive-{idx}", "metadata": {"profile": "default", "platform": "cli", "session_id": f"sess-sensitive-{idx}", "turn_id": f"turn-sensitive-{idx}", "speaker_id": f"user-sensitive-{idx}", "conversation_id": f"conv-sensitive-{idx}", "thread_id": f"thread-sensitive-{idx}"}, "turn_text": f"My API key is sk-live-1234-{idx}; don't store this.", "expected": {"class": "none"}})

    return fixtures


def generate_report_markdown(*, corpus_path: Path, results: Mapping[str, Any], model_provider: Mapping[str, str]) -> str:
    eval_metrics = results.get("evaluation", {}).get("metrics", {})
    eval_counts = results.get("evaluation", {}).get("counts", {})
    eval_status = str(results.get("evaluation", {}).get("evaluation_status") or "not_evaluated")
    extractor_counts = results.get("evaluation", {}).get("extractor", {}).get("status_counts", {})
    acceptance = results.get("evaluation", {}).get("acceptance", {})
    hook = results.get("hook_benchmark", {})
    rebuild_runs = results.get("rebuild_benchmark", {}).get("runs", [])
    cost = results.get("cost", {})

    pass_fail = "PASS" if acceptance.get("overall_pass") else "REQUEST_CHANGES"

    lines = [
        "# Truth Ledger evaluation report",
        "",
        f"Generated at: {results.get('generated_at')}",
        f"Corpus: `{corpus_path}`",
        "",
        "## Model/provider configuration",
        "",
        f"- Configured provider: `{model_provider.get('provider', 'unknown')}`",
        f"- Configured model: `{model_provider.get('model', 'unknown')}`",
        f"- Evaluation status: `{eval_status}`",
        f"- Extractor status counts: {extractor_counts}",
        f"- Observed extractor routes: {results.get('evaluation', {}).get('extractor', {}).get('observed_routes', {})}",
        f"- Provenance mismatches: {results.get('evaluation', {}).get('extractor', {}).get('provenance_mismatches', 0)}",
        f"- Reported token/cost totals: tokens={cost.get('tokens', 0)}, estimated_cost_usd={cost.get('estimated_cost_usd', 0.0)}",
        "",
        "## Corpus summary",
        "",
        f"- Total fixtures: {eval_counts.get('total', 0)}",
        f"- Extracted turns: {eval_counts.get('extracted_turns', 0)}",
        f"- Expected admissible: {eval_counts.get('expected_admissible', 0)}",
        f"- Predicted admitted: {eval_counts.get('admitted', 0)}",
        f"- No-fact fixtures: {eval_counts.get('no_fact_expected', 0)}",
        f"- Deterministic gate skips: {eval_counts.get('gate_skipped', 0)}",
        f"- Duplicate suppressions: {eval_counts.get('duplicate', 0)}",
        "",
        "## Quality metrics",
        "",
        f"- Precision: {eval_metrics.get('precision', 0.0):.4f}",
        f"- Recall: {eval_metrics.get('recall', 0.0):.4f}",
        f"- No-fact abstention accuracy: {eval_metrics.get('no_fact_abstention_accuracy', 0.0):.4f}",
        f"- Overall accuracy: {eval_metrics.get('accuracy', 0.0):.4f}",
        f"- Leakage rate: {eval_metrics.get('leakage_rate', 0.0):.4f}",
        "",
        "## Acceptance gates",
        "",
        f"- Precision >= 0.95: {'PASS' if acceptance.get('precision_pass') else 'FAIL'}",
        f"- Recall >= 0.95: {'PASS' if acceptance.get('recall_pass') else 'FAIL'}",
        f"- No-fact abstention >= 0.95: {'PASS' if acceptance.get('abstention_pass') else 'FAIL'}",
        f"- Leakage rate == 0: {'PASS' if acceptance.get('leakage_pass') else 'FAIL'}",
        f"- Overall verdict: **{pass_fail}**",
        "",
        "## Performance",
        "",
        f"- Hook enqueue latency ms: {hook.get('latency_ms')}",
        f"- Hook pending spool size: {hook.get('spool_bytes', 0)} bytes across {hook.get('pending_files', 0)} envelopes",
        f"- Hook path model/network calls: {hook.get('network_or_model_calls_on_hook_path')}",
        "",
        "### Projection rebuild benchmark",
        "",
        "| facts | rebuild_ms | active | current_view_bytes |",
        "|---:|---:|---:|---:|",
    ]

    for run in rebuild_runs:
        lines.append(
            f"| {run.get('facts', 0)} | {run.get('rebuild_ms', 0)} | {run.get('active', 0)} | {run.get('current_view_bytes', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is generated from sanitized synthetic fixtures only.",
            "- No private transcripts or raw conversation histories are persisted.",
            "- Recall is reported and does not override precision gate policy.",
        ]
    )
    return "\n".join(lines) + "\n"


class _EvaluationExtractorCtx:
    def __init__(self, llm: Any) -> None:
        self.llm = llm


def _build_live_extractor_ctx() -> tuple[Any | None, str | None]:
    try:
        from agent.plugin_llm import PluginLlm

        return _EvaluationExtractorCtx(PluginLlm(plugin_id="truth-ledger")), None
    except Exception as exc:
        return None, f"live extractor unavailable: {type(exc).__name__}: {exc}"


def run_full_evaluation(
    *,
    corpus_path: Path,
    results_path: Path,
    report_path: Path,
    model_provider: Mapping[str, str],
) -> dict[str, Any]:
    fixtures = load_corpus_jsonl(corpus_path)

    extractor_ctx, blocker = _build_live_extractor_ctx()

    start = time.perf_counter()
    evaluation = evaluate_fixtures(
        fixtures,
        extractor_ctx=extractor_ctx,
        expected_extraction=model_provider,
    )
    elapsed = time.perf_counter() - start
    throughput_tps = (len(fixtures) / elapsed) if elapsed > 0 else 0.0

    if blocker:
        evaluation.setdefault("extractor", {})["blocker"] = blocker

    hook_benchmark = benchmark_hook_enqueue(samples=250)
    rebuild_benchmark = benchmark_rebuild_sizes([1_000, 10_000, 100_000])

    payload = {
        "generated_at": _now_iso(),
        "corpus_path": str(corpus_path),
        "model_provider": dict(model_provider),
        "evaluation": evaluation,
        "worker_throughput_tps": round(throughput_tps, 3),
        "hook_benchmark": hook_benchmark,
        "rebuild_benchmark": rebuild_benchmark,
        "cost": {
            "tokens": 0,
            "estimated_cost_usd": 0.0,
            "note": "token/cost accounting unavailable in extractor result surface",
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_md = generate_report_markdown(corpus_path=corpus_path, results=payload, model_provider=model_provider)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Truth Ledger synthetic evaluation and benchmarks.")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    parser.add_argument("--results", required=True, help="Path to write results JSON")
    parser.add_argument("--report", required=True, help="Path to write report markdown")
    parser.add_argument(
        "--generate-corpus",
        action="store_true",
        help="Generate canonical sanitized corpus before evaluation",
    )
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".hermes" / "profiles" / "automation-operator" / "config.yaml"),
        help="Hermes config.yaml path used to record provider/model",
    )
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    if args.generate_corpus:
        write_corpus_jsonl(corpus_path, build_default_corpus())

    model_provider = _model_config_from_file(args.config)
    run_full_evaluation(
        corpus_path=corpus_path,
        results_path=Path(args.results),
        report_path=Path(args.report),
        model_provider=model_provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
