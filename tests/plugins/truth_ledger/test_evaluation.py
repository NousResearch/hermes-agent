from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_evaluation_module():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = repo_root / "plugins" / "truth-ledger"
    module_path = plugin_dir / "evaluation.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger.evaluation",
        module_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None
    assert spec.loader is not None

    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if "hermes_plugins.truth_ledger" not in sys.modules:
        pkg = types.ModuleType("hermes_plugins.truth_ledger")
        pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.truth_ledger"] = pkg

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.truth_ledger"
    sys.modules["hermes_plugins.truth_ledger.evaluation"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_fixtures_reports_precision_abstention_and_leakage_zero():
    mod = _load_evaluation_module()

    extracted = {
        "scope": "user",
        "kind": "preference",
        "subject": "platform-user:cli:user-1",
        "key": "response.style",
        "value": "concise",
        "operation": "assert",
        "evidence_type": "user_stated",
    }

    calls = []

    def _extractor(*, row, envelope):
        calls.append({"row": row["fixture_id"], "envelope": envelope})
        if row["fixture_id"] == "f-assert-1":
            return {
                "status": "ok",
                "facts": [extracted],
                "reason": None,
                "extraction": {"provider": "test-provider", "model": "test-model"},
            }
        return {
            "status": "none",
            "facts": [],
            "reason": "none",
            "extraction": {"provider": "test-provider", "model": "test-model"},
        }

    fixtures = [
        {
            "fixture_id": "f-assert-1",
            "stream": "s1",
            "metadata": {
                "profile": "default",
                "platform": "cli",
                "session_id": "sess-1",
                "turn_id": "turn-1",
                "speaker_id": "user-1",
                "conversation_id": "conv-1",
                "thread_id": "thread-1",
            },
            "turn_text": "User: Keep responses concise.",
            "expected": {
                "class": "assert",
                "scope": "user",
                "kind": "preference",
                "subject": "platform-user:cli:user-1",
                "key": "response.style",
                "value": "concise",
            },
        },
        {
            "fixture_id": "f-none-1",
            "stream": "s2",
            "metadata": {
                "profile": "default",
                "platform": "cli",
                "session_id": "sess-2",
                "turn_id": "turn-2",
                "conversation_id": "conv-2",
            },
            "turn_text": "User: Thanks!",
            "expected": {"class": "none"},
        },
    ]

    out = mod.evaluate_fixtures(fixtures, extractor_fn=_extractor)

    assert out["counts"]["total"] == 2
    assert out["counts"]["extracted_turns"] == 2
    assert out["counts"]["admitted"] == 1
    assert out["counts"]["expected_admissible"] == 1
    assert out["metrics"]["precision"] == 1.0
    assert out["metrics"]["recall"] == 1.0
    assert out["metrics"]["no_fact_abstention_accuracy"] == 1.0
    assert out["metrics"]["leakage_rate"] == 0.0
    assert out["acceptance"]["recall_pass"] is True
    assert out["evaluation_status"] == "measured"
    assert len(calls) == 2


def test_corpus_round_trip_jsonl(tmp_path):
    mod = _load_evaluation_module()

    fixtures = [
        {"fixture_id": "f1", "stream": "s", "metadata": {}, "turn_text": "Thanks", "expected": {"class": "none"}},
        {"fixture_id": "f2", "stream": "s", "metadata": {}, "turn_text": "Noted", "expected": {"class": "none"}},
    ]
    corpus_path = tmp_path / "corpus.jsonl"

    mod.write_corpus_jsonl(corpus_path, fixtures)
    loaded = mod.load_corpus_jsonl(corpus_path)

    assert [row["fixture_id"] for row in loaded] == ["f1", "f2"]
    assert all("expected" in row for row in loaded)

    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_build_default_corpus_has_turn_text_and_required_coverage_classes():
    mod = _load_evaluation_module()

    corpus = mod.build_default_corpus()

    assert len(corpus) >= 100
    assert all(isinstance(row.get("turn_text"), str) and row["turn_text"].strip() for row in corpus)
    assert all("candidate" not in row for row in corpus)

    categories = {str(row.get("category") or "") for row in corpus}
    assert {"no-fact", "correction", "duplicate", "temporal-update", "identity-ambiguous", "sensitive"}.issubset(categories)


def test_build_default_corpus_user_subjects_match_trusted_metadata():
    mod = _load_evaluation_module()

    for row in mod.build_default_corpus():
        expected = row.get("expected", {})
        if expected.get("class") == "none" or expected.get("scope") != "user":
            continue
        metadata = row["metadata"]
        assert expected["subject"] == (
            f"platform-user:{metadata['platform']}:{metadata['speaker_id']}"
        )


def test_build_default_corpus_duplicate_rows_are_same_source_replays():
    mod = _load_evaluation_module()
    corpus = mod.build_default_corpus()

    for idx in range(10):
        seed = next(row for row in corpus if row["fixture_id"] == f"duplicate-seed-{idx}")
        replay = next(row for row in corpus if row["fixture_id"] == f"duplicate-main-{idx}")
        assert replay["metadata"] == seed["metadata"]


def test_retract_preserves_active_fact_kind_when_extractor_labels_correction():
    mod = _load_evaluation_module()
    base = {
        "profile": "default", "platform": "cli", "session_id": "sess-retract",
        "speaker_id": "user-1", "conversation_id": "conv-1", "thread_id": "thread-1",
    }
    fixtures = [
        {
            "fixture_id": "seed", "stream": "retract",
            "metadata": {**base, "turn_id": "turn-seed"},
            "turn_text": "My timezone is UTC.",
            "expected": {
                "class": "assert", "scope": "user", "kind": "preference",
                "subject": "platform-user:cli:user-1", "key": "timezone", "value": "UTC",
            },
        },
        {
            "fixture_id": "retract", "stream": "retract",
            "metadata": {**base, "turn_id": "turn-retract"},
            "turn_text": "Correction: do not store a timezone preference for me.",
            "expected": {
                "class": "retract", "scope": "user", "kind": "preference",
                "subject": "platform-user:cli:user-1", "key": "timezone",
            },
        },
    ]

    def _extractor(*, row, envelope):
        del envelope
        if row["fixture_id"] == "seed":
            return {
                "status": "ok",
                "facts": [{
                    "scope": "user", "kind": "preference",
                    "subject": "platform-user:cli:user-1", "key": "timezone",
                    "value": "UTC", "operation": "assert", "evidence_type": "user_stated",
                }],
            }
        return {
            "status": "ok",
            "facts": [{
                "scope": "user", "kind": "correction",
                "subject": "platform-user:cli:user-1", "key": "timezone",
                "value": None, "operation": "retract", "evidence_type": "user_stated",
            }],
        }

    out = mod.evaluate_fixtures(fixtures, extractor_fn=_extractor)

    assert out["evaluated"][1]["predicted_class"] == "retract"
    assert out["evaluated"][1]["match"] is True


def test_supersede_keeps_new_candidate_kind():
    mod = _load_evaluation_module()
    base = {
        "profile": "default", "platform": "cli", "session_id": "sess-change",
        "speaker_id": "user-1", "conversation_id": "conv-1", "thread_id": "thread-1",
    }
    fixtures = [
        {
            "fixture_id": "seed", "stream": "change",
            "metadata": {**base, "turn_id": "turn-seed"}, "turn_text": "Use concise responses.",
            "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": "platform-user:cli:user-1", "key": "response.style", "value": "concise"},
        },
        {
            "fixture_id": "change", "stream": "change",
            "metadata": {**base, "turn_id": "turn-change"}, "turn_text": "Detailed responses are now required.",
            "expected": {"class": "supersede", "scope": "user", "kind": "constraint", "subject": "platform-user:cli:user-1", "key": "response.style", "value": "detailed"},
        },
    ]

    def _extractor(*, row, envelope):
        del envelope
        value = "concise" if row["fixture_id"] == "seed" else "detailed"
        kind = "preference" if row["fixture_id"] == "seed" else "constraint"
        return {"status": "ok", "facts": [{"scope": "user", "kind": kind, "subject": "platform-user:cli:user-1", "key": "response.style", "value": value, "operation": "assert", "evidence_type": "user_stated"}]}

    out = mod.evaluate_fixtures(fixtures, extractor_fn=_extractor)

    assert out["evaluated"][1]["match"] is True


def test_retry_is_retried_to_a_measured_result():
    mod = _load_evaluation_module()
    fixtures = [{"fixture_id": "retry", "stream": "s", "metadata": {}, "turn_text": "Keep responses concise.", "expected": {"class": "none"}}]
    calls = 0

    def _extractor(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "retry", "facts": [], "retry_delay_ms": 0}
        return {"status": "none", "facts": []}

    out = mod.evaluate_fixtures(fixtures, extractor_fn=_extractor)

    assert calls == 2
    assert out["counts"]["extracted_turns"] == 1
    assert out["evaluation_status"] == "measured"


def test_retry_result_marks_evaluation_not_evaluated():
    mod = _load_evaluation_module()
    fixtures = [{"fixture_id": "retry", "stream": "s", "metadata": {}, "turn_text": "Keep responses concise.", "expected": {"class": "assert"}}]

    out = mod.evaluate_fixtures(fixtures, extractor_fn=lambda **_kwargs: {"status": "retry", "facts": []})

    assert out["counts"]["extracted_turns"] == 0
    assert out["evaluation_status"] == "not_evaluated"


def test_leakage_scans_extracted_candidates():
    mod = _load_evaluation_module()
    fixtures = [{"fixture_id": "leak", "stream": "s", "metadata": {}, "turn_text": "Do not store this secret.", "expected": {"class": "none"}}]
    leaking = {"scope": "user", "kind": "preference", "subject": "user", "key": "secret", "value": "sk-proj-abcdefghijklmnopqrstuvwxyz123456", "operation": "assert", "evidence_type": "user_stated"}

    out = mod.evaluate_fixtures(fixtures, extractor_fn=lambda **_kwargs: {"status": "ok", "facts": [leaking]})

    assert out["metrics"]["leakage_rate"] == 1.0
    assert out["acceptance"]["leakage_pass"] is False


def test_strict_mismatch_records_structured_diagnostics():
    mod = _load_evaluation_module()
    fixtures = [{"fixture_id": "mismatch", "stream": "s", "metadata": {"profile": "default", "platform": "cli", "session_id": "s", "turn_id": "t", "speaker_id": "u"}, "turn_text": "Keep responses concise.", "expected": {"class": "assert", "scope": "user", "kind": "preference", "subject": "platform-user:cli:u", "key": "response.style", "value": "concise"}}]
    extracted = {"scope": "user", "kind": "preference", "subject": "platform-user:cli:u", "key": "response.style", "value": "verbose", "operation": "assert", "evidence_type": "user_stated"}

    out = mod.evaluate_fixtures(fixtures, extractor_fn=lambda **_kwargs: {"status": "ok", "facts": [extracted]})
    row = out["evaluated"][0]

    assert row["match"] is False
    assert row["expected_fact"]["value"] == "concise"
    assert row["predicted_fact"]["value"] == "verbose"
    assert row["mismatch_fields"] == ["value"]


def test_evaluation_records_and_enforces_observed_extractor_provenance():
    mod = _load_evaluation_module()
    fixtures = [{
        "fixture_id": "provenance", "stream": "s", "metadata": {},
        "turn_text": "Thanks.", "expected": {"class": "none"},
    }]

    def _extractor(**_kwargs):
        return {
            "status": "none",
            "facts": [],
            "extraction": {
                "schema_name": "truth-ledger.fact-candidates.v1",
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "prompt_version": 2,
            },
        }

    expected_route = {"provider": "openai-codex", "model": "gpt-5.6-sol"}
    out = mod.evaluate_fixtures(
        fixtures,
        extractor_fn=_extractor,
        expected_extraction=expected_route,
    )

    assert out["evaluation_status"] == "measured"
    assert out["extractor"]["observed_routes"] == {"openai-codex/gpt-5.6-sol": 1}
    assert out["evaluated"][0]["extraction"]["provider"] == "openai-codex"
    assert out["evaluated"][0]["extraction"]["model"] == "gpt-5.6-sol"
    report = mod.generate_report_markdown(
        corpus_path=Path("corpus.jsonl"),
        results={"generated_at": out["generated_at"], "evaluation": out},
        model_provider=expected_route,
    )
    assert "Observed extractor routes: {'openai-codex/gpt-5.6-sol': 1}" in report
    assert "Provenance mismatches: 0" in report

    wrong = mod.evaluate_fixtures(
        fixtures,
        extractor_fn=_extractor,
        expected_extraction={"provider": "openai-api", "model": "gpt-5.6-sol"},
    )
    assert wrong["evaluation_status"] == "not_evaluated"
    assert wrong["extractor"]["provenance_mismatches"] == 1
    assert wrong["acceptance"]["overall_pass"] is False


def test_all_extracted_facts_are_scored_and_spurious_fact_lowers_precision():
    mod = _load_evaluation_module()
    expected = {
        "scope": "user", "kind": "preference",
        "subject": "platform-user:cli:u", "key": "response.style",
        "value": "concise", "operation": "assert", "evidence_type": "user_stated",
    }
    spurious = {
        "scope": "project", "kind": "project", "subject": "projects:other",
        "key": "status", "value": "active", "operation": "assert",
        "evidence_type": "user_stated",
    }
    fixtures = [{
        "fixture_id": "multi", "stream": "s",
        "metadata": {
            "profile": "default", "platform": "cli", "session_id": "s",
            "turn_id": "t", "speaker_id": "u",
        },
        "turn_text": "Keep responses concise.",
        "expected": {
            "class": "assert", "scope": "user", "kind": "preference",
            "subject": "platform-user:cli:u", "key": "response.style", "value": "concise",
        },
    }]

    out = mod.evaluate_fixtures(
        fixtures,
        extractor_fn=lambda **_kwargs: {"status": "ok", "facts": [expected, spurious]},
    )

    assert out["counts"]["admitted"] == 2
    assert out["metrics"]["precision"] == 0.5
    assert out["metrics"]["recall"] == 1.0
    assert out["evaluated"][0]["match"] is False
    assert len(out["evaluated"][0]["predictions"]) == 2
