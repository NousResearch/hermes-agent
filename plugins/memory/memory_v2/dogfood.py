"""Repeatable Memory v2 dogfood harness.

The helpers in this module make dogfood runs profile-safe and repeatable:
``fresh=True`` removes the target profile's ``memory_v2`` tree before importing
core memory and running scenarios, so test artifacts do not accumulate across
runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import MemoryV2Provider
from .core_importer import import_core_memory_from_context_files
from .evals.baselines import MemoryV2Baseline, NoMemoryBaseline, RawFTSBaseline
from .evals.datasets import load_eval_dataset
from .evals.runners import run_eval

DEFAULT_CORE_BUDGET = 24
DEFAULT_CATEGORY_MINIMUMS = {
    "user": 5,
    "assistant_identity": 5,
    "environment": 2,
    "operating_rule": 4,
}
SECRET_SENTINEL = "DOGFOOD_SECRET_12345"


def prepare_dogfood_profile(
    *,
    target_hermes_home: str | Path,
    source_hermes_home: str | Path,
    fresh: bool = False,
    core_budget: int = DEFAULT_CORE_BUDGET,
    category_minimums: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prepare a dogfood profile with Memory v2 enabled and pruned core imported.

    Args:
        target_hermes_home: Profile directory to exercise.
        source_hermes_home: Profile/root directory containing context files to import.
        fresh: If true, delete only ``target_hermes_home/memory_v2`` before setup.
        core_budget: Prompt-core record budget for context import.
        category_minimums: Per-category minimums protected during pruning.
    """
    target_home = Path(target_hermes_home).expanduser().resolve()
    source_home = Path(source_hermes_home).expanduser().resolve()
    target_home.mkdir(parents=True, exist_ok=True)

    memory_dir = target_home / "memory_v2"
    fresh_reset = False
    if fresh and memory_dir.exists():
        shutil.rmtree(memory_dir)
        fresh_reset = True
    elif fresh:
        fresh_reset = True

    _enable_memory_v2_config(target_home / "config.yaml")
    import_report = import_core_memory_from_context_files(
        target_hermes_home=target_home,
        source_hermes_home=source_home,
        core_budget=core_budget,
        category_minimums=category_minimums or DEFAULT_CATEGORY_MINIMUMS,
        archive_pruned=True,
    )

    provider = MemoryV2Provider()
    provider.initialize("dogfood-prepare", hermes_home=target_home, platform="dogfood", agent_context="primary")
    core_counts = _core_counts(provider)
    return {
        "success": True,
        "target_hermes_home": str(target_home),
        "source_hermes_home": str(source_home),
        "fresh_reset": fresh_reset,
        "config_provider": _read_config(target_home / "config.yaml").get("memory", {}).get("provider"),
        "import_report": import_report,
        "core_counts": core_counts,
    }


def run_dogfood_scenario_tests(
    *,
    target_hermes_home: str | Path,
    source_hermes_home: str | Path,
    default_hermes_home: str | Path | None = None,
    fresh: bool = False,
    core_budget: int = DEFAULT_CORE_BUDGET,
    category_minimums: dict[str, int] | None = None,
    run_local_eval: bool = False,
) -> dict[str, Any]:
    """Run high-signal dogfood scenarios against a real profile-local store."""
    target_home = Path(target_hermes_home).expanduser().resolve()
    default_home = Path(default_hermes_home).expanduser().resolve() if default_hermes_home else Path.home() / ".hermes"
    default_memory_v2_signature = _tree_signature(default_home / "memory_v2")
    prepare_report = prepare_dogfood_profile(
        target_hermes_home=target_home,
        source_hermes_home=source_hermes_home,
        fresh=fresh,
        core_budget=core_budget,
        category_minimums=category_minimums,
    )

    session_id = f"dogfood-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    results: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        results.append({"name": name, "ok": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"{name}: {details}")

    def tool(provider: MemoryV2Provider, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return json.loads(provider.handle_tool_call(name, args or {}))

    def newest_candidate(provider: MemoryV2Provider, previous_count: int) -> dict[str, Any]:
        candidates = provider.store.list_candidates()
        check("candidate_count_incremented", len(candidates) > previous_count, {"before": previous_count, "after": len(candidates)})
        return candidates[-1].to_dict()

    provider = MemoryV2Provider()
    provider.initialize(session_id, hermes_home=target_home, platform="dogfood-test", agent_context="primary")
    initial_counts = _scenario_counts(provider)
    local_eval_report: dict[str, Any] | None = None

    try:
        config = _read_config(target_home / "config.yaml")
        check("profile_config_uses_memory_v2", config.get("memory", {}).get("provider") == "memory_v2", config.get("memory", {}))

        status = tool(provider, "memory_v2_status")
        check("provider_initialized_in_dogfood_profile", status["success"] and status["initialized"], status)
        check("provider_base_dir_is_profile_local", target_home / "memory_v2" == provider.base_dir, {"display": status["base_dir"], "actual": str(provider.base_dir)})
        check(
            "default_profile_memory_v2_store_unchanged",
            _tree_signature(default_home / "memory_v2") == default_memory_v2_signature,
            str(default_home / "memory_v2"),
        )

        core_counts = _core_counts(provider)
        check("core_import_record_budget", len(provider.store.list_core_memory_records()) == core_budget, core_counts)
        minimums = category_minimums or DEFAULT_CATEGORY_MINIMUMS
        check(
            "core_import_category_minimums",
            all(core_counts.get(category, 0) >= minimum for category, minimum in minimums.items()),
            core_counts,
        )
        prompt = provider.system_prompt_block()
        check("system_prompt_block_bounded", 0 < len(prompt) <= 1200, {"chars": len(prompt), "preview": prompt[:240]})
        check("system_prompt_block_uses_source_refs_not_paths", "source_refs=" in prompt and str(target_home) not in prompt, prompt[:500])

        provider.on_turn_start(
            1,
            "We are dogfooding Memory v2 and need to preserve active focus through topic switches.",
            model="dogfood-model",
            remaining_tokens=12345,
        )
        working_packet = provider.prefetch("What are we working on / current task / left off?")
        check("working_memory_prefetch_returns_current_task", "route: current_task" in working_packet, working_packet)
        check("working_memory_contains_current_focus", "dogfooding Memory v2" in working_packet, working_packet)

        before = len(provider.store.list_candidates())
        preference_text = "remember that I prefer Memory v2 dogfood reports to include source-grounding checks and concrete failure IDs."
        provider.sync_turn(preference_text, "Noted; I will queue that as a Memory v2 candidate.", session_id=session_id)
        pref_candidate = newest_candidate(provider, before)
        check("preference_write_gate_creates_pending_candidate", pref_candidate["type"] == "preference" and pref_candidate["gate_decision"] == "pending", pref_candidate)
        check("preference_candidate_has_source_ref", bool(pref_candidate.get("source_refs")), pref_candidate)
        pref_source = tool(provider, "memory_v2_show_source", {"id": pref_candidate["id"]})
        check("candidate_source_lookup_falls_back_to_raw_event", pref_source["success"] and pref_source["sources"], pref_source)

        promoted = tool(provider, "memory_v2_promote", {"candidate_id": pref_candidate["id"]})
        check("manual_promotion_succeeds_with_source_validation", promoted["success"] and promoted["promoted"] == 1, promoted)
        promoted_id = promoted["promoted_ids"][0]
        search = tool(provider, "memory_v2_search", {"query": "source-grounding checks concrete failure IDs", "limit": 5})
        check("promoted_preference_is_searchable", any(r["id"] == promoted_id and r.get("status") == "active" for r in search["results"]), search)
        recalled = provider.prefetch("What does Alex prefer about Memory v2 dogfood reports?")
        check("promoted_preference_prefetch_recall", promoted_id in recalled and "type: preference" in recalled, recalled)
        shown = tool(provider, "memory_v2_show_source", {"id": promoted_id})
        check("promoted_memory_has_resolvable_sources", shown["success"] and shown["sources"] and not shown.get("missing_source_refs"), shown)

        before = len(provider.store.list_candidates())
        provider.sync_turn(
            "remember that Hermes is not running on WSL; it is running on macOS instead of Linux.",
            "Queued for contradiction review, not silently overwritten.",
            session_id=session_id,
        )
        conflict_candidate = newest_candidate(provider, before)
        check(
            "conflict_candidate_requires_review_not_silent_overwrite",
            conflict_candidate["type"] == "environment"
            and conflict_candidate["gate_decision"] == "pending"
            and ("conflict" in conflict_candidate["promotion_reason"].lower() or "supersede" in conflict_candidate["promotion_reason"].lower()),
            conflict_candidate,
        )

        before = len(provider.store.list_candidates())
        provider.sync_turn(
            "remember to follow up on Memory v2 dogfood failure IDs tomorrow",
            "I'll track that as an open loop.",
            session_id=session_id,
        )
        loop_candidate = newest_candidate(provider, before)
        check("open_loop_candidate_beats_ephemeral_filter", loop_candidate["proposed_destination"] == "working/open_loops.yaml", loop_candidate)
        loop_promoted = tool(provider, "memory_v2_promote", {"candidate_id": loop_candidate["id"]})
        check("open_loop_manual_route_succeeds", loop_promoted["success"] and loop_promoted["promoted_ids"], loop_promoted)
        loop_id = loop_promoted["promoted_ids"][0]
        loops = provider.store.list_open_loops(status="open")
        check("open_loop_is_in_working_memory_not_semantic", any(loop.get("id") == loop_id for loop in loops), loops)
        loop_packet = provider.prefetch("What open loops are pending?")
        check("open_loop_prefetch_surfaces_pending_loop", loop_id in loop_packet and "failure IDs" in loop_packet, loop_packet)

        before = len(provider.store.list_candidates())
        provider.sync_turn(f"remember that token is {SECRET_SENTINEL}", "Queued securely.", session_id=session_id)
        redacted_candidate = newest_candidate(provider, before)
        raw_dump = json.dumps(provider.store.read_raw_events(limit=5), sort_keys=True)
        candidate_dump = json.dumps(redacted_candidate, sort_keys=True)
        check("sensitive_raw_archive_redacted", SECRET_SENTINEL not in raw_dump and "[REDACTED]" in raw_dump, raw_dump)
        check("sensitive_candidate_redacted", SECRET_SENTINEL not in candidate_dump and "[REDACTED]" in candidate_dump, candidate_dump)
        check("sensitive_candidate_auto_archived", redacted_candidate["gate_decision"] == "archived_only", redacted_candidate)

        provider.on_session_end([
            {"role": "user", "content": "Dogfood Memory v2."},
            {"role": "assistant", "content": "Ran tests."},
        ])
        episodic_files = list((target_home / "memory_v2" / "episodic" / "sessions").glob("*.yaml"))
        check("session_end_archives_episode", bool(episodic_files), [p.name for p in episodic_files[-5:]])

        if run_local_eval:
            local_eval_report = _run_local_eval_fixture()
            check(
                "local_eval_fixture_runs",
                local_eval_report["dataset"] == "local_memory_eval_v1"
                and bool(local_eval_report.get("summary", {}).get("memory_v2")),
                local_eval_report.get("summary"),
            )

        report = {
            "success": all(item["ok"] for item in results),
            "session_id": session_id,
            "dogfood_home": str(target_home),
            "fresh_reset": bool(fresh),
            "prepare_report": prepare_report,
            "initial_counts": initial_counts,
            "final_counts": _scenario_counts(provider),
            "core_counts": core_counts,
            "promoted_preference_id": promoted_id,
            "open_loop_id": loop_id,
            "results": results,
        }
        if local_eval_report is not None:
            report["local_eval"] = local_eval_report
    except Exception as exc:
        report = {
            "success": False,
            "session_id": session_id,
            "dogfood_home": str(target_home),
            "fresh_reset": bool(fresh),
            "prepare_report": prepare_report,
            "initial_counts": initial_counts,
            "final_counts": _scenario_counts(provider),
            "error": repr(exc),
            "results": results,
        }
        if local_eval_report is not None:
            report["local_eval"] = local_eval_report
        _write_report(target_home, report, failed=True)
        raise

    report_path = _write_report(target_home, report, failed=False)
    report["report_path"] = str(report_path)
    return report


def _enable_memory_v2_config(config_path: Path) -> None:
    config = _read_config(config_path)
    memory = dict(config.get("memory") or {})
    memory["memory_enabled"] = True
    memory.setdefault("user_profile_enabled", True)
    memory["provider"] = "memory_v2"
    config["memory"] = memory
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _core_counts(provider: MemoryV2Provider) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in provider.store.list_core_memory_records():
        category = getattr(record.category, "value", str(record.category))
        counts[category] = counts.get(category, 0) + 1
    return counts


def _scenario_counts(provider: MemoryV2Provider) -> dict[str, int]:
    return {
        "raw_events": provider.store.count_raw_events(),
        "pending_candidates": provider.store.count_pending_candidates(),
        "open_loops": len(provider.store.list_open_loops(status="open")),
        "memory_items": len(provider.store.list_memory_items()),
    }


def _run_local_eval_fixture() -> dict[str, Any]:
    """Run the packaged local eval fixture with deterministic local baselines only."""
    fixture_path = Path(__file__).parent / "evals" / "fixtures" / "local_memory_eval_v1.yaml"
    dataset = load_eval_dataset(fixture_path)
    with tempfile.TemporaryDirectory(prefix="memory-v2-dogfood-eval-") as temp_dir:
        workdir = Path(temp_dir)
        report = run_eval(
            dataset,
            baselines=[
                NoMemoryBaseline(),
                RawFTSBaseline(workdir / "raw_fts" / "raw.sqlite"),
                MemoryV2Baseline(workdir / "memory_v2"),
            ],
        )
    payload = report.to_dict()
    payload["fixture_path"] = fixture_path.as_posix()
    payload["baselines"] = sorted(payload.get("summary", {}).keys())
    payload["external_baselines"] = []
    return payload


def _tree_signature(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    if path.is_file():
        return [(path.name, path.stat().st_size)]
    signature: list[tuple[str, int]] = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            signature.append((child.relative_to(path).as_posix(), child.stat().st_size))
    return signature


def _write_report(target_home: Path, report: dict[str, Any], *, failed: bool) -> Path:
    reports_dir = target_home / "memory_v2" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_FAILED" if failed else ""
    report_path = reports_dir / f"dogfood_report_{report['session_id']}{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run repeatable Memory v2 dogfood scenarios against a profile.")
    parser.add_argument("--target-home", default="~/.hermes/profiles/memory-v2-dogfood", help="Dogfood Hermes profile path.")
    parser.add_argument("--source-home", default="~/.hermes", help="Context-file source Hermes home/profile path.")
    parser.add_argument("--default-home", default="~/.hermes", help="Default profile path used only for isolation checks.")
    parser.add_argument("--fresh", action="store_true", help="Delete target memory_v2 before importing core/running scenarios.")
    parser.add_argument("--core-budget", type=int, default=DEFAULT_CORE_BUDGET, help="Prompt-core import budget.")
    parser.add_argument(
        "--run-local-eval",
        action="store_true",
        help="Run the packaged local deterministic eval fixture and include its summary in the dogfood report.",
    )
    args = parser.parse_args(argv)

    report = run_dogfood_scenario_tests(
        target_hermes_home=args.target_home,
        source_hermes_home=args.source_home,
        default_hermes_home=args.default_home,
        fresh=args.fresh,
        core_budget=args.core_budget,
        run_local_eval=args.run_local_eval,
    )
    print(
        json.dumps(
            {
                "success": report["success"],
                "report_path": report["report_path"],
                "final_counts": report["final_counts"],
                "local_eval_summary": report.get("local_eval", {}).get("summary"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
