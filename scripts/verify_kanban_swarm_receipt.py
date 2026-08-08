#!/usr/bin/env python3
"""Independently read back and authenticate a Kanban swarm TLC receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

from verify_kanban_swarm_tla import (
    _normalize_tlc_output,
    _validate_semantics,
    _write_semantics_module,
)


EXPECTED_INVARIANTS = {
    "TypeOK",
    "AtomicGraphVisibility",
    "NoPartialGraphBeforeCommit",
    "SingleIdempotentGraph",
    "SynthesisRequiresAuthority",
    "UnknownGateFailsClosed",
    "ForgedCommentIsNotAuthority",
    "WorkerBoardPinning",
    "ProductionRefinementMap",
}
EXPECTED_GATE_CASES = {
    "done_pass",
    "done_fail",
    "done_missing",
    "done_malformed",
    "archived_pass",
    "running_pass",
    "unknown_gate_done_pass",
}
EXPECTED_BOARD_CASES = {
    "worker_missing_pin_default_route",
    "worker_missing_pin_explicit_route",
    "worker_db_pin_default_route",
    "worker_db_pin_explicit_route",
    "worker_board_pin_default_route",
    "worker_matching_board_route",
    "worker_mismatched_board_route",
}
EXPECTED_SOURCE_PATHS = {
    "scripts/export_kanban_swarm_semantics.py",
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_decompose.py",
    "hermes_cli/kanban_swarm.py",
    "tools/kanban_tools.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tlc_metrics(output: str) -> dict[str, int | None]:
    states = re.search(
        r"(?P<generated>[0-9,]+) states generated, "
        r"(?P<distinct>[0-9,]+) distinct states found",
        output,
    )
    depth = re.search(
        r"depth of the complete state graph search is (?P<depth>[0-9,]+)",
        output,
        re.IGNORECASE,
    )
    queue = re.search(r"(?P<queued>[0-9,]+) states left on queue", output)

    def number(match: re.Match[str] | None, name: str) -> int | None:
        return int(match.group(name).replace(",", "")) if match else None

    return {
        "states_generated": number(states, "generated"),
        "distinct_states": number(states, "distinct"),
        "depth": number(depth, "depth"),
        "states_left_on_queue": number(queue, "queued"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-age-seconds", type=int, default=3600)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    failures: list[str] = []
    repo = Path(__file__).resolve().parents[1]

    def bound_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(receipt.get("schema") == "hermes.kanban-swarm.tlc-receipt.v3", "schema")
    require(receipt.get("fresh_execution") is True, "fresh_execution")
    require(receipt.get("success") is True, "success")
    require(receipt.get("exit_code") == 0, "exit_code")
    require(
        receipt.get("result", {}).get("states_generated", 0) > 0, "states_generated"
    )
    require(receipt.get("result", {}).get("distinct_states", 0) > 0, "distinct_states")
    require(receipt.get("result", {}).get("states_left_on_queue") == 0, "queue")
    require(
        receipt.get("production_semantics", {}).get("bound") is True, "semantic binding"
    )
    semantic = receipt.get("production_semantics", {})
    require(semantic.get("validation_failures") == [], "semantic validation")
    gate_cases = semantic.get("case_names", [])
    require(
        isinstance(gate_cases, list)
        and len(gate_cases) == len(EXPECTED_GATE_CASES)
        and len(gate_cases) == len(set(gate_cases))
        and set(gate_cases) == EXPECTED_GATE_CASES,
        "gate cases",
    )
    board_cases = semantic.get("board_case_names", [])
    require(
        isinstance(board_cases, list)
        and len(board_cases) == len(EXPECTED_BOARD_CASES)
        and len(board_cases) == len(set(board_cases))
        and set(board_cases) == EXPECTED_BOARD_CASES,
        "board cases",
    )
    invariants = receipt.get("config", {}).get("invariants", [])
    require(
        isinstance(invariants, list)
        and len(invariants) == len(EXPECTED_INVARIANTS)
        and len(invariants) == len(set(invariants))
        and set(invariants) == EXPECTED_INVARIANTS,
        "invariant identities",
    )

    started = datetime.fromisoformat(receipt["started_at_utc"])
    finished = datetime.fromisoformat(receipt["finished_at_utc"])
    now = datetime.now(timezone.utc)
    require(started <= finished <= now, "timestamp ordering")
    age = (datetime.now(timezone.utc) - finished).total_seconds()
    require(0 <= age <= args.max_age_seconds, f"receipt age {age:.1f}s")
    try:
        run_id = str(uuid.UUID(receipt["run_id"]))
        require(run_id == receipt["run_id"], "canonical run_id")
    except (KeyError, TypeError, ValueError):
        require(False, "valid run_id")
        run_id = None
    if run_id is not None:
        duplicates = 0
        for candidate in args.receipt.parent.glob("*.json"):
            if candidate.resolve() == args.receipt.resolve():
                continue
            try:
                duplicates += (
                    json.loads(candidate.read_text(encoding="utf-8")).get("run_id")
                    == run_id
                )
            except (OSError, json.JSONDecodeError):
                continue
        require(duplicates == 0, "unique run_id")

    checks = [
        (bound_path(receipt["model"]["path"]), receipt["model"]["sha256"], "model"),
        (
            bound_path(receipt["config"]["path"]),
            receipt["config"]["sha256"],
            "config",
        ),
        (
            Path(receipt["tool"]["jar_path"]),
            receipt["tool"]["jar_sha256"],
            "TLC jar",
        ),
        (
            Path(receipt["production_semantics"]["path"]),
            receipt["production_semantics"]["sha256"],
            "production semantics",
        ),
        (
            Path(receipt["production_semantics"]["tla_module_path"]),
            receipt["production_semantics"]["tla_module_sha256"],
            "generated production semantics module",
        ),
        (
            Path(receipt["result"]["log_path"]),
            receipt["result"]["stdout_sha256"],
            "TLC log",
        ),
    ]
    for path, expected, label in checks:
        require(path.is_file(), f"{label} missing")
        if path.is_file():
            require(_sha256(path) == expected, f"{label} hash")

    result_data = receipt.get("result", {})
    bound_log_path = Path(result_data.get("log_path", ""))
    if bound_log_path.is_file():
        bound_output = bound_log_path.read_text(encoding="utf-8", errors="replace")
        bound_metrics = _tlc_metrics(bound_output)
        require(
            "Model checking completed. No error has been found." in bound_output,
            "bound TLC success marker",
        )
        require(
            hashlib.sha256(_normalize_tlc_output(bound_output).encode()).hexdigest()
            == result_data.get("normalized_stdout_sha256"),
            "bound TLC normalized output correspondence",
        )
        for field, actual in bound_metrics.items():
            require(
                actual == result_data.get(field),
                f"bound TLC {field} correspondence",
            )

    config_path = bound_path(receipt["config"]["path"])
    if config_path.is_file():
        actual_invariants = [
            line.split(maxsplit=1)[1]
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("INVARIANT ")
        ]
        require(
            actual_invariants == invariants, "config-to-receipt invariant identities"
        )
        require(
            len(actual_invariants) == len(EXPECTED_INVARIANTS)
            and len(set(actual_invariants)) == len(actual_invariants)
            and set(actual_invariants) == EXPECTED_INVARIANTS,
            "config invariant content",
        )

    sources = receipt.get("production_semantics", {}).get("sources", {})
    require(
        isinstance(sources, dict) and set(sources) == EXPECTED_SOURCE_PATHS,
        "production source identities",
    )
    semantics_path = Path(semantic.get("path", ""))
    semantics_data = None
    semantics_source_root = repo
    if semantics_path.is_file():
        semantics_data = json.loads(semantics_path.read_text(encoding="utf-8"))
        source_root_value = semantics_data.get("source_root")
        if isinstance(source_root_value, str):
            semantics_source_root = Path(source_root_value).resolve()
        require(semantics_source_root == repo.resolve(), "semantics source root")

    for source, binding in sources.items():
        path = semantics_source_root / source
        require(path.is_file(), f"source missing: {source}")
        if path.is_file():
            actual = _sha256(path)
            require(
                actual == binding["expected"] == binding["actual"],
                f"source hash: {source}",
            )

    module_path = Path(semantic.get("tla_module_path", ""))
    if semantics_data is not None and module_path.is_file():
        actual_gate_names = [
            case.get("name") for case in semantics_data.get("cases", [])
        ]
        actual_board_names = [
            case.get("name") for case in semantics_data.get("board_cases", [])
        ]
        actual_sources = semantics_data.get("sources")
        require(
            sorted(actual_gate_names) == gate_cases, "semantics-to-receipt gate cases"
        )
        require(
            sorted(actual_board_names) == board_cases,
            "semantics-to-receipt board cases",
        )
        require(
            isinstance(actual_sources, dict)
            and set(actual_sources) == EXPECTED_SOURCE_PATHS,
            "semantics source identities",
        )
        if isinstance(actual_sources, dict):
            for source, expected_hash in actual_sources.items():
                binding = sources.get(source, {}) if isinstance(sources, dict) else {}
                require(
                    expected_hash == binding.get("expected") == binding.get("actual"),
                    f"semantics-to-receipt source hash: {source}",
                )
        require(
            _validate_semantics(semantics_data) == [], "semantic content validation"
        )
        with tempfile.TemporaryDirectory(prefix="kanban-receipt-verify-") as temp_dir:
            regenerated = Path(temp_dir) / "KanbanSwarmProductionSemantics.tla"
            _write_semantics_module(regenerated, semantics_data)
            require(
                regenerated.read_bytes() == module_path.read_bytes(),
                "semantics-to-generated-module correspondence",
            )

    model_path = bound_path(receipt["model"]["path"])
    jar_path = Path(receipt["tool"]["jar_path"])
    argv = receipt.get("argv")
    expected_argv = [
        argv[0] if isinstance(argv, list) and argv else "",
        "-cp",
        str(jar_path),
        "tlc2.TLC",
        "-cleanup",
        "-workers",
        "1",
        "-config",
        config_path.name,
        model_path.name,
    ]
    require(argv == expected_argv, "TLC argv")
    java_path = Path(expected_argv[0])
    require(java_path.is_file(), "Java executable missing")

    # A historical log is not self-authenticating merely because the same
    # receipt hashes it. Re-run the exact model/config/generated module and
    # compare normalized output plus all state metrics before accepting it.
    if not failures:
        with tempfile.TemporaryDirectory(prefix="kanban-receipt-tlc-") as temp_dir:
            run_dir = Path(temp_dir)
            for source_path in (model_path, config_path, module_path):
                shutil.copy2(source_path, run_dir / source_path.name)
            completed = subprocess.run(
                expected_argv,
                cwd=run_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        output = completed.stdout
        state_match = re.search(
            r"(?P<generated>[0-9,]+) states generated, "
            r"(?P<distinct>[0-9,]+) distinct states found",
            output,
        )
        depth_match = re.search(
            r"depth of the complete state graph search is (?P<depth>[0-9,]+)",
            output,
            re.IGNORECASE,
        )
        queue_match = re.search(r"(?P<queued>[0-9,]+) states left on queue", output)
        require(completed.returncode == receipt.get("exit_code") == 0, "TLC rerun exit")
        require(
            "Model checking completed. No error has been found." in output,
            "TLC rerun success marker",
        )
        require(
            state_match is not None
            and int(state_match.group("generated").replace(",", ""))
            == result_data.get("states_generated")
            and int(state_match.group("distinct").replace(",", ""))
            == result_data.get("distinct_states"),
            "TLC rerun states",
        )
        require(
            depth_match is not None
            and int(depth_match.group("depth").replace(",", ""))
            == result_data.get("depth"),
            "TLC rerun depth",
        )
        require(
            queue_match is not None
            and int(queue_match.group("queued").replace(",", ""))
            == result_data.get("states_left_on_queue")
            == 0,
            "TLC rerun queue",
        )
        require(
            hashlib.sha256(_normalize_tlc_output(output).encode()).hexdigest()
            == result_data.get("normalized_stdout_sha256"),
            "TLC normalized output correspondence",
        )

    result = {
        "receipt": str(args.receipt),
        "run_id": receipt.get("run_id"),
        "failures": failures,
        "success": not failures,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
