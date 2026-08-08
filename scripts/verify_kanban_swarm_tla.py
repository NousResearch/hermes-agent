#!/usr/bin/env python3
"""Execute TLC for the Kanban swarm model and emit a bound JSON receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "unknown")


def _normalize_tlc_output(text: str) -> str:
    """Remove TLC run-local noise while preserving parsed/model results."""
    text = re.sub(r"with fp \d+ and seed -?\d+", "with fp FP and seed SEED", text)
    text = re.sub(r"\[pid: \d+\]", "[pid: PID]", text)
    text = re.sub(r"/tmp/(?:kanban(?:-receipt)?-tlc|tlc)-[^/\s)]+", "/tmp/TLC", text)
    text = re.sub(r"\(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\)", "(TIMESTAMP)", text)
    text = re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", "TIMESTAMP", text)
    text = re.sub(r"Finished in \S+ at", "Finished in DURATION at", text)
    return text


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


def _gate_evidence(metadata: object) -> str:
    if not isinstance(metadata, dict) or "gate" not in metadata:
        return "missing"
    value = metadata["gate"]
    return (
        value if isinstance(value, str) and value in {"pass", "fail"} else "malformed"
    )


def _validate_semantics(semantics: dict) -> list[str]:
    failures: list[str] = []
    cases = semantics.get("cases")
    board_cases = semantics.get("board_cases")
    if not isinstance(cases, list) or (
        len(cases) != len(EXPECTED_GATE_CASES)
        or len({case.get("name") for case in cases}) != len(cases)
        or {case.get("name") for case in cases} != EXPECTED_GATE_CASES
    ):
        failures.append("exact gate case names")
    if not isinstance(board_cases, list) or (
        len(board_cases) != len(EXPECTED_BOARD_CASES)
        or len({case.get("name") for case in board_cases}) != len(board_cases)
        or {case.get("name") for case in board_cases} != EXPECTED_BOARD_CASES
    ):
        failures.append("exact board case names")
    if failures:
        return failures
    assert isinstance(cases, list)
    assert isinstance(board_cases, list)
    for case in cases:
        evidence = _gate_evidence(case.get("verifier_metadata"))
        abstract = (
            case.get("gate_kind") == "metadata_gate_pass"
            and case.get("verifier_status") == "done"
            and evidence == "pass"
        )
        observations = (
            case.get("promoted"),
            case.get("claimable_when_forced_ready"),
            case.get("review_claimable"),
            case.get("recovery_ready"),
        )
        if case.get("expected") is not abstract or any(
            value is not abstract for value in observations
        ):
            failures.append(f"gate case values: {case.get('name')}")
    for case in board_cases:
        pin_kind = case.get("pin_kind")
        requested = case.get("requested_board")
        pinned = case.get("pinned_board")
        abstract = (pin_kind == "db" and requested == "none") or (
            pin_kind == "board" and (requested == "none" or requested == pinned)
        )
        if case.get("expected") is not abstract or case.get("allowed") is not abstract:
            failures.append(f"board case values: {case.get('name')}")
    return failures


def _case_function(name: str, cases: list[dict], key: str) -> str:
    arms = []
    for case in cases:
        value = case[key]
        rendered = (
            "TRUE"
            if value is True
            else "FALSE"
            if value is False
            else json.dumps(value)
        )
        arms.append(f"c = {json.dumps(case['name'])} -> {rendered}")
    return f"{name}(c) == CASE " + " [] ".join(arms)


def _write_semantics_module(path: Path, semantics: dict) -> None:
    cases = semantics["cases"]
    board_cases = semantics["board_cases"]
    gate_names = ", ".join(json.dumps(case["name"]) for case in cases)
    board_names = ", ".join(json.dumps(case["name"]) for case in board_cases)
    normalized_cases = [
        case | {"evidence": _gate_evidence(case["verifier_metadata"])} for case in cases
    ]
    text = "\n".join([
        "---- MODULE KanbanSwarmProductionSemantics ----",
        f"ProductionGateCases == {{{gate_names}}}",
        _case_function("ProductionGateStatus", normalized_cases, "verifier_status"),
        _case_function("ProductionGateEvidence", normalized_cases, "evidence"),
        _case_function("ProductionGateKind", normalized_cases, "gate_kind"),
        _case_function("ProductionGateExpected", normalized_cases, "expected"),
        f"ProductionBoardCases == {{{board_names}}}",
        _case_function("ProductionPinKind", board_cases, "pin_kind"),
        _case_function("ProductionPinnedBoard", board_cases, "pinned_board"),
        _case_function("ProductionRequestedBoard", board_cases, "requested_board"),
        _case_function("ProductionBoardExpected", board_cases, "expected"),
        "====",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", required=True, type=Path)
    parser.add_argument("--jar", required=True, type=Path)
    parser.add_argument("--semantics", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model_dir = repo / "formal" / "kanban_swarm"
    model = model_dir / "KanbanSwarmSecurity.tla"
    config = model_dir / "KanbanSwarmSecurity.cfg"
    for required in (args.java, args.jar, args.semantics, model, config):
        if not required.is_file():
            parser.error(f"required file does not exist: {required}")

    semantics = json.loads(args.semantics.read_text(encoding="utf-8"))
    semantic_failures = _validate_semantics(semantics)
    semantic_sources = semantics.get("sources", {})
    source_bindings = {
        path: {"expected": expected, "actual": _sha256(repo / path)}
        for path, expected in semantic_sources.items()
    }
    semantics_bound = (
        not semantic_failures
        and bool(semantics.get("success"))
        and all(
            binding["expected"] == binding["actual"]
            for binding in source_bindings.values()
        )
    )
    started_at = datetime.now(timezone.utc)
    run_id = str(uuid.uuid4())

    semantics_module = args.receipt.with_name(
        "KanbanSwarmProductionSemantics.tla"
    ).resolve()
    semantics_module.parent.mkdir(parents=True, exist_ok=True)
    _write_semantics_module(semantics_module, semantics)
    command = [
        str(args.java),
        "-cp",
        str(args.jar),
        "tlc2.TLC",
        "-cleanup",
        "-workers",
        "1",
        "-config",
        config.name,
        model.name,
    ]
    with tempfile.TemporaryDirectory(prefix="kanban-tlc-") as temp:
        run_dir = Path(temp)
        shutil.copy2(model, run_dir / model.name)
        shutil.copy2(config, run_dir / config.name)
        shutil.copy2(semantics_module, run_dir / semantics_module.name)
        completed = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output = completed.stdout
    finished_at = datetime.now(timezone.utc)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.receipt.with_suffix(".log")
    log_path.write_text(output, encoding="utf-8")

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
    success_marker = "Model checking completed. No error has been found."
    successful = (
        completed.returncode == 0
        and success_marker in output
        and semantics_bound
        and state_match is not None
        and int(state_match.group("distinct").replace(",", "")) > 0
        and queue_match is not None
        and int(queue_match.group("queued").replace(",", "")) == 0
    )
    invariants = [
        line.split(maxsplit=1)[1]
        for line in config.read_text(encoding="utf-8").splitlines()
        if line.startswith("INVARIANT ")
    ]
    java_version = subprocess.run(
        [str(args.java), "-version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout

    receipt = {
        "schema": "hermes.kanban-swarm.tlc-receipt.v3",
        "run_id": run_id,
        "fresh_execution": True,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "success": successful,
        "exit_code": completed.returncode,
        "argv": command,
        "java_version": _first_line(java_version),
        "tlc_version": _first_line(output),
        "model": {
            "path": str(model.relative_to(repo)),
            "sha256": _sha256(model),
        },
        "config": {
            "path": str(config.relative_to(repo)),
            "sha256": _sha256(config),
            "invariants": invariants,
        },
        "tool": {
            "jar_path": str(args.jar),
            "jar_sha256": _sha256(args.jar),
        },
        "production_semantics": {
            "path": str(args.semantics),
            "sha256": _sha256(args.semantics),
            "bound": semantics_bound,
            "validation_failures": semantic_failures,
            "cases": len(semantics.get("cases", [])),
            "board_cases": len(semantics.get("board_cases", [])),
            "case_names": sorted(case["name"] for case in semantics.get("cases", [])),
            "board_case_names": sorted(
                case["name"] for case in semantics.get("board_cases", [])
            ),
            "tla_module_path": str(semantics_module),
            "tla_module_sha256": _sha256(semantics_module),
            "sources": source_bindings,
        },
        "result": {
            "states_generated": (
                int(state_match.group("generated").replace(",", ""))
                if state_match
                else None
            ),
            "distinct_states": (
                int(state_match.group("distinct").replace(",", ""))
                if state_match
                else None
            ),
            "depth": (
                int(depth_match.group("depth").replace(",", ""))
                if depth_match
                else None
            ),
            "states_left_on_queue": (
                int(queue_match.group("queued").replace(",", ""))
                if queue_match
                else None
            ),
            "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "normalized_stdout_sha256": hashlib.sha256(
                _normalize_tlc_output(output).encode()
            ).hexdigest(),
            "log_path": str(log_path),
        },
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if successful and depth_match else 1


if __name__ == "__main__":
    sys.exit(main())
