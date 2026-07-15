from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from hermes_cli import provider_evaluation as pe


FAKE_HERMES = r"""#!/usr/bin/env python3
import hashlib
import os
import re
import sys
from pathlib import Path

from hermes_state import SessionDB

argv = sys.argv[1:]
source = argv[argv.index("--source") + 1]
prompt = argv[argv.index("-q") + 1]
resume = argv[argv.index("--resume") + 1] if "--resume" in argv else None
parts = source.split(":")
case_id = parts[-2]
home = Path(os.environ["HERMES_HOME"])
db = SessionDB(db_path=home / "state.db")
if resume:
    session_id = db.resolve_resume_session_id(resume)
else:
    session_id = "fake-" + hashlib.sha256(source.encode()).hexdigest()[:20]
    db.create_session(session_id, "evaluation", model="fake", cwd=str(Path.cwd()))

marker_match = re.search(r"(?:exactly with|marker) ([A-Z][A-Z0-9_]+)", prompt)
marker = marker_match.group(1) if marker_match else "FAKE_OK"
tool_by_case = {
    "tier0.read_file": "read_file",
    "tier0.search_files": "search_files",
    "tier0.failed_read_recovery": "read_file",
    "tools.safe_file_mutation": "write_file",
    "tools.terminal_observation": "terminal",
    "tools.search_decoys": "search_files",
    "tools.local_memory_search": "session_search",
    "continuity.failed_tool_correction": "read_file",
    "continuity.artifact_verification": "read_file",
}
tool_name = tool_by_case.get(case_id)
if "compression" in case_id and not resume:
    db.append_message(session_id, "user", prompt)
    db.append_message(session_id, "assistant", marker)
    db.end_session(session_id, "compression")
    child = session_id + "-child"
    db.create_session(child, "evaluation", model="fake", parent_session_id=session_id, cwd=str(Path.cwd()))
    session_id = child
if tool_name:
    call_id = "call-1"
    arguments = {"path": str(Path.cwd() / "read_marker.txt")}
    db.append_message(session_id, "user", prompt)
    db.append_message(session_id, "assistant", tool_calls=[{"id": call_id, "function": {"name": tool_name, "arguments": arguments}}])
    db.append_message(session_id, "tool", "observed", tool_name=tool_name, tool_call_id=call_id)
    db.append_message(session_id, "assistant", marker)
else:
    db.append_message(session_id, "user", prompt)
    db.append_message(session_id, "assistant", marker)
print(f"session_id: {session_id}")
print(marker)
"""


def _fake_executable(tmp_path: Path) -> Path:
    path = tmp_path / "fake-hermes"
    path.write_text(FAKE_HERMES, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _manifest(tmp_path: Path) -> dict:
    value = {
        "schema_version": "candidate-stack-manifest.v1",
        "weights": {"model_id": "fake", "revision": "r", "quantization": "fp16"},
        "runtime": {
            "provider_id": "fake",
            "model": "fake",
            "endpoint_class": "local",
            "runtime_name": "fake",
            "server_version": "1",
            "protocol": "chat",
        },
        "template_and_parser": {
            "chat_template_sha256": "a",
            "tool_call_template_sha256": "b",
            "parser_name": "fake",
            "parser_version": "1",
            "parser_mode": "json",
        },
        "decoding": {
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 10,
            "seed_policy": "fixed",
        },
        "context": {
            "model_context_length": 1024,
            "hermes_context_setting": "default",
            "compression_enabled": True,
            "system_prompt_sha256": "a",
            "tool_schema_sha256": "b",
        },
        "hermes": {
            "revision": "r",
            "dirty_tree": False,
            "package_lock_sha256": "a",
            "profile": "test",
            "config_sha256": "a",
            "source_tag": "test",
            "rules": [],
            "skills": [],
            "memory": {},
            "toolsets": ["hermes-cli"],
            "disabled_toolsets": [],
            "mcp_catalog_digest": "a",
        },
        "hardware": {
            "host_class": "test",
            "os": "linux",
            "python": "3",
            "accelerator_family": "cpu",
            "device_count": 1,
            "driver_major": "0",
        },
        "lane": {
            "lane_id": "cli-full-v1",
            "suite_id": "full-hermes-cli-v1",
            "suite_version": 1,
            "external_network": False,
            "filesystem_scope": "fixture-only",
            "approval_policy": "configured",
        },
        "rollback": {
            "current_route_id": "route",
            "recipe": "restore",
            "owner": "test",
            "artifact_sha256": "a",
        },
    }
    return pe.load_manifest(
        _write_json(tmp_path / "manifest.json", value), capture_tools=False
    )


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_full_catalog_executes_through_fake_subprocess_and_real_sessiondb(
    tmp_path: Path,
):
    home = tmp_path / "home"
    (home / "skills" / "fixture-skill").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {default: fake}\n", encoding="utf-8")
    (home / "MEMORY.md").write_text("MEMORY_MARKER=loaded\n", encoding="utf-8")
    (home / "skills" / "fixture-skill" / "SKILL.md").write_text(
        "SKILL_MARKER=loaded\n", encoding="utf-8"
    )
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "AGENTS.md").write_text("RULE_MARKER=loaded\n", encoding="utf-8")
    (fixture / "read_marker.txt").write_text("READINESS_OK\n", encoding="utf-8")
    fake = _fake_executable(tmp_path)
    # Use one loaded manifest for each arm; the evaluator only compares the
    # captured identity and the fake boundary makes both arms deterministic.
    manifest_value = {
        "schema_version": "candidate-stack-manifest.v1",
        "weights": {"model_id": "fake", "revision": "r", "quantization": "fp16"},
        "runtime": {
            "provider_id": "fake",
            "model": "fake",
            "endpoint_class": "local",
            "runtime_name": "fake",
            "server_version": "1",
            "protocol": "chat",
        },
        "template_and_parser": {
            "chat_template_sha256": "a",
            "tool_call_template_sha256": "b",
            "parser_name": "fake",
            "parser_version": "1",
            "parser_mode": "json",
        },
        "decoding": {
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 10,
            "seed_policy": "fixed",
        },
        "context": {
            "model_context_length": 1024,
            "hermes_context_setting": "default",
            "compression_enabled": True,
            "system_prompt_sha256": "a",
            "tool_schema_sha256": "b",
        },
        "hermes": {
            "revision": "r",
            "dirty_tree": False,
            "package_lock_sha256": "a",
            "profile": "test",
            "config_sha256": "a",
            "source_tag": "test",
            "rules": [],
            "skills": [],
            "memory": {},
            "toolsets": ["hermes-cli"],
            "disabled_toolsets": [],
            "mcp_catalog_digest": "a",
        },
        "hardware": {
            "host_class": "test",
            "os": "linux",
            "python": "3",
            "accelerator_family": "cpu",
            "device_count": 1,
            "driver_major": "0",
        },
        "lane": {
            "lane_id": "cli-full-v1",
            "suite_id": "full-hermes-cli-v1",
            "suite_version": 1,
            "external_network": False,
            "filesystem_scope": "fixture-only",
            "approval_policy": "configured",
        },
        "rollback": {
            "current_route_id": "route",
            "recipe": "restore",
            "owner": "test",
            "artifact_sha256": "a",
        },
    }
    manifest_path = _write_json(tmp_path / "manifest.json", manifest_value)
    manifest = pe.load_manifest(manifest_path, capture_tools=False)
    root = tmp_path / "run"
    root.mkdir()
    pe._write_json_atomic(root / "manifest.candidate.json", manifest)
    pe._write_json_atomic(root / "manifest.incumbent.json", manifest)
    receipts = []
    schedule = []
    for index, case in enumerate(pe.get_full_suite_cases(), start=1):
        pair = {
            "pair_id": f"pair-{index:03d}",
            "case_id": case.case_id,
            "repetition": 1,
            "seed": 20260715,
            "arm_order": ["candidate", "incumbent"],
            "run_id": root.name,
        }
        schedule.append(pair)
        for arm in pair["arm_order"]:
            receipt = pe._run_attempt(
                case=case,
                pair=pair,
                arm=arm,
                manifest=manifest,
                attempt_root=root / "raw" / arm / pair["pair_id"] / "1",
                run_root=root,
                base_home=home,
                fixture_root=fixture,
                hermes_executable=str(fake),
                timeout=10,
            )
            receipts.append(receipt)
    pe._write_receipts(root / "receipts.jsonl", receipts)
    cases = {case.case_id: case for case in pe.get_full_suite_cases()}
    observations, failures = pe._observations_from_receipts(
        root, receipts, schedule, cases
    )
    assert not failures
    assert len(observations) == 27
    summary = pe.scoring.score_evaluation(
        observations, repetitions=1, expected_case_ids=cases, replicates=16
    )
    assert summary["status"] == "SCREEN-PASS"
    assert summary["candidate"]["hfs"] == 100.0
    assert all((root / receipt["raw"]["session"]).is_file() for receipt in receipts)
    # Compression receipts prove the SessionDB parent/child end_reason was
    # observed through the real database, not supplied by the fake response.
    compression = [
        receipt for receipt in receipts if receipt["case_id"].startswith("compression.")
    ]
    assert compression and all(
        receipt["session"]["compression_events"] > 0 for receipt in compression
    )
    pe._write_json_atomic(root / "run.json", {"seed": 20260715, "repetitions": 1})
    pe._write_json_atomic(root / "schedule.json", schedule)
    pe._write_checksums(root)
    code, offline = pe.score_run(root)
    assert code == 0
    assert offline["status"] == "SCREEN-PASS"
    first_stdout = root / receipts[0]["raw"]["stdout"]
    first_stdout.write_text(
        first_stdout.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    code, tampered = pe.score_run(root)
    assert code == 1
    assert tampered["status"] == "GATE-FAILED"
    assert any(item.startswith("tampered:") for item in tampered["hard_gate_failures"])
