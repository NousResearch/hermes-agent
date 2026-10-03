from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import replace
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import continuity_bridge
import continuity_gate
import install_continuity_adapters as continuity_adapters
from continuity_bridge import (
    build_preflight,
    tool_events_from_payload,
    validate_tool_adjacency,
)
from continuity_common import (
    ContinuityError,
    atomic_write_json,
    git_state,
    read_markdown_frontmatter,
    render_markdown_frontmatter,
    sha256_file,
)
from continuity_event import dispatch
from continuity_gate import (
    create_receipt,
    managed_manifest_file_errors,
    manifest_membership_errors,
    promote,
    static_validate,
    verify_receipt,
)
from install_continuity_adapters import (
    install_hermes_adapter,
    render_project_adapters,
    rollback_hermes_adapter,
)


GOAL = "Pilot deterministic cross-agent continuity in Hermes"
TASK_ID = "hermes-continuity-b6l"
CHANGE_ID = "001-global-continuity-pilot"


def _lock_worker(config_path: str, marker_path: str, label: str) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    with continuity_gate._promotion_lock(config, timeout=5):
        with Path(marker_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{label}-start\n")
        time.sleep(0.2)
        with Path(marker_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{label}-end\n")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _frontmatter(data: dict, body: str = "fixture") -> str:
    import yaml

    return f"---\n{yaml.safe_dump(data, sort_keys=False).strip()}\n---\n\n{body}\n"


def _lock_fake_bd(lock_path: Path, fake_bd: Path) -> None:
    import platform

    machine = (
        platform
        .machine()
        .lower()
        .replace("aarch64", "arm64")
        .replace("x86_64", "amd64")
    )
    atomic_write_json(
        lock_path,
        {
            "schema_version": 1,
            "tools": {
                "beads": {
                    "version": "fixture",
                    f"{platform.system().lower()}-{machine}-sha256": sha256_file(
                        fake_bd
                    ),
                }
            },
        },
    )


@pytest.fixture()
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    _git(repo, "init", "-b", "pilot")
    _git(repo, "config", "user.email", "pilot@example.invalid")
    _git(repo, "config", "user.name", "Pilot")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-m", "seed")

    for relative in ("AGENTS.md", ".specify/memory/constitution.md"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    spec_path = repo / "specs/001-global-continuity-pilot/spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        _frontmatter({
            "project": "Hermes",
            "repo_id": "hermes-agent",
            "goal": GOAL,
            "state": "ACTIVE",
            "active_task": TASK_ID,
            "change_id": CHANGE_ID,
            "folder": str(repo.resolve()),
        }),
        encoding="utf-8",
    )
    card_path = state / "current.md"
    card_path.write_text(
        _frontmatter({
            "project": "Hermes",
            "folder": str(repo.resolve()),
            "goal": GOAL,
            "user_context": "bounded fixture",
            "state": "ACTIVE",
            "active_task": TASK_ID,
            "spec_change": CHANGE_ID,
            "evidence": {"commit": _git(repo, "rev-parse", "HEAD"), "receipt": None},
            "next_action": "run the pilot",
            "blockers": [],
            "supersedes": "none",
        }),
        encoding="utf-8",
    )
    task = {
        "id": TASK_ID,
        "title": GOAL,
        "status": "in_progress",
        "spec_id": "specs/001-global-continuity-pilot/spec.md",
    }
    issues = state / "issues.jsonl"
    issues.write_text(json.dumps(task) + "\n", encoding="utf-8")
    fake_bd = state / "bd"
    fake_bd.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['BEADS_DIR']).parent / 'issues.jsonl'\n"
        "items = [json.loads(line) for line in path.read_text().splitlines() if line]\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'close':\n"
        "    items[0]['status'] = 'closed'\n"
        "    path.write_text('\\n'.join(json.dumps(item) for item in items) + '\\n')\n"
        "print(json.dumps(items))\n",
        encoding="utf-8",
    )
    fake_bd.chmod(0o755)
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "evidence_pass.py").write_text(
        "print('=== Summary: 1 files, 1 tests passed, 0 failed (100% complete) ===')\n",
        encoding="utf-8",
    )
    (scripts_dir / "evidence_fail.py").write_text(
        "import sys\nprint('=== Summary: 1 files, 0 tests passed, 1 failed ===')\nsys.exit(1)\n",
        encoding="utf-8",
    )
    (scripts_dir / "full_suite.py").write_text(
        "print('=== Summary: 1 files, 1 tests passed, 0 failed (100% complete) ===')\n",
        encoding="utf-8",
    )
    (scripts_dir / "runtime_check.py").write_text(
        "print('runtime ok')\n", encoding="utf-8"
    )
    (scripts_dir / "rollback_check.py").write_text(
        "print('rollback ok')\n", encoding="utf-8"
    )
    (scripts_dir / "zero_tests.py").write_text(
        "print('command ran without a test summary')\n", encoding="utf-8"
    )
    (scripts_dir / "env_check.py").write_text(
        "import os, sys\n"
        "print('=== Summary: 1 files, 1 tests passed, 0 failed ===')\n"
        "sys.exit(1 if 'TOP_SECRET' in os.environ else 0)\n",
        encoding="utf-8",
    )
    lock_path = repo / ".continuity/toolchain.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lock_fake_bd(lock_path, fake_bd)

    event_log = state / "events.jsonl"
    config = {
        "schema_version": 1,
        "project": "Hermes",
        "repo_id": "hermes-agent",
        "risk_tier": "T3",
        "integration_ref": "HEAD",
        "toolchain_lock": ".continuity/toolchain.lock.json",
        "goal": GOAL,
        "expected_repo_root": str(repo.resolve()),
        "external_volume": "/",
        "state_root": str(state),
        "max_output_chars": 8000,
        "basic_memory": {"project": "pilot", "card_path": str(card_path)},
        "beads": {
            "binary": str(fake_bd),
            "data_dir": str(state / "beads"),
            "issues_path": str(issues),
            "active_task": TASK_ID,
            "timeout_seconds": 7,
            "completion_timeout_seconds": 7,
        },
        "spec": {
            "change_id": CHANGE_ID,
            "path": "specs/001-global-continuity-pilot/spec.md",
        },
        "instructions": ["AGENTS.md", ".specify/memory/constitution.md"],
        "evidence_policy": {
            "focused_suite": {
                "name": "focused-suite",
                "argv": [sys.executable, "scripts/evidence_pass.py"],
                "timeout_seconds": 30,
            },
            "full_suite": {
                "name": "full-suite",
                "argv": [sys.executable, "scripts/full_suite.py"],
                "timeout_seconds": 30,
            },
            "runtime_checks": [
                {
                    "name": "sandbox-hosts",
                    "argv": [sys.executable, "scripts/runtime_check.py"],
                    "surface": "sandbox",
                    "timeout_seconds": 30,
                }
            ],
            "rollback_check": {
                "name": "rollback",
                "argv": [sys.executable, "scripts/rollback_check.py"],
                "mode": "dry-run",
                "timeout_seconds": 30,
            },
        },
        "evidence_executables": [
            {"path": sys.executable, "sha256": sha256_file(Path(sys.executable))}
        ],
        "external_instructions": [],
        "event_log": str(event_log),
        "receipt_dir": str(state / "receipts"),
    }
    config_path = repo / ".continuity/config.json"
    atomic_write_json(config_path, config)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pilot fixture")
    card_data, card_body = read_markdown_frontmatter(card_path)
    card_data["evidence"]["commit"] = _git(repo, "rev-parse", "HEAD")
    card_path.write_text(
        render_markdown_frontmatter(card_data, card_body), encoding="utf-8"
    )
    monkeypatch.setattr(continuity_gate, "CANONICAL_CONFIG_PATH", config_path)
    return {
        "repo": repo,
        "state": state,
        "config": config_path,
        "card": card_path,
        "spec": spec_path,
        "issues": issues,
        "bd": fake_bd,
        "lock": lock_path,
        "events": event_log,
    }


def _preflight(pilot: dict[str, Path]) -> dict:
    return build_preflight(
        pilot["config"], cwd=pilot["repo"], require_mounted_volume=False
    )


def _commit_and_rebind(pilot: dict[str, Path], message: str) -> None:
    _git(pilot["repo"], "add", ".")
    _git(pilot["repo"], "commit", "-m", message)
    card_data, card_body = read_markdown_frontmatter(pilot["card"])
    card_data["evidence"]["commit"] = _git(pilot["repo"], "rev-parse", "HEAD")
    pilot["card"].write_text(
        render_markdown_frontmatter(card_data, card_body), encoding="utf-8"
    )


def _policy(pilot: dict[str, Path]) -> dict:
    return json.loads(pilot["config"].read_text(encoding="utf-8"))["evidence_policy"]


def _passing_receipt(pilot: dict[str, Path], target: str = "TESTED") -> dict:
    policy = _policy(pilot)
    return create_receipt(
        pilot["config"],
        cwd=pilot["repo"],
        risk_tier="T3",
        lifecycle_target=target,
        commands=[policy["full_suite" if target == "ENFORCED" else "focused_suite"]],
        runtime_checks=policy["runtime_checks"],
        rollback=policy["rollback_check"],
        require_mounted_volume=False,
    )


def test_healthy_preflight_recovers_exact_scope(pilot: dict[str, Path]) -> None:
    first = _preflight(pilot)
    second = _preflight(pilot)
    assert first == second
    assert first["status"] == "AVAILABLE"
    assert first["completion_allowed"] is True
    assert first["active_task"] == TASK_ID
    assert first["spec_change"] == CHANGE_ID
    assert len(json.dumps(first, separators=(",", ":"))) <= 8000


def test_preflight_detects_repository_mutation_mid_read(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = continuity_bridge.git_state
    calls = 0

    def changing_state(*args, **kwargs):
        nonlocal calls
        calls += 1
        state = original(*args, **kwargs)
        return replace(state, fingerprint="f" * 64) if calls == 2 else state

    monkeypatch.setattr(continuity_bridge, "git_state", changing_state)
    result = _preflight(pilot)
    assert any("changed while preflight" in error for error in result["errors"])


def test_wrong_folder_and_missing_drive_block(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = pilot["state"] / "other"
    other.mkdir()
    _git(other, "init", "-b", "other")
    _git(other, "config", "user.email", "other@example.invalid")
    _git(other, "config", "user.name", "Other")
    (other / "seed.txt").write_text("other\n", encoding="utf-8")
    _git(other, "add", "seed.txt")
    _git(other, "commit", "-m", "other")
    wrong = build_preflight(pilot["config"], cwd=other, require_mounted_volume=False)
    assert wrong["status"] == "BLOCKED"
    assert any("wrong repository folder" in error for error in wrong["errors"])

    monkeypatch.setattr(
        continuity_bridge, "external_volume_available", lambda _path: False
    )
    missing = build_preflight(
        pilot["config"], cwd=pilot["repo"], require_mounted_volume=True
    )
    assert missing["completion_allowed"] is False
    assert any("not mounted" in error for error in missing["errors"])


def test_stale_task_missing_card_and_incomplete_spec_block(
    pilot: dict[str, Path],
) -> None:
    task = json.loads(pilot["issues"].read_text(encoding="utf-8"))
    task["status"] = "closed"
    pilot["issues"].write_text(json.dumps(task) + "\n", encoding="utf-8")
    assert any("stale or terminal" in error for error in _preflight(pilot)["errors"])

    pilot["card"].unlink()
    assert any(
        "cannot read authority" in error for error in _preflight(pilot)["errors"]
    )

    pilot["card"].write_text("---\nproject: [unterminated\n---\n", encoding="utf-8")
    assert any("invalid YAML" in error for error in _preflight(pilot)["errors"])

    pilot["card"].write_text(_frontmatter({"project": "Hermes"}), encoding="utf-8")
    pilot["spec"].write_text(_frontmatter({"project": "Hermes"}), encoding="utf-8")
    errors = _preflight(pilot)["errors"]
    assert any("Basic Memory card missing" in error for error in errors)
    assert any("Spec Kit change missing" in error for error in errors)


def test_beads_timeout_uses_degraded_jsonl_and_forbids_completion(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["beads"]["timeout_seconds"] = 0.01
    atomic_write_json(pilot["config"], config)
    monkeypatch.setattr(
        continuity_bridge,
        "run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ContinuityError("bd ready timed out")
        ),
    )
    result = _preflight(pilot)
    assert result["status"] == "DEGRADED"
    assert result["completion_allowed"] is False
    assert "JSONL" in result["warnings"][0]


def test_beads_mutation_uses_completion_timeout(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, float] = {}

    def capture_run(argv, *, cwd, timeout, env):
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    monkeypatch.setattr(continuity_gate, "run_command", capture_run)
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    continuity_gate._run_beads(config, ["close", TASK_ID], pilot["repo"])
    assert captured["timeout"] == 7


def test_beads_mutation_failure_redacts_child_output(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "customer-token=TOP-SECRET"
    monkeypatch.setattr(
        continuity_gate,
        "run_command",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 9, sentinel, sentinel
        ),
    )
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    with pytest.raises(ContinuityError, match="status 9") as exc:
        continuity_gate._run_beads(config, ["close", TASK_ID], pilot["repo"])
    assert sentinel not in str(exc.value)


def test_python_evidence_resolves_binary_but_keeps_virtualenv_imports(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    python = repo / ".venv/bin/python"
    script = repo / "scripts/evidence.py"
    site_packages = repo / (
        f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    python.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (site_packages / "venv_probe.py").write_text(
        "VALUE = 'venv-ok'\n", encoding="utf-8"
    )
    script.write_text("import venv_probe\nprint(venv_probe.VALUE)\n", encoding="utf-8")
    spec = {"argv": [".venv/bin/python", "scripts/evidence.py"]}

    argv = continuity_gate._resolved_argv(spec, repo)
    child_env = continuity_gate.minimal_child_env(
        continuity_gate._python_venv_context(spec, repo)
    )
    python.unlink()
    python.symlink_to("/bin/sh")
    result = subprocess.run(
        argv, cwd=repo, env=child_env, text=True, capture_output=True, check=False
    )

    assert argv == [str(Path(sys.executable).resolve()), str(script)]
    assert result.returncode == 0
    assert result.stdout.strip() == "venv-ok"


def test_interrupted_tool_adjacency_is_rejected() -> None:
    events = [
        {"type": "server_tool_use", "id": "call-1"},
        {"type": "user", "content": "interrupt"},
        {"type": "tool_result", "tool_use_id": "call-1"},
    ]
    assert "interrupted" in validate_tool_adjacency(events)[0]
    assert not validate_tool_adjacency([
        {"type": "tool_use", "id": "call-1"},
        {"type": "tool_result", "tool_use_id": "call-1"},
    ])
    assert not validate_tool_adjacency([
        {"type": "tool_use", "id": "call-1"},
        {"type": "tool_use", "id": "call-2"},
        {"type": "tool_result", "tool_use_id": "call-1"},
        {"type": "tool_result", "tool_use_id": "call-2"},
    ])
    assert validate_tool_adjacency([
        {"type": "tool_use", "id": "call-1"},
        {"type": "tool_use", "id": "call-2"},
        {"type": "tool_result", "tool_use_id": "call-2"},
        {"type": "tool_result", "tool_use_id": "call-1"},
    ])


def test_native_history_preserves_content_block_interruptions() -> None:
    def native_events(history: list[dict]) -> list[dict]:
        events = tool_events_from_payload({"extra": {"conversation_history": history}})
        assert events is not None
        return events

    text_before_result = native_events([
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-1"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "interrupt"},
                {"type": "tool_result", "tool_use_id": "call-1"},
            ],
        },
    ])
    assert validate_tool_adjacency(text_before_result)

    text_between_uses = native_events([
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call-1"},
                {"type": "text", "text": "interrupt"},
                {"type": "tool_use", "id": "call-2"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1"},
                {"type": "tool_result", "tool_use_id": "call-2"},
            ],
        },
    ])
    assert validate_tool_adjacency(text_between_uses)

    ordered_parallel = native_events([
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call-1"},
                {"type": "tool_use", "id": "call-2"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1"},
                {"type": "tool_result", "tool_use_id": "call-2"},
            ],
        },
    ])
    assert not validate_tool_adjacency(ordered_parallel)

    openai_single = native_events([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1"}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "result payload",
        },
    ])
    assert not validate_tool_adjacency(openai_single)

    openai_parallel = native_events([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1"}, {"id": "call-2"}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "first result payload",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "second result payload",
        },
    ])
    assert not validate_tool_adjacency(openai_parallel)


def test_failed_test_and_changed_sha_invalidate_receipt(pilot: dict[str, Path]) -> None:
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["evidence_policy"]["focused_suite"]["argv"] = [
        sys.executable,
        "scripts/evidence_fail.py",
    ]
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "bind failing evidence fixture")
    policy = _policy(pilot)
    failed = create_receipt(
        pilot["config"],
        cwd=pilot["repo"],
        risk_tier="T3",
        lifecycle_target="TESTED",
        commands=[policy["focused_suite"]],
        runtime_checks=policy["runtime_checks"],
        rollback=policy["rollback_check"],
        require_mounted_volume=False,
    )
    assert failed["result"] == "FAIL"

    enforced = _passing_receipt(pilot, target="ENFORCED")
    assert enforced["result"] == "PASS"

    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["evidence_policy"]["focused_suite"]["argv"] = [
        sys.executable,
        "scripts/evidence_pass.py",
    ]
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "restore passing evidence fixture")

    receipt_path = pilot["state"] / "receipt.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot))
    assert verify_receipt(pilot["config"], receipt_path, cwd=pilot["repo"]) == []
    (pilot["repo"] / "seed.txt").write_text("changed\n", encoding="utf-8")
    _git(pilot["repo"], "add", "seed.txt")
    _git(pilot["repo"], "commit", "-m", "change exact SHA")
    assert any(
        "evidence commit is stale" in error for error in _preflight(pilot)["errors"]
    )
    assert any(
        "stale" in error
        for error in verify_receipt(pilot["config"], receipt_path, cwd=pilot["repo"])
    )


def test_signed_receipt_rejects_forged_evidence(pilot: dict[str, Path]) -> None:
    receipt_path = pilot["state"] / "signed.json"
    receipt = _passing_receipt(pilot)
    atomic_write_json(receipt_path, receipt)
    assert verify_receipt(pilot["config"], receipt_path, cwd=pilot["repo"]) == []
    receipt["commands"][0]["test_count"] = 999
    atomic_write_json(receipt_path, receipt)
    assert any(
        "signature is invalid" in error
        for error in verify_receipt(pilot["config"], receipt_path, cwd=pilot["repo"])
    )


def test_full_suite_identity_and_integration_ancestry_are_gate_owned(
    pilot: dict[str, Path],
) -> None:
    with pytest.raises(ContinuityError, match="does not match committed policy"):
        create_receipt(
            pilot["config"],
            cwd=pilot["repo"],
            risk_tier="T3",
            lifecycle_target="ENFORCED",
            commands=[
                {
                    "name": "forged-full",
                    "argv": [sys.executable, "scripts/evidence_pass.py"],
                    "scope": "full",
                }
            ],
            runtime_checks=[],
            rollback={},
            require_mounted_volume=False,
        )

    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["integration_ref"] = "integration"
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "bind integration authority")
    _git(pilot["repo"], "branch", "integration")
    _git(pilot["repo"], "checkout", "integration")
    (pilot["repo"] / "base.txt").write_text("advanced\n", encoding="utf-8")
    _git(pilot["repo"], "add", "base.txt")
    _git(pilot["repo"], "commit", "-m", "advance integration")
    _git(pilot["repo"], "checkout", "pilot")
    with pytest.raises(ContinuityError, match="include the integration SHA"):
        _passing_receipt(pilot)
    assert any("not based" in error for error in _preflight(pilot)["errors"])


def test_only_gate_promotes_terminal_state(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = pilot["state"] / "receipt.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot))
    promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="TESTED")
    result = _preflight(pilot)
    assert result["lifecycle"] == "TESTED"
    assert result["completion_allowed"] is True

    with pytest.raises(Exception, match="only TESTED or ENFORCED"):
        promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ACTIVE")


def test_gate_executes_evidence_and_rejects_legacy_attestations(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ContinuityError, match="risk tier is committed"):
        create_receipt(
            pilot["config"],
            cwd=pilot["repo"],
            risk_tier="T1",
            lifecycle_target="TESTED",
            commands=[
                {
                    "name": "fabricated",
                    "argv": [sys.executable, "scripts/evidence_pass.py"],
                    "passed": True,
                }
            ],
            runtime_checks=[],
            rollback={},
            require_mounted_volume=False,
        )

    policy = _policy(pilot)
    alternate = pilot["state"] / "alternate-config.json"
    alternate.write_text(pilot["config"].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ContinuityError, match="requires committed config"):
        create_receipt(
            alternate,
            cwd=pilot["repo"],
            risk_tier="T3",
            lifecycle_target="TESTED",
            commands=[policy["focused_suite"]],
            runtime_checks=policy["runtime_checks"],
            rollback=policy["rollback_check"],
            require_mounted_volume=False,
        )

    attacker = pilot["state"] / "attacker-repo"
    attacker.mkdir()
    _git(attacker, "init", "-b", "attacker")
    _git(attacker, "config", "user.email", "attacker@example.invalid")
    _git(attacker, "config", "user.name", "Attacker")
    attacker_config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    attacker_config["expected_repo_root"] = str(attacker)
    self_pointing = attacker / ".continuity/config.json"
    atomic_write_json(self_pointing, attacker_config)
    _git(attacker, "add", ".")
    _git(attacker, "commit", "-m", "self-pointing alternate authority")
    with pytest.raises(ContinuityError, match="requires committed config"):
        create_receipt(
            self_pointing,
            cwd=attacker,
            risk_tier="T3",
            lifecycle_target="TESTED",
            commands=[policy["focused_suite"]],
            runtime_checks=policy["runtime_checks"],
            rollback=policy["rollback_check"],
            require_mounted_volume=False,
        )

    (pilot["repo"] / "seed.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ContinuityError, match="clean repository"):
        _passing_receipt(pilot)
    (pilot["repo"] / "seed.txt").write_text("seed\n", encoding="utf-8")

    with pytest.raises(ContinuityError, match="does not match committed policy"):
        create_receipt(
            pilot["config"],
            cwd=pilot["repo"],
            risk_tier="T3",
            lifecycle_target="TESTED",
            commands=[
                {
                    "name": "fabricated",
                    "argv": [sys.executable, "scripts/evidence_pass.py"],
                    "passed": True,
                }
            ],
            runtime_checks=[],
            rollback={},
            require_mounted_volume=False,
        )

    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["evidence_policy"]["focused_suite"]["argv"] = [
        sys.executable,
        "scripts/zero_tests.py",
    ]
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "bind zero-test evidence fixture")
    zero = _passing_receipt(pilot)
    assert zero["result"] == "FAIL"

    monkeypatch.setenv("TOP_SECRET", "must-not-reach-child")
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["evidence_policy"]["focused_suite"]["argv"] = [
        sys.executable,
        "scripts/env_check.py",
        "sk-live-example",
    ]
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "bind minimal environment evidence fixture")
    clean_env = _passing_receipt(pilot)
    assert clean_env["result"] == "PASS"
    assert "stdout" not in clean_env["commands"][0]
    assert "argv" not in clean_env["commands"][0]
    assert "sk-live-example" not in json.dumps(clean_env)


def test_failed_beads_output_is_never_persisted(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "customer-token=TOP-SECRET"
    real_run = continuity_gate.run_command

    def fail_beads_show(argv, **kwargs):
        if argv[:2] == [str(pilot["bd"]), "show"]:
            return subprocess.CompletedProcess(argv, 9, "", sentinel)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(continuity_gate, "run_command", fail_beads_show)
    with pytest.raises(ContinuityError, match="strict authority check failed") as exc:
        _passing_receipt(pilot)
    assert sentinel not in str(exc.value)
    assert not (pilot["state"] / "receipts").exists()
    monkeypatch.setattr(continuity_bridge, "run_command", fail_beads_show)
    assert sentinel not in json.dumps(_preflight(pilot))


def test_evidence_pin_coverage_is_closed(pilot: dict[str, Path]) -> None:
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["evidence_executables"] = []
    atomic_write_json(pilot["config"], config)
    assert any(
        "do not exactly match policy entrypoints" in error
        for error in static_validate(pilot["config"])
    )
    _commit_and_rebind(pilot, "remove evidence pins for negative control")
    with pytest.raises(
        ContinuityError, match="do not exactly match policy entrypoints"
    ):
        _passing_receipt(pilot)

    config["evidence_executables"] = [{"path": "unrelated", "sha256": "0" * 64}]
    atomic_write_json(pilot["config"], config)
    assert any(
        "do not exactly match policy entrypoints" in error
        for error in static_validate(pilot["config"])
    )


def test_pinned_inputs_invalidate_preflight_and_receipt(pilot: dict[str, Path]) -> None:
    instruction = pilot["state"] / "external-skill.md"
    instruction.write_text("v1\n", encoding="utf-8")
    config = json.loads(pilot["config"].read_text(encoding="utf-8"))
    config["external_instructions"] = [
        {"path": str(instruction), "sha256": sha256_file(instruction)}
    ]
    atomic_write_json(pilot["config"], config)
    _commit_and_rebind(pilot, "bind external instruction fixture")
    receipt_path = pilot["state"] / "receipt.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot))
    instruction.write_text("v2\n", encoding="utf-8")
    assert any(
        "external instruction" in error
        for error in verify_receipt(pilot["config"], receipt_path, cwd=pilot["repo"])
    )

    pilot["bd"].write_text(pilot["bd"].read_text(encoding="utf-8") + "\n")
    assert any("digest mismatch" in error for error in _preflight(pilot)["errors"])


def test_black_box_dispatch_and_host_block_shapes(pilot: dict[str, Path]) -> None:
    entry = REPO_ROOT / ".specify/events.py"
    for surface in ("claude", "codex", "hermes"):
        result = subprocess.run(
            [
                sys.executable,
                str(entry),
                "session_start" if surface != "hermes" else "on_session_start",
                "--surface",
                surface,
                "--config",
                str(pilot["config"]),
            ],
            cwd=pilot["repo"],
            input=json.dumps({"session_id": "black-box"}),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout).get("additional_context")

    interrupted = {
        "hook_event_name": "pre_llm_call",
        "session_id": "interrupted-hermes",
        "extra": {
            "turn_id": "interrupted-turn",
            "conversation_history": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1"}],
                },
                {"role": "user", "content": "interrupt"},
                {"role": "tool", "tool_call_id": "call-1", "content": "late"},
            ],
        },
    }
    hermes_llm = subprocess.run(
        [
            sys.executable,
            str(entry),
            "pre_llm_call",
            "--surface",
            "hermes",
            "--config",
            str(pilot["config"]),
        ],
        cwd=pilot["repo"],
        input=json.dumps(interrupted),
        text=True,
        capture_output=True,
        check=False,
    )
    assert hermes_llm.returncode == 0
    assert json.loads(hermes_llm.stdout)["completion_allowed"] is False
    hermes_process = subprocess.run(
        [
            sys.executable,
            str(entry),
            "pre_tool_call",
            "--surface",
            "hermes",
            "--config",
            str(pilot["config"]),
        ],
        cwd=pilot["repo"],
        input=json.dumps({
            "session_id": "interrupted-hermes",
            "extra": {"turn_id": "interrupted-turn"},
        }),
        text=True,
        capture_output=True,
        check=False,
    )
    assert hermes_process.returncode == 2
    assert json.loads(hermes_process.stdout)["action"] == "block"

    healthy_llm = subprocess.run(
        [
            sys.executable,
            str(entry),
            "pre_llm_call",
            "--surface",
            "hermes",
            "--config",
            str(pilot["config"]),
        ],
        cwd=pilot["repo"],
        input=json.dumps({
            "session_id": "healthy-hermes",
            "extra": {
                "turn_id": "healthy-turn",
                "conversation_history": [{"role": "user", "content": "healthy"}],
            },
        }),
        text=True,
        capture_output=True,
        check=False,
    )
    assert healthy_llm.returncode == 0
    for _ in range(2):
        healthy_tool = subprocess.run(
            [
                sys.executable,
                str(entry),
                "pre_tool_call",
                "--surface",
                "hermes",
                "--config",
                str(pilot["config"]),
            ],
            cwd=pilot["repo"],
            input=json.dumps({
                "session_id": "healthy-hermes",
                "extra": {"turn_id": "healthy-turn"},
            }),
            text=True,
            capture_output=True,
            check=False,
        )
        assert healthy_tool.returncode == 0
        assert json.loads(healthy_tool.stdout).get("action") != "block"

    cross_turn = subprocess.run(
        [
            sys.executable,
            str(entry),
            "pre_tool_call",
            "--surface",
            "hermes",
            "--config",
            str(pilot["config"]),
        ],
        cwd=pilot["repo"],
        input=json.dumps({
            "session_id": "healthy-hermes",
            "extra": {"turn_id": "other-turn"},
        }),
        text=True,
        capture_output=True,
        check=False,
    )
    assert cross_turn.returncode == 2
    assert json.loads(cross_turn.stdout)["action"] == "block"

    pilot["card"].unlink()
    for surface in ("claude", "codex"):
        stopped = subprocess.run(
            [
                sys.executable,
                str(entry),
                "stop",
                "--surface",
                surface,
                "--config",
                str(pilot["config"]),
            ],
            cwd=pilot["repo"],
            input="{}",
            text=True,
            capture_output=True,
            check=False,
        )
        assert stopped.returncode == 0
        assert json.loads(stopped.stdout)["decision"] == "block"
    audit = pilot["events"].read_text(encoding="utf-8")
    assert '"preflight_status":"BLOCKED"' in audit


def test_hermes_adapter_has_owned_reversible_before_image(
    pilot: dict[str, Path], tmp_path: Path
) -> None:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    original = {
        "model": "fixture",
        "hooks": {
            "pre_tool_call": [{"command": "existing", "timeout": 3}],
            "unrelated": [{"command": "preserve-me"}],
        },
    }
    config_path = hermes_home / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")
    installed = install_hermes_adapter(REPO_ROOT, hermes_home)
    assert installed["changed"] is True
    active = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert active["hooks"]["pre_tool_call"][0]["fail_closed"] is True
    assert (
        rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=False)["rollback_valid"]
        is True
    )
    rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=True)
    restored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert restored == original


def test_interrupted_hermes_install_remains_rollback_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    hermes_home = tmp_path / "interrupted-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    original = {"model": "fixture", "hooks": {"unrelated": [{"command": "keep"}]}}
    config_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    real_write = continuity_adapters.atomic_write_text

    def interrupt_final_manifest(path: Path, content: str) -> None:
        if (
            path.name == continuity_adapters.HERMES_MANIFEST
            and '"INSTALLED"' in content
        ):
            raise OSError("simulated manifest finalization interruption")
        real_write(path, content)

    monkeypatch.setattr(
        continuity_adapters, "atomic_write_text", interrupt_final_manifest
    )
    with pytest.raises(OSError, match="finalization interruption"):
        install_hermes_adapter(REPO_ROOT, hermes_home)
    monkeypatch.setattr(continuity_adapters, "atomic_write_text", real_write)
    assert rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=False)[
        "rollback_valid"
    ]
    rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=True)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == original


def test_interrupted_hermes_rollback_finalization_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    hermes_home = tmp_path / "rollback-interrupted-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    original = {"model": "fixture", "hooks": {"unrelated": [{"command": "keep"}]}}
    config_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    install_hermes_adapter(REPO_ROOT, hermes_home)
    real_write = continuity_adapters.atomic_write_text

    def interrupt_rolled_back_manifest(path: Path, content: str) -> None:
        if (
            path.name == continuity_adapters.HERMES_MANIFEST
            and '"ROLLED_BACK"' in content
        ):
            raise OSError("simulated rollback finalization interruption")
        real_write(path, content)

    monkeypatch.setattr(
        continuity_adapters, "atomic_write_text", interrupt_rolled_back_manifest
    )
    with pytest.raises(OSError, match="rollback finalization interruption"):
        rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=True)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == original

    monkeypatch.setattr(continuity_adapters, "atomic_write_text", real_write)
    assert rollback_hermes_adapter(REPO_ROOT, hermes_home, apply=True)["applied"]


def test_promotion_recovers_prepared_and_card_written_stages(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = pilot["state"] / "enforced-stages.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot, target="ENFORCED"))
    real_card_write = continuity_gate.atomic_write_text

    def interrupt_card(path: Path, content: str) -> None:
        if path == pilot["card"]:
            raise OSError("simulated card interruption")
        real_card_write(path, content)

    monkeypatch.setattr(continuity_gate, "atomic_write_text", interrupt_card)
    with pytest.raises(OSError, match="card interruption"):
        promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    journal = json.loads((pilot["state"] / "promotion.json").read_text())
    assert journal["status"] == "PREPARED"

    monkeypatch.setattr(continuity_gate, "atomic_write_text", real_card_write)
    real_journal_write = continuity_gate.atomic_write_json

    def interrupt_card_written(path: Path, value: dict) -> None:
        real_journal_write(path, value)
        if path.name == "promotion.json" and value.get("status") == "CARD_WRITTEN":
            raise KeyboardInterrupt("simulated process death")

    monkeypatch.setattr(continuity_gate, "atomic_write_json", interrupt_card_written)
    with pytest.raises(KeyboardInterrupt, match="process death"):
        promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    journal = json.loads((pilot["state"] / "promotion.json").read_text())
    assert journal["status"] == "CARD_WRITTEN"

    monkeypatch.setattr(continuity_gate, "atomic_write_json", real_journal_write)
    promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    assert _preflight(pilot)["status"] == "AVAILABLE"


def test_enforced_promotion_verifies_beads_close_effect(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = pilot["state"] / "enforced-noop.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot, target="ENFORCED"))
    monkeypatch.setattr(continuity_gate, "_run_beads", lambda *_args, **_kwargs: None)
    with pytest.raises(ContinuityError, match="did not reach closed"):
        promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    journal = json.loads((pilot["state"] / "promotion.json").read_text())
    assert journal["status"] == "RECOVERY_REQUIRED"


def test_promotion_lock_serializes_processes(pilot: dict[str, Path]) -> None:
    marker = pilot["state"] / "lock-order.txt"
    context = multiprocessing.get_context("fork")
    first = context.Process(
        target=_lock_worker,
        args=(str(pilot["config"]), str(marker), "first"),
    )
    second = context.Process(
        target=_lock_worker,
        args=(str(pilot["config"]), str(marker), "second"),
    )
    first.start()
    time.sleep(0.05)
    second.start()
    first.join(5)
    second.join(5)
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "first-start",
        "first-end",
        "second-start",
        "second-end",
    ]


def test_enforced_promotion_closes_task_and_recovers_interruption(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = pilot["state"] / "enforced.json"
    atomic_write_json(receipt_path, _passing_receipt(pilot, target="ENFORCED"))
    real_run_beads = continuity_gate._run_beads
    monkeypatch.setattr(
        continuity_gate,
        "_run_beads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ContinuityError("simulated close interruption")
        ),
    )
    with pytest.raises(ContinuityError, match="simulated close interruption"):
        promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    assert any("RECOVERY_REQUIRED" in error for error in _preflight(pilot)["errors"])

    monkeypatch.setattr(continuity_gate, "_run_beads", real_run_beads)
    promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")
    result = _preflight(pilot)
    assert result["status"] == "AVAILABLE"
    assert result["lifecycle"] == "ENFORCED"
    assert result["task_status"] == "closed"
    promote(pilot["config"], receipt_path, cwd=pilot["repo"], target="ENFORCED")


def test_dispatcher_logs_metadata_not_payload(pilot: dict[str, Path]) -> None:
    payload = {"session_id": "session-secret", "tool_input": {"token": "TOP-SECRET"}}
    result = dispatch("post_tool", "hermes", pilot["config"], payload)
    assert result["handled"] is True
    log = pilot["events"].read_text(encoding="utf-8")
    assert "TOP-SECRET" not in log
    assert "session-secret" not in log
    assert "post_tool" in log


def test_unmounted_volume_causes_no_state_writes(
    pilot: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import continuity_event

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("state write attempted while volume unavailable")

    monkeypatch.setattr(continuity_event, "external_volume_available", lambda _p: False)
    monkeypatch.setattr(continuity_event, "atomic_write_json", forbidden_write)
    monkeypatch.setattr(continuity_event, "_append_redacted_event", forbidden_write)
    dispatch(
        "pre_llm_call",
        "hermes",
        pilot["config"],
        {
            "session_id": "unmounted",
            "extra": {"turn_id": "unmounted-turn", "conversation_history": []},
        },
    )

    monkeypatch.setattr(continuity_gate, "external_volume_available", lambda _p: False)
    policy = _policy(pilot)
    with pytest.raises(ContinuityError, match="external volume is not mounted"):
        create_receipt(
            pilot["config"],
            cwd=pilot["repo"],
            risk_tier="T3",
            lifecycle_target="TESTED",
            commands=[policy["focused_suite"]],
            runtime_checks=policy["runtime_checks"],
            rollback=policy["rollback_check"],
            require_mounted_volume=True,
        )
    assert not (pilot["state"] / "receipt-signing.key").exists()
    assert not (pilot["state"] / "evidence-home").exists()
    with pytest.raises(ContinuityError, match="external volume is not mounted"):
        promote(
            pilot["config"],
            pilot["state"] / "not-read.json",
            cwd=pilot["repo"],
            target="TESTED",
        )
    assert not (pilot["state"] / "promotion.lock").exists()


def test_committed_adapters_match_single_contract() -> None:
    for path, rendered in render_project_adapters(REPO_ROOT).items():
        assert path.read_text(encoding="utf-8") == rendered


def test_supply_chain_manifest_membership_is_closed() -> None:
    installed = ["claude", "codex", "hermes"]
    assert not manifest_membership_errors(
        installed, ["claude", "codex", "hermes", "speckit"]
    )
    assert any(
        "membership mismatch" in error
        for error in manifest_membership_errors(installed, ["claude"])
    )
    assert any(
        "duplicated" in error
        for error in manifest_membership_errors(
            installed, ["claude", "codex", "hermes", "speckit", "claude"]
        )
    )
    assert not managed_manifest_file_errors({"one", "two"}, {"one", "two"})
    assert any(
        "missing from manifests" in error
        for error in managed_manifest_file_errors({"one"}, {"one", "two"})
    )
    assert any(
        "outside managed" in error
        for error in managed_manifest_file_errors({"one", "extra"}, {"one"})
    )


def test_continuity_workflow_covers_authority_and_supply_chain_paths() -> None:
    workflow = (REPO_ROOT / ".github/workflows/continuity-gate.yml").read_text(
        encoding="utf-8"
    )
    for protected in (
        ".agents/skills/speckit-*/**",
        ".claude/skills/speckit-*/**",
        "scripts/run_tests.sh",
        "AGENTS.md",
        ".github/workflows/continuity-gate.yml",
        ".github/actions/retry/**",
        "pyproject.toml",
        "uv.lock",
        "docs/continuity-pilot.md",
    ):
        assert workflow.count(f"- '{protected}'") == 2
    assert "astral-sh/setup-uv@fac544c07dec837d0ccb6301d7b5580bf5edae39" in workflow
    assert "uv sync --locked --python 3.11 --extra dev" in workflow
    assert ".venv/bin/python scripts/continuity_gate.py static" in workflow
    assert "scripts/run_tests.sh tests/test_continuity_control_plane.py" in workflow
    patterns = {
        stripped[3:-1]
        for line in workflow.splitlines()
        if (stripped := line.strip()).startswith("- '") and stripped.endswith("'")
    }
    config = json.loads((REPO_ROOT / ".continuity/config.json").read_text())
    protected_paths = set(config["instructions"])
    policy = config["evidence_policy"]
    for spec in [
        policy["focused_suite"],
        policy["full_suite"],
        *policy["runtime_checks"],
        policy["rollback_check"],
    ]:
        if len(spec["argv"]) > 1 and not Path(spec["argv"][1]).is_absolute():
            protected_paths.add(spec["argv"][1])
    for manifest_path in (REPO_ROOT / ".specify/integrations").glob("*.manifest.json"):
        protected_paths.update(json.loads(manifest_path.read_text())["files"])
    assert all(
        any(fnmatch(path, pattern) for pattern in patterns) for path in protected_paths
    )
