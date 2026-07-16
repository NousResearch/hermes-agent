"""Verifier-supervisor tests for protected merge recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/65671"
HEAD_SHA = "abc123def456"
BASE_BRANCH = "main"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _contract() -> dict:
    return {
        "pr_url": PR_URL,
        "approved_head": HEAD_SHA,
        "approved_base": BASE_BRANCH,
        "readiness_evidence": {"ready": True},
    }


def _claim_with_authorization(conn, *, task_id: str = "t_verify") -> tuple[kb.Task, dict, int]:
    task_id = kb.create_task(conn, title="verify protected merge", assignee="alice")
    claim = kb.claim_task(conn, task_id)
    assert claim is not None
    claim.verification_contract = _contract()
    binding = kb.make_verifier_result_binding(claim, _contract(), nonce="nonce-stage2")
    auth = kb.authorize_verifier_result(conn, **binding)
    assert auth.state == "authorized"
    assert kb.reserve_verifier_result(conn, authorization_id=auth.id, **binding).state == "reserved"
    assert kb.start_verifier_result(conn, authorization_id=auth.id, **binding).state == "started"
    return claim, binding, auth.id


def _claim_with_reserved_authorization(conn) -> tuple[kb.Task, dict, int]:
    task_id = kb.create_task(conn, title="verify protected merge", assignee="alice")
    claim = kb.claim_task(conn, task_id)
    assert claim is not None
    claim.verification_contract = _contract()
    binding = kb.make_verifier_result_binding(claim, _contract(), nonce="nonce-stage3")
    auth = kb.authorize_verifier_result(conn, **binding)
    assert auth.state == "authorized"
    assert kb.reserve_verifier_result(conn, authorization_id=auth.id, **binding).state == "reserved"
    return claim, binding, auth.id


def _frame(binding: dict, **overrides) -> bytes:
    payload = {
        "version": 1,
        "task_id": binding["task_id"],
        "run_id": binding["run_id"],
        "claim_lock": binding["claim_lock"],
        "contract_id": binding["contract_id"],
        "contract_hash": binding["contract_hash"],
        "pr_url": binding["pr_url"],
        "approved_head": binding["approved_head"],
        "approved_base": binding["approved_base"],
        "nonce": binding["nonce"],
        "action": "complete",
        "summary": "verified and complete",
    }
    payload.update(overrides)
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def test_result_frame_complete_applies_once_to_bound_claim(kanban_home):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)

        result = kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding),
        )

        assert result.action == "complete"
        assert kb.get_task(conn, claim.id).status == "done"
        assert kb.latest_summary(conn, claim.id) == "verified and complete"
        assert kb.consume_verifier_result(
            conn, authorization_id=authorization_id, **binding
        ).reason == "already_applied"


def test_result_frame_reblock_maps_to_needs_input_block(kanban_home):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)

        result = kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(
                binding,
                action="re-block",
                summary="merge no longer matches approval",
                reason="head changed after approval",
            ),
        )

        task = kb.get_task(conn, claim.id)
        assert result.action == "re-block"
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert kb.latest_summary(conn, claim.id) == "head changed after approval"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"task_id": "other"}, "binding_mismatch:task_id"),
        ({"run_id": 999}, "binding_mismatch:run_id"),
        ({"claim_lock": "other"}, "binding_mismatch:claim_lock"),
        ({"contract_id": "other"}, "binding_mismatch:contract_id"),
        ({"contract_hash": "0" * 64}, "binding_mismatch:contract_hash"),
        ({"pr_url": "https://github.com/NousResearch/hermes-agent/pull/1"}, "binding_mismatch:pr_url"),
        ({"approved_head": "other"}, "binding_mismatch:approved_head"),
        ({"approved_base": "other"}, "binding_mismatch:approved_base"),
        ({"nonce": "other"}, "binding_mismatch:nonce"),
    ],
)
def test_result_frame_binding_failures_do_not_mutate(kanban_home, overrides, reason):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)

        result = kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding, **overrides),
        )

        assert result.reason == reason
        assert kb.get_task(conn, claim.id).status == "running"


@pytest.mark.parametrize(
    "raw, reason",
    [
        (b"", "no_frame"),
        (b'{"version":1', "partial_frame"),
        (b'{"version":1}\n{"version":1}\n', "multiple_frames"),
        (b'{"version":1,"version":1}\n', "duplicate_key:version"),
        (b'{"version":1,"task_id":"t","extra":true}\n', "unknown_field:extra"),
        ("not utf8 \udcff\n".encode("utf-8", "surrogatepass"), "invalid_utf8"),
    ],
)
def test_result_frame_rejects_malformed_partial_unknown_duplicate(raw, reason):
    result = kb.parse_verifier_result_frame(raw, max_bytes=256)
    assert result.valid is False
    assert result.reason == reason


def test_result_frame_rejects_oversized_payload():
    raw = b'{"version":1,"task_id":"' + (b"x" * 300) + b'"}\n'
    result = kb.parse_verifier_result_frame(raw, max_bytes=128)
    assert result.valid is False
    assert result.reason == "oversized"


@pytest.mark.parametrize("field", ["summary", "reason"])
def test_result_frame_rejects_oversized_text_fields(field):
    payload = {
        "version": 1,
        "task_id": "t_verify",
        "run_id": 1,
        "claim_lock": "claim",
        "contract_id": kb.VERIFIER_RESULT_CONTRACT_ID,
        "contract_hash": "a" * 64,
        "pr_url": PR_URL,
        "approved_head": HEAD_SHA,
        "approved_base": BASE_BRANCH,
        "nonce": "nonce",
        "action": "complete",
        field: "x" * (kb._CTX_MAX_FIELD_BYTES + 1),
    }
    result = kb.parse_verifier_result_frame(
        (json.dumps(payload) + "\n").encode("utf-8")
    )
    assert result.valid is False
    assert result.reason == f"field_too_large:{field}"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"run_id": 0}, "invalid_value:run_id"),
        ({"run_id": -1}, "invalid_value:run_id"),
        ({"action": True}, "invalid_type:action"),
        ({"action": "COMPLETE"}, "invalid_action"),
        ({"action": "reblock"}, "invalid_action"),
        ({"contract_hash": "not-sha256"}, "invalid_value:contract_hash"),
    ],
)
def test_result_frame_rejects_strict_contract_values(overrides, reason):
    binding = {
        "task_id": "t_verify",
        "run_id": 1,
        "claim_lock": "claim",
        "contract_id": kb.VERIFIER_RESULT_CONTRACT_ID,
        "contract_hash": "a" * 64,
        "pr_url": PR_URL,
        "approved_head": HEAD_SHA,
        "approved_base": BASE_BRANCH,
        "nonce": "nonce",
    }

    result = kb.parse_verifier_result_frame(_frame(binding, **overrides))

    assert result.valid is False
    assert result.reason == reason


def test_result_frame_rejects_trailing_bytes_after_newline():
    binding = {
        "task_id": "t_verify",
        "run_id": 1,
        "claim_lock": "claim",
        "contract_id": kb.VERIFIER_RESULT_CONTRACT_ID,
        "contract_hash": "a" * 64,
        "pr_url": PR_URL,
        "approved_head": HEAD_SHA,
        "approved_base": BASE_BRANCH,
        "nonce": "nonce",
    }

    result = kb.parse_verifier_result_frame(_frame(binding) + b"trailing")

    assert result.valid is False
    assert result.reason == "trailing_bytes"


def test_result_pipe_waits_for_eof_before_accepting_a_frame():
    read_fd, writer_fd = os.pipe()

    def write_late_trailing_bytes():
        os.write(writer_fd, b'{"version":1}\n')
        time.sleep(0.05)
        os.write(writer_fd, b"late")
        os.close(writer_fd)

    writer = threading.Thread(target=write_late_trailing_bytes)
    writer.start()
    raw = kb._read_verifier_result_pipe(read_fd, deadline_seconds=2, max_bytes=256)
    writer.join(timeout=1)

    assert raw == b'{"version":1}\nlate'
    assert kb.parse_verifier_result_frame(raw, max_bytes=256).reason == "trailing_bytes"


@pytest.mark.parametrize("legacy_verdict", ["approve", "reject"])
def test_verifier_result_tool_rejects_legacy_verdict_names(
    kanban_home,
    monkeypatch,
    legacy_verdict,
):
    monkeypatch.setenv("HERMES_KANBAN_VERIFY_ONLY", "1")
    from tools.kanban_verifier_result_tool import _handle_verifier_result

    result = _handle_verifier_result(legacy_verdict, "summary", "reason")

    assert "must be one of" in result
    assert "approved" in result


def _wait_for_status(task_id: str, status: str, *, timeout: float = 2.0) -> kb.Task:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        with kb.connect() as conn:
            last = kb.get_task(conn, task_id)
            if last and last.status == status:
                return last
        time.sleep(0.02)
    assert last is not None
    raise AssertionError(f"expected {task_id} to reach {status}, got {last.status}")


def _wait_for_missing(path: Path, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"expected {path} to be removed")


@pytest.mark.parametrize("returncode", [1, -9])
def test_supervisor_nonzero_or_signal_reblocks_without_result_frame(
    kanban_home,
    returncode,
):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    read_fd, writer_fd = os.pipe()
    os.close(writer_fd)

    class FakeProc:
        def wait(self, timeout=None):
            return returncode

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
    )

    task = _wait_for_status(claim.id, "blocked")
    with kb.connect() as conn:
        summary = kb.latest_summary(conn, claim.id) or ""
    assert "verifier result channel failed" in summary


def test_supervisor_timeout_reblocks_and_kills_child(kanban_home):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    read_fd, writer_fd = os.pipe()
    os.close(writer_fd)
    killed = {"value": False}

    class FakeProc:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("verifier", timeout)

        def kill(self):
            killed["value"] = True

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
    )

    _wait_for_status(claim.id, "blocked")
    assert killed["value"] is True


def test_supervisor_apply_exception_reblocks_fail_closed(kanban_home, monkeypatch):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    read_fd, writer_fd = os.pipe()
    os.write(writer_fd, _frame(binding))
    os.close(writer_fd)

    class FakeProc:
        def wait(self, timeout=None):
            return 0

    def boom(*args, **kwargs):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(kb, "apply_verifier_supervisor_result", boom)

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
    )

    task = _wait_for_status(claim.id, "blocked")
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
        summary = kb.latest_summary(conn, claim.id) or ""

    assert task.block_kind == "needs_input"
    assert row["state"] == "reconciled"
    assert row["reason_code"] == "apply_exception"
    assert "verifier result channel failed: apply_exception" in summary


def test_timeout_wins_race_before_late_valid_result_can_apply(kanban_home):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    read_fd, writer_fd = os.pipe()
    os.close(writer_fd)

    class FakeProc:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("verifier", timeout)

        def kill(self):
            pass

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
    )

    _wait_for_status(claim.id, "blocked")
    with kb.connect() as conn:
        late = kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding),
        )
        row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
        task = kb.get_task(conn, claim.id)

    assert late.ok is False
    assert late.reason == "already_reconciled"
    assert row["state"] == "reconciled"
    assert row["reason_code"] == "timeout"
    assert task.status == "blocked"


def test_supervisor_valid_result_cleans_verifier_root(kanban_home):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    verifier_root = kb._create_verifier_root(board=None)
    assert verifier_root.is_dir()
    read_fd, writer_fd = os.pipe()
    os.write(writer_fd, _frame(binding))
    os.close(writer_fd)

    class FakeProc:
        def wait(self, timeout=None):
            return 0

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
        verifier_root=verifier_root,
    )

    _wait_for_status(claim.id, "done")
    _wait_for_missing(verifier_root)


def test_verifier_root_cleanup_failure_is_audited_and_retried(
    kanban_home,
    monkeypatch,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    verifier_root = kb._create_verifier_root(board=None)
    read_fd, writer_fd = os.pipe()
    os.write(writer_fd, _frame(binding))
    os.close(writer_fd)
    calls = {"count": 0}
    real_rmtree = kb.shutil.rmtree

    def flaky_rmtree(path):
        calls["count"] += 1
        if Path(path) == verifier_root and calls["count"] == 1:
            raise OSError("busy")
        return real_rmtree(path)

    class FakeProc:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(kb.shutil, "rmtree", flaky_rmtree)
    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
        verifier_root=verifier_root,
    )

    _wait_for_status(claim.id, "done")
    assert verifier_root.exists()
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT cleanup_error FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
        events = [
            e for e in kb.list_events(conn, claim.id)
            if e.kind == "verifier_root_cleanup_failed"
        ]
        retried = kb.reconcile_verifier_roots(conn, board=None)

    assert "busy" in row["cleanup_error"]
    assert events[-1].payload["authorization_id"] == authorization_id
    assert "nonce-stage2" not in json.dumps(events[-1].payload)
    assert retried == [authorization_id]
    assert not verifier_root.exists()


def test_orphan_verifier_root_reconciliation_unlinks_symlink_without_following(
    kanban_home,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    parent = kb._verifier_roots_parent(board=None)
    parent.mkdir(parents=True)
    orphan = parent / "run-orphan"
    orphan.mkdir(mode=0o700)
    outside = kanban_home / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    symlink = parent / "run-symlink"
    try:
        symlink.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this platform")

    removed = kb.reconcile_orphan_verifier_roots(board=None)

    assert sorted(removed) == ["run-orphan", "run-symlink"]
    assert not orphan.exists()
    assert not symlink.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_cleanup_rejects_same_named_parent_outside_expected_verifier_roots(
    kanban_home,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
        assert kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding),
        ).ok

    attacker_parent = kanban_home.parent / "attacker" / "verifier-roots"
    attacker_root = attacker_parent / "run-outside"
    attacker_root.mkdir(parents=True)
    sentinel = attacker_root / "sentinel.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    kb._cleanup_verifier_root(
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        task_id=claim.id,
        run_id=int(binding["run_id"]),
        verifier_root=attacker_root,
    )

    assert sentinel.read_text(encoding="utf-8") == "must survive"
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT cleanup_error FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
    assert row["cleanup_error"]
    assert "refusing" in row["cleanup_error"]


def test_reconcile_root_retry_rejects_outside_same_named_parent(
    kanban_home,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
        assert kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding),
        ).ok

    attacker_root = kanban_home.parent / "attacker" / "verifier-roots" / "run-outside"
    attacker_root.mkdir(parents=True)
    sentinel = attacker_root / "sentinel.txt"
    sentinel.write_text("must survive retry", encoding="utf-8")
    with kb.connect() as conn:
        conn.execute(
            "UPDATE verifier_result_authorizations "
            "SET cleanup_error = 'prior failure', verifier_root = ? WHERE id = ?",
            (str(attacker_root), authorization_id),
        )
        retried = kb.reconcile_verifier_roots(conn, board=None)
        row = conn.execute(
            "SELECT cleanup_error FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()

    assert retried == []
    assert sentinel.read_text(encoding="utf-8") == "must survive retry"
    assert "refusing" in row["cleanup_error"]


def test_windows_supervisor_timeout_does_not_block_on_pipe_read(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    read_fd, writer_fd = os.pipe()
    os.close(writer_fd)
    killed = {"value": False}

    def fake_read(read_fd_arg, **kwargs):
        raise AssertionError("windows timeout path must not read before waiting")

    class FakeProc:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("verifier", timeout)

        def kill(self):
            killed["value"] = True

    monkeypatch.setattr(kb, "_read_verifier_result_pipe", fake_read)

    kb._start_verifier_supervisor_waiter(
        read_fd=read_fd,
        proc=FakeProc(),
        db_path=kb.kanban_db_path(),
        authorization_id=authorization_id,
        binding=binding,
    )

    _wait_for_status(claim.id, "blocked")
    assert killed["value"] is True


def test_valid_frame_losing_live_claim_race_does_not_apply(kanban_home):
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
        conn.execute(
            "UPDATE task_runs SET status='reclaimed' WHERE id=?",
            (claim.current_run_id,),
        )
        result = kb.apply_verifier_supervisor_result(
            conn,
            authorization_id=authorization_id,
            binding=binding,
            raw_frame=_frame(binding),
        )

        assert result.ok is False
        assert result.reason == "stale_run"
        assert kb.get_task(conn, claim.id).status == "running"


def test_default_spawn_verifier_uses_posix_inherited_one_shot_fd(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    monkeypatch.setattr(kb, "_start_verifier_supervisor_waiter", lambda **_: None, raising=False)
    captured = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, *args, **kwargs):
        pass_fds = kwargs.get("pass_fds")
        assert pass_fds and len(pass_fds) == 1
        writer_fd = pass_fds[0]
        assert os.get_inheritable(writer_fd) is True
        captured.update(cmd=cmd, env=kwargs["env"], kwargs=kwargs, writer_fd=writer_fd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()

    pid = kb._default_spawn(task, str(kanban_home), board=None)

    env = captured["env"]
    assert pid == 12345
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT state, supervisor_owner FROM verifier_result_authorizations WHERE task_id = ?",
            (task.id,),
        ).fetchone()
    assert row["state"] == "started"
    assert row["supervisor_owner"] == f"{kb.socket.gethostname()}:12345"
    assert env["HERMES_KANBAN_VERIFY_RESULT_FD"] == str(captured["writer_fd"])
    assert env["HERMES_KANBAN_VERIFY_ONLY"] == "1"
    assert env["HERMES_KANBAN_VERIFY_CONTRACT_HASH"]
    verifier_home = Path(env["HERMES_HOME"])
    assert verifier_home.parent.name == "verifier-roots"
    assert verifier_home.is_dir()
    assert (verifier_home / "config.yaml").is_file()
    assert (verifier_home.stat().st_mode & 0o777) == 0o700
    assert Path(env["HOME"]) == verifier_home
    assert Path(env["TMPDIR"]).is_relative_to(verifier_home)
    assert captured["kwargs"]["cwd"] == str(verifier_home)
    assert "HERMES_PROFILE" not in env
    assert "HERMES_ACCEPT_HOOKS" not in env
    assert "HERMES_KANBAN_DB" not in env
    assert "HERMES_KANBAN_BOARD" not in env
    assert "HERMES_KANBAN_WORKSPACES_ROOT" not in env
    assert "HERMES_KANBAN_TASK" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "--toolsets" in captured["cmd"]
    toolsets_index = captured["cmd"].index("--toolsets")
    assert captured["cmd"][toolsets_index + 1] == "kanban_verifier"
    assert "terminal" not in captured["cmd"]
    assert "file" not in captured["cmd"]
    assert "chat" in captured["cmd"]
    assert "-q" in captured["cmd"]
    assert "-Q" in captured["cmd"]
    assert "kanban_verifier_result" not in captured["cmd"]
    assert "-p" not in captured["cmd"]
    assert "--profile" not in captured["cmd"]
    assert "--accept-hooks" not in captured["cmd"]
    assert captured["kwargs"]["close_fds"] is True

    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_KANBAN_VERIFY_ONLY", "1")
        scoped.delenv("HERMES_KANBAN_TASK", raising=False)
        from model_tools import get_tool_definitions
        from tools.registry import invalidate_check_fn_cache

        invalidate_check_fn_cache()
        names = {
            t["function"]["name"]
            for t in get_tool_definitions(
                enabled_toolsets=["kanban_verifier"],
                quiet_mode=True,
            )
        }
    assert names == {"kanban_verifier_result"}
    assert not {
        "terminal",
        "read_file",
        "write_file",
        "patch",
        "execute_code",
        "kanban_show",
        "kanban_complete",
        "kanban_block",
        "kanban_create",
    } & names


def test_default_spawn_verifier_closes_parent_log_handle(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    monkeypatch.setattr(kb, "_start_verifier_supervisor_waiter", lambda **_: None, raising=False)
    opened = {}

    class FakeLog:
        closed = False

        def close(self):
            self.closed = True

    class FakeProc:
        pid = 12345

    real_open = open

    def fake_open(*args, **kwargs):
        if args and not str(args[0]).endswith(".log"):
            return real_open(*args, **kwargs)
        log = FakeLog()
        opened.setdefault("logs", []).append(log)
        opened["log"] = log
        return log

    def fake_popen(cmd, *args, **kwargs):
        assert kwargs["stdout"] is opened["log"]
        return FakeProc()

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()

    kb._default_spawn(task, str(kanban_home), board=None)

    assert any(log.closed for log in opened["logs"])


def test_default_spawn_verifier_cleans_root_when_popen_fails(
    kanban_home,
    monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    monkeypatch.setattr(kb, "_start_verifier_supervisor_waiter", lambda **_: None, raising=False)
    created_roots = []
    real_create = kb._create_verifier_root

    def tracking_create(*, board):
        root = real_create(board=board)
        created_roots.append(root)
        return root

    def fake_popen(cmd, *args, **kwargs):
        raise FileNotFoundError("missing hermes")

    monkeypatch.setattr(kb, "_create_verifier_root", tracking_create)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()

    with pytest.raises(RuntimeError, match="executable not found"):
        kb._default_spawn(task, str(kanban_home), board=None)

    assert created_roots
    assert all(not root.exists() for root in created_roots)
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT state, reason_code, supervisor_owner, verifier_root "
            "FROM verifier_result_authorizations WHERE task_id = ?",
            (task.id,),
        ).fetchone()
    assert row["state"] == "reconciled"
    assert row["reason_code"] == "pre_start_restore"
    assert row["supervisor_owner"] is None
    assert row["verifier_root"] is None


def test_default_spawn_verifier_does_not_create_root_when_authorization_denied(
    kanban_home,
    monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    created = {"count": 0}

    def forbidden_create(*, board):
        created["count"] += 1
        raise AssertionError("root must not be created before authorization reservation")

    monkeypatch.setattr(kb, "_create_verifier_root", forbidden_create)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()
        conn.execute(
            "UPDATE tasks SET claim_lock = 'stale', current_run_id = NULL WHERE id = ?",
            (task.id,),
        )

    with pytest.raises(RuntimeError, match="authorization denied"):
        kb._default_spawn(task, str(kanban_home), board=None)

    assert created["count"] == 0


def test_default_spawn_verifier_windows_uses_explicit_handle_list(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(kb, "_start_verifier_supervisor_waiter", lambda **_: None, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    class FakeStartupInfo:
        def __init__(self):
            self.lpAttributeList = {}

    monkeypatch.setattr(subprocess, "STARTUPINFO", FakeStartupInfo, raising=False)
    fake_msvcrt = types.SimpleNamespace(get_osfhandle=lambda fd: 987654)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    handle_inheritable = []
    monkeypatch.setattr(
        os,
        "set_handle_inheritable",
        lambda handle, inheritable: handle_inheritable.append((handle, inheritable)),
        raising=False,
    )
    captured = {}

    class FakeProc:
        pid = 222

    def fake_popen(cmd, *args, **kwargs):
        captured.update(env=kwargs["env"], kwargs=kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()

    kb._default_spawn(task, str(kanban_home), board=None)

    startupinfo = captured["kwargs"]["startupinfo"]
    handle_list = startupinfo.lpAttributeList["handle_list"]
    assert handle_list == [987654]
    assert captured["env"]["HERMES_KANBAN_VERIFY_RESULT_HANDLE"] == "987654"
    assert handle_inheritable == [(987654, True)]
    assert captured["kwargs"]["close_fds"] is True
    assert "pass_fds" not in captured["kwargs"]


def test_default_spawn_verifier_preserves_only_configured_provider_credential(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    monkeypatch.setattr(kb, "_start_verifier_supervisor_waiter", lambda **_: None, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-live")
    captured = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, *args, **kwargs):
        captured.update(cmd=cmd, env=kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    (kanban_home / "config.yaml").write_text(
        "model:\n  provider: openai-api\n  name: gpt-5-mini\n",
        encoding="utf-8",
    )
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()

    kb._default_spawn(task, str(kanban_home), board=None)

    env = captured["env"]
    assert env["OPENAI_API_KEY"] == "sk-live"
    assert "ANTHROPIC_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env


def test_verifier_auth_store_is_provider_scoped_and_supports_pool_credentials(
    kanban_home,
    tmp_path,
):
    profile_home = tmp_path / "profile"
    verifier_root = tmp_path / "verifier"
    profile_home.mkdir()
    verifier_root.mkdir()
    (profile_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai-codex": {"tokens": {"access_token": "wanted"}},
                    "anthropic": {"tokens": {"access_token": "unrelated"}},
                },
                "credential_pool": {
                    "openai-codex": [{"token": "wanted-pool"}],
                    "anthropic": [{"token": "unrelated-pool"}],
                },
            }
        ),
        encoding="utf-8",
    )

    kb._copy_provider_auth_store(
        verifier_root=verifier_root,
        profile_home=profile_home,
        provider="openai-codex",
    )

    copied = json.loads((verifier_root / "auth.json").read_text(encoding="utf-8"))
    assert copied == {
        "version": 1,
        "providers": {"openai-codex": {"tokens": {"access_token": "wanted"}}},
        "credential_pool": {"openai-codex": [{"token": "wanted-pool"}]},
    }


def test_verifier_contract_redacts_readiness_evidence_from_bound_payload():
    contract = {
        "pr_url": PR_URL,
        "approved_head": HEAD_SHA,
        "approved_base": BASE_BRANCH,
        "readiness_evidence": {"ready": True, "private_review": "do-not-persist"},
    }

    canonical = kb._canonical_verifier_contract_payload(contract)

    assert canonical["readiness_evidence"]["ready"] is True
    assert len(canonical["readiness_evidence"]["digest"]) == 64
    assert "private_review" not in json.dumps(canonical)
    assert "do-not-persist" not in kb._verifier_prompt(canonical)


def test_verifier_result_tool_posix_e2e_emits_bound_frame(kanban_home, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX inherited-fd E2E")
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify merge", assignee="alice")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.verification_contract = _contract()
        binding = kb.make_verifier_result_binding(task, _contract(), nonce="nonce-posix")

    verifier_root = kb._create_verifier_root(board=None)
    read_fd, writer_fd = os.pipe()
    try:
        os.set_inheritable(read_fd, False)
        os.set_inheritable(writer_fd, True)
        env = kb._build_verifier_child_env(
            base_env=dict(os.environ),
            workspace=str(kanban_home),
            contract=_contract(),
            binding=binding,
            writer_fd=writer_fd,
            verifier_root=verifier_root,
            profile_arg="alice",
            task=task,
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from model_tools import handle_function_call; "
                    "print(handle_function_call("
                    "'kanban_verifier_result', "
                    "{'verdict':'approved','summary':'verified'}, "
                    "enabled_toolsets=['kanban_verifier']"
                    "))"
                ),
            ],
            cwd=str(verifier_root),
            env=env,
            pass_fds=(writer_fd,),
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        assert '"emitted": true' in proc.stdout
    finally:
        try:
            os.close(writer_fd)
        except OSError:
            pass
    raw = os.read(read_fd, kb.VERIFIER_RESULT_MAX_BYTES)
    os.close(read_fd)
    parsed = kb.parse_verifier_result_frame(raw)
    assert parsed.valid is True
    assert parsed.payload is not None
    assert parsed.payload["task_id"] == task.id
    assert parsed.payload["run_id"] == task.current_run_id
    assert parsed.payload["claim_lock"] == task.claim_lock
    assert parsed.payload["action"] == "complete"
    assert parsed.payload["contract_hash"] == kb.canonical_verifier_contract_hash(_contract())


def test_reconcile_reserved_pre_start_keeps_live_supervisor_owner(kanban_home):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        _claim, _binding, authorization_id = _claim_with_reserved_authorization(conn)
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            ("host:111", authorization_id),
        )
        restored = kb.reconcile_verifier_authorizations(
            conn,
            owner_id="host:111",
            live_owner_ids={"host:111"},
        )

        row = conn.execute(
            "SELECT state, reason_code, supervisor_owner FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()

    assert restored == []
    assert row["state"] == "reserved"
    assert row["reason_code"] == "reserve"
    assert row["supervisor_owner"] == "host:111"


def test_dispatch_restores_reserved_pre_start_from_dead_supervisor_once(
    kanban_home,
    monkeypatch,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == os.getpid())
    host = kb.socket.gethostname()
    dead_owner = f"{host}:999999"
    with kb.connect() as conn:
        claim, _binding, authorization_id = _claim_with_reserved_authorization(conn)
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            (dead_owner, authorization_id),
        )

        result = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        again = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)

        row = conn.execute(
            "SELECT state, reason_code, supervisor_owner FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
        events = [
            e for e in kb.list_events(conn, claim.id)
            if e.kind == "verifier_authorization_reconciled"
        ]
        task_status = kb.get_task(conn, claim.id).status

    assert result.skipped_locked is False
    assert again.skipped_locked is False
    assert row["state"] == "authorized"
    assert row["reason_code"] == "pre_start_restore"
    assert row["supervisor_owner"] is None
    assert task_status == "running"
    assert [e.payload for e in events] == [
        {
            "authorization_id": authorization_id,
            "from_state": "reserved",
            "to_state": "authorized",
            "reason": "pre_start_restore",
        }
    ]


def test_reconcile_reserved_pre_start_does_not_restore_stale_reclaimed_run(
    kanban_home,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="verify protected merge", assignee="alice")
        claim = kb.claim_task(conn, task_id)
        assert claim is not None
        binding = kb.make_verifier_result_binding(claim, _contract(), nonce="nonce-stage3")
        auth = kb.authorize_verifier_result(conn, **binding)
        assert auth.state == "authorized"
        assert kb.reserve_verifier_result(conn, authorization_id=auth.id, **binding).state == "reserved"
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            ("host:111", auth.id),
        )
        conn.execute(
            "UPDATE task_runs SET status = 'reclaimed', outcome = 'reclaimed' WHERE id = ?",
            (binding["run_id"],),
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, current_run_id = NULL WHERE id = ?",
            (claim.id,),
        )

        restored = kb.reconcile_verifier_authorizations(
            conn,
            owner_id="host:111",
            live_owner_ids=set(),
        )
        row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (auth.id,),
        ).fetchone()
        task = kb.get_task(conn, claim.id)

    assert restored == []
    assert row["state"] == "reserved"
    assert row["reason_code"] == "reserve"
    assert task.status == "ready"
    assert task.current_run_id is None


def test_reconcile_started_missing_owner_fail_closed_once(kanban_home):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        live_claim, live_binding, live_auth_id = _claim_with_authorization(conn)
        dead_claim, dead_binding, dead_auth_id = _claim_with_authorization(conn)
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            ("host:live", live_auth_id),
        )
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            ("host:dead", dead_auth_id),
        )

        reconciled = kb.reconcile_verifier_authorizations(
            conn,
            owner_id="host:scanner",
            live_owner_ids={"host:live"},
        )
        again = kb.reconcile_verifier_authorizations(
            conn,
            owner_id="host:scanner",
            live_owner_ids={"host:live"},
        )

        live_row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (live_auth_id,),
        ).fetchone()
        dead_row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (dead_auth_id,),
        ).fetchone()
        dead_events = [
            e for e in kb.list_events(conn, dead_claim.id)
            if e.kind == "verifier_authorization_reconciled"
        ]
        dead_task = kb.get_task(conn, dead_claim.id)
        live_task = kb.get_task(conn, live_claim.id)
        dead_summary = kb.latest_summary(conn, dead_claim.id) or ""

    assert reconciled == [dead_auth_id]
    assert again == []
    assert live_row["state"] == "started"
    assert live_row["reason_code"] == "start"
    assert live_task.status == "running"
    assert dead_row["state"] == "reconciled"
    assert dead_row["reason_code"] == "post_start_owner_missing"
    assert dead_task.status == "blocked"
    assert dead_task.block_kind == "needs_input"
    assert "verifier result channel failed: post_start_owner_missing" in dead_summary
    assert dead_events[-1].payload == {
        "authorization_id": dead_auth_id,
        "from_state": "started",
        "to_state": "reconciled",
        "reason": "post_start_owner_missing",
    }


def test_dispatch_runs_verifier_reconciliation_before_spawning(kanban_home, monkeypatch):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == os.getpid())
    host = kb.socket.gethostname()
    with kb.connect() as conn:
        live_claim, live_binding, live_auth_id = _claim_with_authorization(conn)
        dead_claim, dead_binding, dead_auth_id = _claim_with_authorization(conn)
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            (f"{host}:{os.getpid()}", live_auth_id),
        )
        conn.execute(
            "UPDATE verifier_result_authorizations SET supervisor_owner = ? WHERE id = ?",
            (f"{host}:999999", dead_auth_id),
        )

        result = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)

        live_row = conn.execute(
            "SELECT state FROM verifier_result_authorizations WHERE id = ?",
            (live_auth_id,),
        ).fetchone()
        dead_row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (dead_auth_id,),
        ).fetchone()
        live_task = kb.get_task(conn, live_claim.id)
        dead_task = kb.get_task(conn, dead_claim.id)

    assert result.skipped_locked is False
    assert live_row["state"] == "started"
    assert live_task.status == "running"
    assert dead_row["state"] == "reconciled"
    assert dead_row["reason_code"] == "post_start_owner_missing"
    assert dead_task.status == "blocked"


def test_verifier_result_failure_race_does_not_leave_consumed_running(
    kanban_home,
):
    assert Path(os.environ["HERMES_HOME"]) == kanban_home
    with kb.connect() as conn:
        claim, binding, authorization_id = _claim_with_authorization(conn)
    start = threading.Event()
    outcome = {}

    def apply_result():
        assert start.wait(timeout=2)
        with kb.connect() as conn:
            outcome["result"] = kb.apply_verifier_supervisor_result(
                conn,
                authorization_id=authorization_id,
                binding=binding,
                raw_frame=_frame(binding),
            )

    def fail_closed():
        assert start.wait(timeout=2)
        kb._reconcile_verifier_authorization_failure(
            db_path=kb.kanban_db_path(),
            authorization_id=authorization_id,
            binding=binding,
            reason="timeout",
        )
        kb._verifier_failure_reblock(
            db_path=kb.kanban_db_path(),
            task_id=claim.id,
            run_id=int(binding["run_id"]),
            reason="timeout",
        )

    threads = [threading.Thread(target=apply_result), threading.Thread(target=fail_closed)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    with kb.connect() as conn:
        row = conn.execute(
            "SELECT state, reason_code FROM verifier_result_authorizations WHERE id = ?",
            (authorization_id,),
        ).fetchone()
        task = kb.get_task(conn, claim.id)

    assert (row["state"], task.status) in {
        ("applied", "done"),
        ("reconciled", "blocked"),
    }
