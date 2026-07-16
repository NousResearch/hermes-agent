"""Stage 2 verifier-supervisor tests for protected merge recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _frame(binding: dict, **overrides) -> bytes:
    payload = {
        "version": 1,
        "task_id": binding["task_id"],
        "run_id": binding["run_id"],
        "claim_lock": binding["claim_lock"],
        "contract_id": binding["contract_id"],
        "pr_url": binding["pr_url"],
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
        ({"pr_url": "https://github.com/NousResearch/hermes-agent/pull/1"}, "binding_mismatch:pr_url"),
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
        "pr_url": PR_URL,
        "nonce": "nonce",
        "action": "complete",
        field: "x" * (kb._CTX_MAX_FIELD_BYTES + 1),
    }
    result = kb.parse_verifier_result_frame(
        (json.dumps(payload) + "\n").encode("utf-8")
    )
    assert result.valid is False
    assert result.reason == f"field_too_large:{field}"


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
    assert env["HERMES_KANBAN_VERIFY_RESULT_FD"] == str(captured["writer_fd"])
    assert env["HERMES_IGNORE_USER_CONFIG"] == "1"
    assert "HERMES_HOME" not in env
    assert "HERMES_PROFILE" not in env
    assert "HERMES_ACCEPT_HOOKS" not in env
    assert "HERMES_KANBAN_DB" not in env
    assert "HERMES_KANBAN_BOARD" not in env
    assert "HERMES_KANBAN_WORKSPACES_ROOT" not in env
    assert "HERMES_KANBAN_TASK" not in env
    assert "--toolsets" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--toolsets") + 1] == "terminal,file"
    assert "-p" not in captured["cmd"]
    assert "--profile" not in captured["cmd"]
    assert "--accept-hooks" not in captured["cmd"]
    assert captured["kwargs"]["close_fds"] is True


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

    def fake_open(*args, **kwargs):
        opened["log"] = FakeLog()
        return opened["log"]

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

    assert opened["log"].closed is True


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
