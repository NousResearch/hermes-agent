"""Tests for bounded cross-profile execution provenance."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hermes_cli.execution_provenance as provenance
from hermes_cli.execution_provenance import (
    ExecutionAuthorityError,
    _ledger_path,
    _redacted_execution_path,
    authorize_profile_execution,
    format_execution_status,
    is_agent_invocation,
    is_read_only_invocation,
    list_execution_status,
)


def _authority(**overrides):
    data = {
        "authority_class": "direct_one_shot",
        "authority_reference": "DEC-TEST-001",
        "source": "csm",
        "target": "s6",
        "scope": "one bounded test",
        "one_shot": True,
        "expires_at": time.time() + 300,
        "execution_id": "exec-test-001",
        "evidence": "kanban:t_test",
        "terminal_condition": "process exits",
    }
    data.update(overrides)
    return data


def _kanban_db(path: Path, *, assignee="s6", run_id=9, claim="claim-1", expiry=None):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE tasks (id TEXT PRIMARY KEY, assignee TEXT, status TEXT,
            claim_lock TEXT, claim_expires INTEGER, current_run_id INTEGER,
            worker_pid INTEGER);
        CREATE TABLE task_runs (id INTEGER PRIMARY KEY, task_id TEXT,
            profile TEXT, status TEXT, claim_lock TEXT, claim_expires INTEGER,
            worker_pid INTEGER);
    """)
    expiry = int(time.time()) + 300 if expiry is None else expiry
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, 'running', ?, ?, ?, NULL)",
        ("t_1", assignee, claim, expiry, run_id),
    )
    conn.execute(
        "INSERT INTO task_runs VALUES (?, 't_1', ?, 'running', ?, ?, NULL)",
        (run_id, assignee, claim, expiry),
    )
    conn.commit()
    conn.close()
    return path


def test_exact_direct_exception_is_recorded(tmp_path):
    ledger = tmp_path / "executions.jsonl"
    record = authorize_profile_execution(
        source="csm",
        target="s6",
        argv=["hermes", "-p", "s6", "chat", "-q", "work"],
        authority_json=json.dumps(_authority()),
        ledger_path=ledger,
        pid=os.getpid(),
    )
    assert record["authority_class"] == "direct_one_shot"
    assert record["authority_reference"] == "DEC-TEST-001"
    assert record["source"] == "csm"
    assert record["target"] == "s6"
    assert record["execution_path"] == "hermes -p s6 chat -q '[REDACTED]'"
    assert record["kanban_tracked"] is False
    assert record["state"] == "running"
    assert (
        list_execution_status(ledger_path=ledger)[0]["execution_id"] == "exec-test-001"
    )


def test_cli_subprocess_fails_closed_for_admin_token_prompt(tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    env = dict(os.environ)
    env["HERMES_HOME"] = str(root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for key in (
        "HERMES_PROFILE",
        "HERMES_DISPATCH_SOURCE_PROFILE",
        "HERMES_EXECUTION_AUTHORITY",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
    ):
        env.pop(key, None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "-p",
            "coder",
            "chat",
            "-q",
            "gateway",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 77
    assert "requires structured authority" in completed.stderr


def test_independent_cross_profile_chat_is_rejected(tmp_path):
    with pytest.raises(ExecutionAuthorityError, match="structured authority"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["python", "-m", "hermes_cli.main", "chat"],
            authority_json=None,
            ledger_path=tmp_path / "executions.jsonl",
        )


def test_mismatched_expired_and_replayed_authority_are_rejected(tmp_path):
    ledger = tmp_path / "executions.jsonl"
    with pytest.raises(ExecutionAuthorityError, match="target mismatch"):
        authorize_profile_execution(
            source="csm",
            target="s7",
            argv=["hermes", "chat"],
            authority_json=json.dumps(_authority()),
            ledger_path=ledger,
        )
    with pytest.raises(ExecutionAuthorityError, match="expired"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=json.dumps(_authority(expires_at=time.time() - 1)),
            ledger_path=ledger,
        )
    payload = json.dumps(_authority())
    authorize_profile_execution(
        source="csm",
        target="s6",
        argv=["hermes", "chat"],
        authority_json=payload,
        ledger_path=ledger,
    )
    with pytest.raises(ExecutionAuthorityError, match="already used"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=payload,
            ledger_path=ledger,
        )


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), float("-inf")])
def test_direct_authority_rejects_non_finite_expiry(tmp_path, expires_at):
    with pytest.raises(ExecutionAuthorityError, match="finite"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=json.dumps(_authority(expires_at=expires_at)),
            ledger_path=tmp_path / "executions.jsonl",
        )


def test_short_writes_are_completed_and_replay_stays_rejected(tmp_path, monkeypatch):
    ledger = tmp_path / "executions.jsonl"
    real_write = provenance.os.write

    def short_write(fd, data):
        return real_write(fd, data[: max(1, min(7, len(data)))])

    monkeypatch.setattr(provenance.os, "write", short_write)
    payload = json.dumps(_authority())
    assert (
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=payload,
            ledger_path=ledger,
        )
        is not None
    )
    assert (
        json.loads(ledger.read_text(encoding="utf-8"))["execution_id"]
        == "exec-test-001"
    )
    with pytest.raises(ExecutionAuthorityError, match="already used"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=payload,
            ledger_path=ledger,
        )


def test_failed_ledger_append_still_consumes_one_shot_id(tmp_path, monkeypatch):
    ledger = tmp_path / "executions.jsonl"
    real_write_all = provenance._write_all

    def fail_ledger_payload(fd, data):
        if data.startswith(b"{"):
            raise OSError("forced ledger append failure")
        return real_write_all(fd, data)

    monkeypatch.setattr(provenance, "_write_all", fail_ledger_payload)
    payload = json.dumps(_authority())
    with pytest.raises(ExecutionAuthorityError, match="persistence failed"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=payload,
            ledger_path=ledger,
        )
    assert ledger.read_bytes() == b""
    assert len(list(ledger.with_name(ledger.name + ".consumed").iterdir())) == 1
    with pytest.raises(ExecutionAuthorityError, match="already used"):
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=payload,
            ledger_path=ledger,
        )


def test_consumption_marker_directory_entries_are_fsynced(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX directory fsync assertion")
    ledger = tmp_path / "executions.jsonl"
    real_fsync = provenance.os.fsync
    fsync_modes: list[int] = []

    def recording_fsync(fd):
        fsync_modes.append(provenance.os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(provenance.os, "fsync", recording_fsync)
    assert (
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "chat"],
            authority_json=json.dumps(_authority()),
            ledger_path=ledger,
        )
        is not None
    )

    assert sum(stat.S_ISDIR(mode) for mode in fsync_modes) >= 2


def test_kanban_dispatch_is_allowed_and_visible(tmp_path):
    ledger = tmp_path / "executions.jsonl"
    record = authorize_profile_execution(
        source="dispatcher",
        target="s6",
        argv=["hermes", "-p", "s6", "chat", "-q", "work kanban task t_1"],
        authority_json=None,
        ledger_path=ledger,
        kanban_task="t_1",
        kanban_run="9",
        kanban_claim_lock="claim-1",
        kanban_db_path=_kanban_db(tmp_path / "kanban.db"),
    )
    assert record is not None
    assert record["authority_class"] == "kanban_dispatch"
    assert record["authority_reference"] == "kanban:t_1:run:9"
    assert record["kanban_tracked"] is True


@pytest.mark.parametrize(
    "target,run,claim",
    [
        ("s7", "9", "claim-1"),
        ("s6", "8", "claim-1"),
        ("s6", "9", "wrong"),
    ],
)
def test_forged_or_inherited_kanban_custody_is_rejected(tmp_path, target, run, claim):
    with pytest.raises(ExecutionAuthorityError):
        authorize_profile_execution(
            source="untrusted-label",
            target=target,
            argv=["hermes", "-p", target, "chat", "-q", "work"],
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
            kanban_task="t_1",
            kanban_run=run,
            kanban_claim_lock=claim,
            kanban_db_path=_kanban_db(tmp_path / "kanban.db"),
        )


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_kanban_claim_expiry_fails_closed(tmp_path, expiry):
    with pytest.raises(ExecutionAuthorityError, match="custody"):
        authorize_profile_execution(
            source="dispatcher",
            target="s6",
            argv=["hermes", "-p", "s6", "chat", "-q", "work"],
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
            kanban_task="t_1",
            kanban_run="9",
            kanban_claim_lock="claim-1",
            kanban_db_path=_kanban_db(tmp_path / "kanban.db", expiry=expiry),
        )


def test_live_kanban_custody_not_source_label_is_authoritative(tmp_path):
    record = authorize_profile_execution(
        source="untrusted-label",
        target="s6",
        argv=["hermes", "-p", "s6", "chat", "-q", "work"],
        authority_json=None,
        ledger_path=tmp_path / "ledger.jsonl",
        kanban_task="t_1",
        kanban_run="9",
        kanban_claim_lock="claim-1",
        kanban_db_path=_kanban_db(tmp_path / "kanban.db"),
    )
    assert record is not None
    assert record["authority_class"] == "kanban_dispatch"
    assert record["source"] == "untrusted-label"


def test_kanban_run_execution_id_cannot_be_replayed(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    db_path = _kanban_db(tmp_path / "kanban.db")
    kwargs = {
        "source": "dispatcher",
        "target": "s6",
        "argv": ["hermes", "-p", "s6", "chat", "-q", "work"],
        "authority_json": None,
        "ledger_path": ledger,
        "kanban_task": "t_1",
        "kanban_run": "9",
        "kanban_claim_lock": "claim-1",
        "kanban_db_path": db_path,
    }
    assert authorize_profile_execution(**kwargs) is not None
    with pytest.raises(ExecutionAuthorityError, match="already used"):
        authorize_profile_execution(**kwargs)


def test_concurrent_kanban_prebinding_race_accepts_one_execution(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    db_path = _kanban_db(tmp_path / "kanban.db")
    barrier = threading.Barrier(16)
    accepted: list[bool] = []

    def attempt():
        barrier.wait()
        try:
            authorize_profile_execution(
                source="dispatcher",
                target="s6",
                argv=["hermes", "-p", "s6", "chat", "-q", "work"],
                authority_json=None,
                ledger_path=ledger,
                kanban_task="t_1",
                kanban_run="9",
                kanban_claim_lock="claim-1",
                kanban_db_path=db_path,
            )
            accepted.append(True)
        except ExecutionAuthorityError:
            pass

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["execution_id"] for row in rows] == ["kanban-t_1-run-9"]


def test_missing_kanban_database_fails_closed(tmp_path):
    with pytest.raises(ExecutionAuthorityError, match="Kanban custody"):
        authorize_profile_execution(
            source="dispatcher",
            target="s6",
            argv=["hermes", "-p", "s6", "chat", "-q", "work"],
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
            kanban_task="t_1",
            kanban_run="9",
            kanban_claim_lock="claim-1",
            kanban_db_path=tmp_path / "missing.db",
        )


def test_noninteractive_rejects_but_human_tty_selection_is_intentional(tmp_path):
    with pytest.raises(ExecutionAuthorityError, match="structured authority"):
        authorize_profile_execution(
            source="external",
            target="s6",
            argv=["hermes", "-p", "s6", "chat", "-q", "secret prompt"],
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
        )
    with pytest.raises(ExecutionAuthorityError, match="structured authority"):
        authorize_profile_execution(
            source="default",
            target="s6",
            argv=["hermes", "-p", "s6"],
            authority_json=None,
            interactive=False,
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert (
        authorize_profile_execution(
            source="default",
            target="s6",
            argv=["hermes", "-p", "s6"],
            authority_json=None,
            interactive=True,
            ledger_path=tmp_path / "ledger.jsonl",
        )
        is None
    )


_ADMIN_PROMPT_WORDS = [
    "config",
    "cron",
    "doctor",
    "gateway",
    "kanban",
    "mcp",
    "message",
    "profile",
    "session",
    "skills",
    "tools",
    "version",
]


@pytest.mark.parametrize("prompt", _ADMIN_PROMPT_WORDS)
@pytest.mark.parametrize(
    "argv_factory",
    [
        lambda prompt: ["hermes", "-p", "s6", "-q", prompt],
        lambda prompt: ["hermes", f"--query={prompt}", "-p", "s6"],
        lambda prompt: ["hermes", "-p", "s6", "-z", prompt],
        lambda prompt: ["hermes", f"--oneshot={prompt}", "-p", "s6"],
    ],
)
def test_prompt_words_cannot_masquerade_as_admin_commands(
    tmp_path, prompt, argv_factory
):
    argv = argv_factory(prompt)
    assert is_agent_invocation(argv)
    assert not is_read_only_invocation(argv)
    with pytest.raises(ExecutionAuthorityError, match="structured authority"):
        authorize_profile_execution(
            source="external",
            target="s6",
            argv=argv,
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["hermes", "-p", "s6", "-z", "ONESHOT-SECRET"],
        ["hermes", "-p", "s6", "--oneshot", "ONESHOT-SECRET"],
        ["hermes", "--oneshot=ONESHOT-SECRET", "-p", "s6"],
    ],
)
def test_oneshot_is_enforced_and_redacted(tmp_path, argv):
    sentinel = "ONESHOT-SECRET"
    assert is_agent_invocation(argv)
    assert not is_read_only_invocation(argv)
    with pytest.raises(ExecutionAuthorityError, match="structured authority"):
        authorize_profile_execution(
            source="external",
            target="s6",
            argv=argv,
            authority_json=None,
            ledger_path=tmp_path / "ledger.jsonl",
        )
    execution_path = _redacted_execution_path(argv)
    assert sentinel not in execution_path
    assert "[REDACTED]" in execution_path


def test_direct_execution_id_is_consumed_atomically(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    payload = json.dumps(_authority())
    barrier = threading.Barrier(16)
    accepted = []

    def attempt():
        barrier.wait()
        try:
            authorize_profile_execution(
                source="csm",
                target="s6",
                argv=["hermes", "chat"],
                authority_json=payload,
                ledger_path=ledger,
            )
            accepted.append(True)
        except ExecutionAuthorityError:
            pass

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(accepted) == 1
    assert (
        sum(
            json.loads(line)["execution_id"] == "exec-test-001"
            for line in ledger.read_text().splitlines()
        )
        == 1
    )


def test_prompt_payload_is_redacted_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    record = authorize_profile_execution(
        source="csm",
        target="s6",
        argv=["hermes", "-p", "s6", "chat", "-q", "TOP SECRET BODY"],
        authority_json=json.dumps(_authority()),
        ledger_path=ledger,
    )
    assert "TOP SECRET BODY" not in record["execution_path"]
    assert "[REDACTED]" in record["execution_path"]
    assert "TOP SECRET BODY" not in ledger.read_text()


@pytest.mark.parametrize(
    ("argv", "secret"),
    [
        (["hermes", "chat", "--api-key", "sk-live"], "sk-live"),
        (["hermes", "chat", "--access_token=opaque-token"], "opaque-token"),
        (["hermes", "chat", "--client-secret", "oauth-secret"], "oauth-secret"),
        (["hermes", "chat", "--password=hunter2"], "hunter2"),
        (["hermes", "chat", "--authorization", "Bearer abc123"], "Bearer abc123"),
        (["hermes", "chat", "--cookie=session=private"], "session=private"),
        (["hermes", "chat", "--signing_key", "private-material"], "private-material"),
        (["hermes", "chat", "--webhook-token=hook-secret"], "hook-secret"),
    ],
)
def test_sensitive_named_arguments_are_redacted_generically(argv, secret):
    execution_path = _redacted_execution_path(argv)
    assert secret not in execution_path
    assert "[REDACTED]" in execution_path


@pytest.mark.parametrize(
    "option",
    ["--body", "--message", "--content", "--authority", "--system-prompt"],
)
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_explicit_sensitive_options_are_redacted_in_both_forms(option, form):
    secret = f"SENTINEL-{option[2:].upper()}-SECRET"
    argv = (
        ["hermes", "chat", option, secret]
        if form == "separate"
        else ["hermes", "chat", f"{option}={secret}"]
    )

    execution_path = _redacted_execution_path(argv)

    assert secret not in execution_path
    assert "[REDACTED]" in execution_path


def test_non_sensitive_lookalike_arguments_remain_visible():
    argv = [
        "hermes",
        "chat",
        "--token-count",
        "4096",
        "--password-policy",
        "strict",
        "--secret-santa",
        "enabled",
        "--cookie-file-count",
        "2",
        "--body-format",
        "markdown",
        "--message-count",
        "3",
        "--content-type",
        "application/json",
        "--authority-level",
        "staff",
        "--system-prompt-mode",
        "inherit",
    ]
    execution_path = _redacted_execution_path(argv)
    assert "[REDACTED]" not in execution_path
    assert execution_path == shlex.join(argv)


def test_default_ledger_is_shared_across_profile_home_overlay(monkeypatch, tmp_path):
    root = tmp_path / "custom-hermes-root"
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "s6"))

    assert _ledger_path() == root / "execution-provenance.jsonl"


def test_module_imports_when_fcntl_is_unavailable():
    module_file = sys.modules[authorize_profile_execution.__module__].__file__
    assert module_file is not None
    module_path = Path(module_file)
    script = f"""
import builtins
import runpy
real_import = builtins.__import__
def portable_import(name, globals=None, *args, **kwargs):
    if (
        name in {{'fcntl', 'msvcrt'}}
        and globals
        and globals.get('__file__') == {str(module_path)!r}
    ):
        raise ImportError(name)
    return real_import(name, globals, *args, **kwargs)
builtins.__import__ = portable_import
namespace = runpy.run_path({str(module_path)!r})
assert namespace['fcntl'] is None
assert namespace['msvcrt'] is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_status_output_is_bounded_single_line_and_field_complete(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    oversized = "X" * 500 + "\nINJECTED"
    rows = []
    for index in range(10):
        rows.append({
            **_authority(
                execution_id=f"exec-{index}-{oversized}",
                authority_reference=oversized,
                scope=oversized,
                evidence=oversized,
                terminal_condition=oversized,
            ),
            "execution_path": oversized,
            "kanban_tracked": False,
            "pid": 99999999,
            "started_at": time.time() - index,
            "state": oversized,
        })
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    lines = format_execution_status(ledger_path=ledger)

    assert lines
    assert len("\n".join(lines)) <= 1200
    assert all("\n" not in line and len(line) <= 600 for line in lines)
    for field in (
        "source=",
        "authority=",
        "target=",
        "path=",
        "kanban=",
        "scope=",
        "one_shot=",
        "expires=",
        "evidence=",
        "terminal=",
        "state=",
    ):
        assert field in lines[0]
    assert "INJECTED" not in lines[0]


def test_status_reader_contains_malformed_rows_without_crashing(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "not-json\n"
        + json.dumps({
            "execution_id": "malformed",
            "pid": {"not": "a pid"},
            "started_at": "not-a-number",
            "state": "running",
        })
        + "\n",
        encoding="utf-8",
    )

    rows = list_execution_status(ledger_path=ledger)
    lines = format_execution_status(ledger_path=ledger)

    assert len(rows) == 1
    assert rows[0]["state"] == "terminal"
    assert lines and "Execution malformed" in lines[0]


def test_pid_liveness_uses_cross_platform_psutil(monkeypatch):
    observed: list[int] = []

    def pid_exists(pid):
        observed.append(pid)
        return pid == 42

    monkeypatch.setattr(provenance.psutil, "pid_exists", pid_exists)
    monkeypatch.setattr(
        provenance.os,
        "kill",
        lambda *_args: pytest.fail("os.kill must not be used for liveness"),
    )

    assert provenance._pid_alive(42)
    assert not provenance._pid_alive(43)
    assert not provenance._pid_alive(0)
    assert observed == [42, 43]


def test_status_reader_only_materializes_bounded_tail(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "".join(
            json.dumps({
                "execution_id": f"exec-{index}",
                "pid": 99999999,
                "started_at": index,
                "state": "running",
            })
            + "\n"
            for index in range(200)
        ),
        encoding="utf-8",
    )

    rows = list_execution_status(ledger_path=ledger)

    assert len(rows) == 50
    assert {row["execution_id"] for row in rows} == {
        f"exec-{index}" for index in range(150, 200)
    }


def test_status_reader_skips_oversized_tail_without_materializing_it(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"execution_id": "oversized", "value": "X" * 300_000}) + "\n",
        encoding="utf-8",
    )

    assert list_execution_status(ledger_path=ledger) == []
    assert format_execution_status(ledger_path=ledger) == []


def test_same_profile_and_read_only_inspection_need_no_authority(tmp_path):
    ledger = tmp_path / "executions.jsonl"
    assert (
        authorize_profile_execution(
            source="csm",
            target="csm",
            argv=["hermes", "chat"],
            authority_json=None,
            ledger_path=ledger,
        )
        is None
    )
    assert (
        authorize_profile_execution(
            source="csm",
            target="s6",
            argv=["hermes", "-p", "s6", "gateway", "status"],
            authority_json=None,
            ledger_path=ledger,
        )
        is None
    )
    assert not ledger.exists()


def test_untracked_process_visibility_and_terminal_state(tmp_path):
    ledger = tmp_path / "executions.jsonl"
    ledger.write_text(
        json.dumps({
            **_authority(execution_id="exec-dead"),
            "execution_path": "HERMES_HOME=/tmp/s6 python -m hermes_cli.main chat",
            "kanban_tracked": False,
            "pid": 99999999,
            "started_at": time.time() - 10,
            "state": "running",
        })
        + "\n",
        encoding="utf-8",
    )
    row = list_execution_status(ledger_path=ledger)[0]
    assert row["kanban_tracked"] is False
    assert row["state"] == "terminal"
