"""Behavior contract for profile-local CLI one-shot audit provenance."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


def _records(home):
    path = home / "logs" / "oneshot-audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audit_append_is_concurrent_safe_and_excludes_prompt_and_argv(tmp_path, monkeypatch):
    from hermes_cli import oneshot_audit

    probe_home = tmp_path / "probe"
    monkeypatch.setenv("HERMES_HOME", str(probe_home))
    prompt_marker = "raw-prompt-must-not-appear"
    argv_marker = "argv-marker-must-not-appear"
    monkeypatch.setattr(sys, "argv", ["hermes", "-q", argv_marker])

    open_flags = []
    writes = []
    real_open = oneshot_audit.os.open
    real_write = oneshot_audit.os.write

    def traced_open(path, flags, mode):
        open_flags.append(flags)
        return real_open(path, flags, mode)

    def traced_write(fd, payload):
        writes.append(payload)
        return real_write(fd, payload)

    with monkeypatch.context() as patch:
        patch.setattr(oneshot_audit.os, "open", traced_open)
        patch.setattr(oneshot_audit.os, "write", traced_write)
        audit = oneshot_audit.start_oneshot_audit("probe", "programmatic")
        audit.finish("success", 0)

    assert len(writes) == 2
    assert all(flags & oneshot_audit.os.O_APPEND for flags in open_flags)
    assert all(payload.endswith(b"\n") and payload.count(b"\n") == 1 for payload in writes)

    concurrent_home = tmp_path / "concurrent"
    monkeypatch.setenv("HERMES_HOME", str(concurrent_home))

    def write_pair(index: int) -> None:
        audit = oneshot_audit.start_oneshot_audit(f"{prompt_marker}-{index}", "programmatic")
        audit.finish("success", 0, session_id=f"session-{index}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write_pair, range(64)))

    path = concurrent_home / "logs" / "oneshot-audit.jsonl"
    raw = path.read_text(encoding="utf-8")
    records = _records(concurrent_home)
    assert len(records) == 128
    assert prompt_marker not in raw
    assert argv_marker not in raw
    assert "argv" not in raw.lower()

    by_id = {}
    for record in records:
        by_id.setdefault(record["audit_id"], []).append(record)
        assert record["prompt_chars"] > 0
        assert len(record["prompt_sha256"]) == 64
        assert record["input_mode"] == "programmatic"
        assert record["pid"] > 0
        assert record["ppid"] >= 0
        assert "uid" in record
        assert "cwd" in record
        assert "tty" in record
        if record["parent_executable"] is not None:
            assert "/" not in record["parent_executable"]
            assert "\\" not in record["parent_executable"]
    assert len(by_id) == 64
    assert all([entry["event"] for entry in pair] == ["started", "finished"] for pair in by_id.values())


def test_both_cli_oneshot_seams_record_correlated_lifecycle_outcomes(tmp_path, monkeypatch):
    import cli as cli_mod
    from hermes_cli import oneshot, oneshot_audit

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(oneshot, "_validate_explicit_toolsets", lambda _value: (None, None))

    cases = [
        ("success", lambda *_a, **_k: ("ok", {"session_id": "session-ok"}), 0, "success", "session-ok"),
        ("agent-error", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")), 1, "agent_error", None),
        ("interrupted", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()), 130, "interrupted", None),
    ]
    for label, run_agent, exit_code, outcome, session_id in cases:
        monkeypatch.setattr(oneshot, "_run_agent", run_agent)
        before = len(_records(tmp_path)) if (tmp_path / "logs" / "oneshot-audit.jsonl").exists() else 0
        if outcome == "interrupted":
            with pytest.raises(KeyboardInterrupt):
                oneshot.run_oneshot(f"private-{label}", input_mode="prompt")
        else:
            assert oneshot.run_oneshot(f"private-{label}", input_mode="prompt") == exit_code
        pair = _records(tmp_path)[before:]
        assert [entry["event"] for entry in pair] == ["started", "finished"]
        assert pair[0]["audit_id"] == pair[1]["audit_id"]
        assert pair[1]["outcome"] == outcome
        assert pair[1]["exit_code"] == exit_code
        assert pair[1]["session_id"] == session_id
        assert all(entry["input_mode"] == "prompt" for entry in pair)

    before = len(_records(tmp_path))
    assert oneshot.run_oneshot("private-validation", provider="custom", input_mode="prompt") == 2
    validation_pair = _records(tmp_path)[before:]
    assert validation_pair[1]["outcome"] == "validation_error"
    assert validation_pair[1]["exit_code"] == 2

    cli_prompt = "private-query-file"

    def build_cli(*_args, **_kwargs):
        # The start record must exist before any model/agent setup seam is entered.
        assert _records(tmp_path)[-1]["event"] == "started"
        return SimpleNamespace(session_id="cli-session", agent=None)

    monkeypatch.setattr(cli_mod, "_build_cli_from_args", build_cli)
    monkeypatch.setattr(cli_mod, "_install_single_query_signal_handlers", lambda _cli: None)
    monkeypatch.setattr(cli_mod, "_start_worktree_setup", lambda *_args: None)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args: None)

    cli_cases = [
        ("query", None, "success", 0),
        ("query-file", SystemExit(2), "validation_error", 2),
        ("query-file", SystemExit(1), "agent_error", 1),
        ("query-file", KeyboardInterrupt(), "interrupted", 130),
    ]
    for input_mode, raised, outcome, exit_code in cli_cases:
        def run_single_query(fake_cli, *_args, raised=raised):
            fake_cli.agent = SimpleNamespace(session_id="cli-agent-session")
            if raised is not None:
                raise raised

        monkeypatch.setattr(cli_mod, "_run_single_query_mode", run_single_query)
        before = len(_records(tmp_path))
        if raised is None:
            cli_mod.main(query=cli_prompt, quiet=True, _input_mode=input_mode)
        else:
            with pytest.raises(type(raised)) as exc:
                cli_mod.main(query=cli_prompt, quiet=True, _input_mode=input_mode)
            if isinstance(raised, SystemExit):
                assert exc.value.code == exit_code
        pair = _records(tmp_path)[before:]
        assert [entry["event"] for entry in pair] == ["started", "finished"]
        assert pair[0]["audit_id"] == pair[1]["audit_id"]
        assert pair[1]["outcome"] == outcome
        assert pair[1]["exit_code"] == exit_code
        assert pair[1]["session_id"] == "cli-agent-session"
        assert all(entry["input_mode"] == input_mode for entry in pair)

    monkeypatch.setattr(oneshot_audit, "start_oneshot_audit", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(oneshot, "_run_agent", lambda *_a, **_k: ("still works", {}))
    assert oneshot.run_oneshot("fail-open") == 0
