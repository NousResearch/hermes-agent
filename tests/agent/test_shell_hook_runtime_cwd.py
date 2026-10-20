"""Shell hooks must describe the active local session, not the multiplexer launch dir."""
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import sys
from threading import Barrier

import pytest

from agent import runtime_cwd, shell_hooks
from tools.terminal_scope import (
    TerminalPolicyRefusal,
    TerminalPolicyUnavailable,
    reset_terminal_scope,
    set_terminal_scope,
)


@pytest.mark.parametrize("event", ["pre_tool_call", "post_tool_call"])
@pytest.mark.parametrize("session_override", [False, True])
def test_concurrent_hook_callbacks_keep_session_cwd(tmp_path, monkeypatch, event, session_override):
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv("TERMINAL_CWD", str(launch))
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        "(pathlib.Path(sys.argv[1]) / (payload['session_id'] + '.json')).write_text(json.dumps(payload))\n"
    )
    spec = shell_hooks.ShellHookSpec(
        event=event,
        command=shlex.join([sys.executable, str(capture), str(tmp_path)]),
    )
    barrier = Barrier(2)

    def fire(name):
        cwd = tmp_path / name
        cwd.mkdir()
        terminal_token = set_terminal_scope({"TERMINAL_CWD": str(launch if session_override else cwd)})
        session_token = runtime_cwd.set_session_cwd(str(cwd) if session_override else None)
        try:
            barrier.wait(timeout=10)
            shell_hooks._make_callback(spec)(tool_name="terminal", args={"command": "fixture"}, session_id=name)
            payload = json.loads((tmp_path / (name + ".json")).read_text())
            assert payload["cwd"] == str(cwd)
            assert payload["session_id"] == name
            assert payload["hook_event_name"] == event
        finally:
            runtime_cwd.reset_session_cwd(session_token)
            reset_terminal_scope(terminal_token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fire, ["session-a", "session-b"]))


@pytest.mark.parametrize("mode", ["unset", "missing", "refused", "deleted"])
def test_hook_cwd_fallback_and_failure_contract(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    scope_token = set_terminal_scope(TerminalPolicyRefusal("fixture refusal") if mode == "refused" else {})
    session_token = runtime_cwd.set_session_cwd(str(tmp_path / "missing") if mode == "missing" else None)
    try:
        if mode == "deleted":
            def deleted_cwd():
                raise FileNotFoundError("deleted launch directory")
            monkeypatch.setattr(shell_hooks.Path, "cwd", deleted_cwd)
        if mode == "refused":
            with pytest.raises(TerminalPolicyUnavailable, match="fixture refusal"):
                shell_hooks._serialize_payload("pre_tool_call", {"session_id": "same-session"})
        else:
            payload = json.loads(shell_hooks._serialize_payload("pre_tool_call", {"session_id": "same-session"}))
            assert payload["cwd"] == ("" if mode == "deleted" else str(tmp_path))
            assert payload["session_id"] == "same-session"
    finally:
        runtime_cwd.reset_session_cwd(session_token)
        reset_terminal_scope(scope_token)
