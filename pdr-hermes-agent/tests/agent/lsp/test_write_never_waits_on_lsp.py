"""Invariant: a write never waits on the language server, and its diagnostics still reach the model
(the session that made the edit, and only that session).

The write path used to block on a pre-write baseline wait plus a post-write wait, each up to
``lsp.wait_timeout``; a server that stays silent (typescript-language-server on a clean edit) ran
both to the deadline on every edit.  The servers here are held until the test releases them, so the
assertions are about order (the tool result comes back before any verdict exists), not speed.
"""
from __future__ import annotations

import json
import time

import pytest

from agent.lsp import deferred
from agent.lsp.manager import LSPService
from model_tools import handle_function_call
from tests.agent.lsp.test_service import _install_mock_server
from tests.waiting import wait_until

_WAIT = 20.0  # lsp.wait_timeout: a write that waited on the server would take 2x this per edit


@pytest.fixture
def mock_workspace(monkeypatch, tmp_path):
    created = []

    def install(script, **env):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.chdir(str(repo))
        gate = tmp_path / "release-diagnostics"
        monkeypatch.setenv("MOCK_LSP_GATE", str(gate))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        gen = _install_mock_server(monkeypatch, script, "pyright")
        next(gen)
        svc = LSPService(enabled=True, wait_mode="document", wait_timeout=_WAIT, install_strategy="manual",
                         trusted_workspaces=[str(repo)])
        monkeypatch.setattr("agent.lsp.get_service", lambda: svc)
        created.append((svc, gen, gate))
        return repo, gate
    yield install
    for svc, gen, gate in created:
        gate.touch()  # release any held push so shutdown is not stuck behind it
        svc.shutdown()
        try:
            next(gen)
        except StopIteration:
            pass


def _patch(path, old, new, session="S-lsp"):
    handle_function_call("read_file", {"path": str(path)}, task_id="t-" + session, session_id=session)
    started = time.monotonic()
    out = handle_function_call("patch", {"path": str(path), "old_string": old, "new_string": new},
                               task_id="t-" + session, session_id=session)
    return json.loads(out), time.monotonic() - started


def _settled(*sessions):
    with deferred._LOCK:
        return not any(deferred._PENDING.get(s) for s in sessions)


def test_silent_server_does_not_hold_the_write(mock_workspace):
    """A server that never answers costs the write nothing: each edit returns long before one wait
    budget (the old path waited the budget twice per edit)."""
    repo, _gate = mock_workspace("silent")
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    for old, new in (("x = 1", "x = 2"), ("x = 2", "x = 3")):
        result, elapsed = _patch(target, old, new)
        assert result["success"] is True, result
        assert elapsed < _WAIT / 2, f"write blocked {elapsed:.2f}s on a silent language server"
    assert target.read_text(encoding="utf-8-sig") == "x = 3\n"


def test_late_diagnostics_reach_the_session(mock_workspace):
    """A verdict that arrives after the write returned is delivered to the editing session afterwards
    (attached to its next tool result), and only the error the edit introduced is reported."""
    repo, gate = mock_workspace("by_content")
    target = repo / "b.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result, elapsed = _patch(target, "x = 1", "x = bad")
    assert result["success"] is True and "lsp_diagnostics" not in result, result
    assert elapsed < _WAIT / 2, f"write blocked {elapsed:.2f}s on a held language server"
    gate.touch()
    (path, text), = wait_until(lambda: deferred.pop_notes("S-lsp"), timeout=_WAIT,
                               what="the late verdict reaching the session")
    assert path == str(target) and "MOCK001" in text and "earlier edit" in text


def test_late_verdict_reaches_the_model_before_the_turn_ends(mock_workspace, monkeypatch):
    """The turn's last edit gets a verdict after its tool result returned: the turn must not end on
    it (a one-shot run has no next turn).  The final-answer gate waits for it and hands it to the
    model as a new message, continuing the turn."""
    from types import SimpleNamespace

    from agent.turn_stop_gates import apply_stop_gates

    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    repo, gate = mock_workspace("by_content")
    target = repo / "c.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result, _elapsed = _patch(target, "x = 1", "x = bad", session="S-end")
    assert result["success"] is True and "lsp_diagnostics" not in result, result
    gate.touch()
    agent = SimpleNamespace(session_id="S-end", _current_task_id="t-S-end",
                            _interim_content_was_streamed=lambda _text: False,
                            _emit_interim_assistant_message=lambda _msg: None,
                            _flush_messages_to_session_db=lambda *_a: None)
    messages = [{"role": "user", "content": "edit c.py"}]
    verdict = apply_stop_gates(agent, {"role": "assistant", "content": "done"}, final_response="done",
                               messages=messages, conversation_history=[],
                               pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn, "the turn ended with the edit's verdict still undelivered"
    assert messages[-1]["role"] == "user" and "MOCK001" in messages[-1]["content"], messages[-1]


def test_diagnostics_stay_with_the_session_that_made_the_edit(mock_workspace):
    """Two sessions edit one file while the first check is still running: the error session A
    introduced is reported to A, and B (whose edit kept it, not introduced it) hears nothing."""
    repo, gate = mock_workspace("by_content")
    target = repo / "d.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result_a, _ = _patch(target, "x = 1", "x = bad", session="S-A")
    result_b, _ = _patch(target, "x = bad", "x = bad\ny = 2", session="S-B")
    assert result_a["success"] is True and result_b["success"] is True, (result_a, result_b)
    gate.touch()
    wait_until(lambda: _settled("S-A", "S-B"), timeout=_WAIT, what="both edit checks settling")
    notes_b = deferred.pop_notes("S-B")
    notes_a = deferred.pop_notes("S-A")
    assert not notes_b, f"session B was told about session A's error: {notes_b}"
    assert notes_a and "MOCK001" in notes_a[0][1], f"session A never heard about its own error: {notes_a}"


def test_each_session_is_checked_on_the_bytes_it_wrote(mock_workspace):
    """Session A's clean edit is still unchecked when session B writes an error into the same file:
    A's check must diagnose the bytes A wrote (clean), not the file as B left it, and B gets its error."""
    import threading

    repo, gate = mock_workspace("by_content")
    gate.touch()  # the server answers at once; the LSP loop itself is held below
    target = repo / "e.py"
    target.write_text("x = 1\n", encoding="utf-8")
    from agent.lsp import get_service
    svc = get_service()
    held, release = threading.Event(), threading.Event()

    def hold():
        held.set()
        release.wait(_WAIT)

    svc._loop._loop.call_soon_threadsafe(hold)
    assert held.wait(5)
    try:
        result_a, _ = _patch(target, "x = 1", "x = 2", session="S-A2")
        result_b, _ = _patch(target, "x = 2", "x = bad", session="S-B2")
    finally:
        release.set()
    assert result_a["success"] is True and result_b["success"] is True, (result_a, result_b)
    wait_until(lambda: _settled("S-A2", "S-B2"), timeout=_WAIT, what="both edit checks settling")
    notes_a = deferred.pop_notes("S-A2")
    notes_b = deferred.pop_notes("S-B2")
    assert not notes_a and "lsp_diagnostics" not in result_a, f"A was told about B's error: {notes_a or result_a}"
    told_b = " ".join(text for _path, text in notes_b) + str(result_b.get("lsp_diagnostics") or "")
    assert "MOCK001" in told_b, f"session B never heard about its own error: {notes_b}"
