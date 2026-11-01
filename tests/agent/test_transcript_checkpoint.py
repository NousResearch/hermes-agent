"""Turn-boundary transcript checkpoint for agents without a session DB (#92079)."""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent import transcript_checkpoint as tc
from agent.turn_iteration_prep import begin_iteration
from hermes_constants import get_hermes_home


def _dest(session_id: str):
    return get_hermes_home() / "transcripts" / f"transcript_{session_id}.json"


def _agent(**overrides):
    fields = dict(session_id="s1", _session_db=None, _persist_disabled=False, _transcript_checkpoint_every=10)
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("raw, expected", [(None, 10), (5, 5), ("7", 7), (0, 0), ("0", 0), (-3, 0)])
def test_resolve_interval(raw, expected):
    assert tc.resolve_interval(raw) == expected


def test_resolve_interval_malformed_keeps_default_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=tc.logger.name):
        assert tc.resolve_interval("often") == tc.DEFAULT_INTERVAL
    assert "transcript_checkpoint_every" in caplog.text


def test_write_replaces_previous_checkpoint_privately():
    agent = _agent()
    assert tc.write(agent, [{"role": "user", "content": "a"}], 10)
    assert tc.write(agent, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], 20)
    data = json.loads(_dest("s1").read_text(encoding="utf-8"))
    assert (data["api_call_count"], data["message_count"]) == (20, 2)
    assert [p.name for p in _dest("s1").parent.iterdir()] == ["transcript_s1.json"]  # no temp left behind
    if os.name == "posix":
        assert stat.S_IMODE(_dest("s1").stat().st_mode) == 0o600


def test_non_json_payload_is_stringified_not_fatal():
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert tc.write(_agent(), [{"role": "tool", "content": "x", "at": when}], 10)
    assert json.loads(_dest("s1").read_text(encoding="utf-8"))["messages"][0]["at"] == str(when)


@pytest.mark.parametrize("session_id", ["../escape", "a/b", ".hidden", ""])
def test_unsafe_or_missing_session_id_writes_nothing(session_id):
    assert tc.write(_agent(session_id=session_id), [], 10) is False
    assert not (get_hermes_home() / "transcripts").exists() or not any((get_hermes_home() / "transcripts").iterdir())


def test_write_failure_warns_and_does_not_raise(monkeypatch, caplog):
    import utils

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "atomic_json_write", _boom)
    with caplog.at_level(logging.WARNING, logger=tc.logger.name):
        assert tc.write(_agent(), [], 10) is False
    assert "transcript checkpoint failed" in caplog.text


@pytest.mark.parametrize("count, written", [(9, False), (10, True), (11, False), (20, True)])
def test_maybe_write_fires_on_the_interval(count, written):
    tc.maybe_write(_agent(), [], count)
    assert _dest("s1").exists() is written


@pytest.mark.parametrize("overrides", [
    {"_transcript_checkpoint_every": 0},          # disabled
    {"_session_db": object()},                    # the session DB already persists incrementally
    {"_persist_disabled": True},                  # persistence-isolated fork
])
def test_maybe_write_skips(overrides):
    tc.maybe_write(_agent(**overrides), [], 10)
    assert not _dest("s1").exists()


def test_begin_iteration_checkpoints_on_the_interval_call():
    """Wiring: the call that brings ``api_call_count`` to the interval writes the transcript."""
    agent = _agent(
        _interrupt_requested=False, quiet_mode=True, _budget_grace_call=False,
        _drain_pending_redirect=lambda: None, _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
        iteration_budget=SimpleNamespace(consume=lambda: True), _touch_activity=lambda _msg: None,
    )
    messages = [{"role": "user", "content": "hi"}]
    verdict = begin_iteration(
        agent, messages=messages, conversation_history=[], original_user_message="hi",
        api_call_count=9, interrupted=False, _turn_exit_reason="unknown",
    )
    assert verdict.api_call_count == 10
    assert json.loads(_dest("s1").read_text(encoding="utf-8"))["messages"] == messages
