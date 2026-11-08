"""Trigger gating for next-prompt suggestions: when a settled turn spawns the call, and which results land."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import next_prompt_suggestions as nps


class _Buffer:
    def __init__(self, text=""):
        self.text = text
        self.suggestion = None
        self.auto_suggest = None


def _cli(text="", enabled=True, queued=False):
    buf = _Buffer(text)
    cli = SimpleNamespace(
        _app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None),
        _agent_running=False, _pending_input=queue.Queue(), _interrupt_queue=queue.Queue(),
        _last_turn_interrupted=False, _last_turn_failed=False, session_id="sess-1",
        conversation_history=[{"role": "user", "content": "run the tests"},
                              {"role": "assistant", "content": "All green."}],
    )
    if queued:
        cli._pending_input.put("next")
    cli._buf = buf
    return cli


@pytest.fixture
def run_sync(monkeypatch):
    """Run the background worker inline and report the calls made."""
    calls = []

    def fake_request(cli):
        calls.append(cli.session_id)
        return "commit that"

    monkeypatch.setattr(nps, "_request_suggestion", fake_request)
    monkeypatch.setattr(nps, "_spawn", lambda fn: fn())
    return calls


def _enable(on=True):
    return patch.object(nps, "_setting_enabled", return_value=on)


def test_settled_turn_stores_and_shows_suggestion(run_sync):
    cli = _cli()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    assert run_sync == ["sess-1"]
    assert nps.pending_suggestion(cli) == "commit that"
    assert cli._buf.suggestion.text == "commit that"


def test_setting_off_makes_no_call(run_sync):
    cli = _cli()
    with _enable(False):
        nps.maybe_suggest_next_prompt(cli)
    assert run_sync == [] and nps.pending_suggestion(cli) is None


@pytest.mark.parametrize("make", [
    lambda: _cli(text="half typed"),
    lambda: _cli(queued=True),
])
def test_non_empty_input_or_queued_work_makes_no_call(run_sync, make):
    cli = make()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    assert run_sync == []


def test_interrupted_or_failed_turn_makes_no_call(run_sync):
    for attr in ("_last_turn_interrupted", "_last_turn_failed"):
        cli = _cli()
        setattr(cli, attr, True)
        with _enable():
            nps.maybe_suggest_next_prompt(cli)
    assert run_sync == []


def _deferred(monkeypatch, result="commit that"):
    release = threading.Event()
    jobs = []
    monkeypatch.setattr(nps, "_request_suggestion", lambda cli: result)
    monkeypatch.setattr(nps, "_spawn", jobs.append)
    return jobs, release


def test_result_after_user_started_typing_is_dropped(monkeypatch):
    jobs, _ = _deferred(monkeypatch)
    cli = _cli()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    cli._buf.text = "actually, something else"
    jobs[0]()
    assert nps.pending_suggestion(cli) is None and cli._buf.suggestion is None


def test_result_after_new_turn_started_is_dropped(monkeypatch):
    jobs, _ = _deferred(monkeypatch)
    cli = _cli()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    nps.clear_suggestion(cli)  # submit of a new prompt
    jobs[0]()
    assert nps.pending_suggestion(cli) is None


def test_result_while_agent_busy_or_work_queued_is_dropped(monkeypatch):
    for mutate in (lambda c: setattr(c, "_agent_running", True), lambda c: c._pending_input.put("x")):
        jobs, _ = _deferred(monkeypatch)
        cli = _cli()
        with _enable():
            nps.maybe_suggest_next_prompt(cli)
        mutate(cli)
        jobs[0]()
        assert nps.pending_suggestion(cli) is None


def test_empty_result_shows_nothing(monkeypatch):
    jobs, _ = _deferred(monkeypatch, result="")
    cli = _cli()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    jobs[0]()
    assert nps.pending_suggestion(cli) is None


def test_clear_removes_pending_and_visible_ghost(run_sync):
    cli = _cli()
    with _enable():
        nps.maybe_suggest_next_prompt(cli)
    assert cli._buf.suggestion is not None
    nps.clear_suggestion(cli)
    assert nps.pending_suggestion(cli) is None
    assert cli._buf.suggestion is None
