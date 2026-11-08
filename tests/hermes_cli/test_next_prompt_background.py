"""The real post-turn path: the owner verb is asked from a background thread, delivery lands on the app loop."""

import logging
import queue
import threading
from types import SimpleNamespace

from hermes_cli import gateway_side
from hermes_cli import next_prompt_suggestions as nps


class _Buffer:
    def __init__(self):
        self.text = ""
        self.suggestion = None


class _Loop:
    """Collects callbacks the way prompt_toolkit's loop would defer them to the UI thread."""

    def __init__(self):
        self.callbacks = []

    def call_soon_threadsafe(self, fn):
        self.callbacks.append(fn)


def _cli(loop=None):
    buf = _Buffer()
    return SimpleNamespace(
        _app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, loop=loop),
        _agent_running=False, _pending_input=queue.Queue(), _interrupt_queue=queue.Queue(),
        _last_turn_interrupted=False, _last_turn_failed=False, session_id="sess-1",
    ), buf


def _run_background(monkeypatch, cli, *, answer="commit that", connection=object()):
    """Run the hook with the real thread spawner and join the worker, so delivery has been queued on return."""
    threads, started = [], []

    def spawn(fn):
        thread = threading.Thread(target=fn)
        threads.append(thread)
        thread.start()

    def start(conn, session_id, verb, text=None):
        started.append((conn, session_id, verb, text))
        if isinstance(answer, Exception):
            raise answer
        return "np_1", lambda: answer

    monkeypatch.setattr(nps, "_setting_enabled", lambda: True)
    monkeypatch.setattr(nps, "_spawn", spawn)
    monkeypatch.setattr(gateway_side, "connection_for", lambda cli: connection)
    monkeypatch.setattr(gateway_side, "start", start)
    nps.maybe_suggest_next_prompt(cli)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    return started


def test_the_owner_is_asked_for_the_session_with_no_text(monkeypatch):
    cli, _ = _cli()
    connection = object()
    assert _run_background(monkeypatch, cli, connection=connection) == [(connection, "sess-1", "next_prompt", None)]


def test_keystroke_between_check_and_apply_leaves_no_ghost(monkeypatch):
    loop = _Loop()
    cli, buf = _cli(loop=loop)
    _run_background(monkeypatch, cli)
    assert len(loop.callbacks) == 1  # delivery is deferred to the UI thread, not applied by the worker
    assert buf.suggestion is None
    buf.text = "x"  # the user types before the UI thread gets to it
    loop.callbacks[0]()
    assert buf.suggestion is None and nps.pending_suggestion(cli) is None


def test_delivery_on_the_app_loop_shows_the_ghost(monkeypatch):
    loop = _Loop()
    cli, buf = _cli(loop=loop)
    _run_background(monkeypatch, cli)
    loop.callbacks[0]()
    assert buf.suggestion.text == "commit that"


def test_no_gateway_session_asks_nobody_and_shows_nothing(monkeypatch):
    loop = _Loop()
    cli, buf = _cli(loop=loop)
    assert _run_background(monkeypatch, cli, connection=None) == []
    assert loop.callbacks == [] and buf.suggestion is None


def test_owner_failure_logs_at_debug_and_shows_nothing(monkeypatch, caplog):
    loop = _Loop()
    cli, buf = _cli(loop=loop)
    with caplog.at_level(logging.DEBUG):
        _run_background(monkeypatch, cli, answer=RuntimeError("model unavailable"))
    assert loop.callbacks == [] and buf.suggestion is None
    assert not [r for r in caplog.records if r.name == nps.logger.name and r.levelno >= logging.WARNING]
