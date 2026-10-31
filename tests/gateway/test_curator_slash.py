"""Gateway /curator slash command (#68880, #68884 review fix)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@asynccontextmanager
async def _noop_scope():
    yield


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._async_profile_scope_for_source = lambda _source: _noop_scope()
    return runner


def _event(text: str) -> MessageEvent:
    source = SessionSource(platform="telegram", chat_id="1", user_id="u1")
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=None,
    )


def _emit(*lines):
    """Emit through the curator module's output sink (as the real CLI does)."""
    from hermes_cli import curator

    for line in lines:
        curator._emit(line)


@pytest.mark.asyncio
async def test_gateway_curator_status_returns_cli_output():
    """The handler returns the captured output of the curator CLI."""
    runner = _runner()

    def fake_cli_main(tokens):
        assert tokens == ["status"]
        _emit("curator: ENABLED", "  runs:           0")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake_cli_main):
        out = await runner._handle_curator_command(_event("/curator status"))

    assert "curator: ENABLED" in out
    assert "runs:" in out


@pytest.mark.asyncio
async def test_gateway_curator_defaults_to_status():
    runner = _runner()
    seen = {}

    def fake_cli_main(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: ENABLED")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake_cli_main):
        out = await runner._handle_curator_command(_event("/curator"))

    assert seen["tokens"] == ["status"]
    assert "curator: ENABLED" in out


@pytest.mark.asyncio
async def test_gateway_curator_calls_run_slash_not_cli_main_directly():
    """The handler must route through run_slash (the concurrency-safe entry)."""
    runner = _runner()

    with patch("hermes_cli.curator.run_slash", return_value="curator: ok") as run_slash:
        out = await runner._handle_curator_command(_event("/curator status"))

    run_slash.assert_called_once()
    assert run_slash.call_args.args[0] == "status"
    assert out == "curator: ok"


def test_run_slash_status():
    from hermes_cli.curator import run_slash

    with patch("hermes_cli.curator.cli_main", side_effect=lambda tokens: _emit("curator: ENABLED")):
        out = run_slash("status")
    assert "curator: ENABLED" in out


def test_run_slash_defaults_to_status():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: ok")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("")
    assert seen["tokens"] == ["status"]
    assert "curator: ok" in out


def test_run_slash_strips_curator_prefix():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("ok")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        run_slash("/curator status")
    assert seen["tokens"] == ["status"]


def test_run_slash_captures_output_without_global_stream_swap():
    """run_slash must not mutate process-global stdout/stderr."""
    import sys

    from hermes_cli.curator import run_slash

    real_stdout = sys.stdout
    with patch("hermes_cli.curator.cli_main", side_effect=lambda tokens: _emit("curator: ok")):
        out = run_slash("status")
    assert "curator: ok" in out
    assert sys.stdout is real_stdout


def test_run_slash_rollback_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("rollback")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_prune_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("prune")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_adopt_all_unmanaged_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("adopt --all-unmanaged")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_adopt_named_skill_not_blocked():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: adopted")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("adopt my-skill")
    assert "adopted" in out
    assert seen["tokens"][0] == "adopt"


def test_run_slash_rollback_allowed_with_y():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: rolled back")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("rollback -y")
    assert "rolled back" in out
    assert seen["tokens"][0] == "rollback"


@pytest.mark.asyncio
async def test_idle_dispatch_routes_curator_through_run_slash():
    """`/curator` must reach the handler through the idle command table.

    Calling `_handle_curator_command` alone does not prove gateway/run_busy.py
    lists curator and run_inbound routes that name to the handler.
    """
    runner = _runner()
    with patch("hermes_cli.curator.run_slash", return_value="curator: dispatched") as run_slash:
        handled, reply = await runner._hm_dispatch_canonical_command(
            _event("/curator status"), _event("/curator status").source, "k", "curator"
        )

    assert handled is True
    assert reply == "curator: dispatched"
    run_slash.assert_called_once()
    assert run_slash.call_args.args[0] == "status"


def _assert_no_stdio_swap(text: str, needle: str) -> None:
    import io
    import sys

    from hermes_cli.curator import run_slash

    class _Spy(io.TextIOBase):
        def __init__(self):
            self.buf = io.StringIO()

        def write(self, s):
            return self.buf.write(s)

        def flush(self):
            return None

    out_spy, err_spy = _Spy(), _Spy()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_spy, err_spy
    try:
        got = run_slash(text)
        assert sys.stdout is out_spy
        assert sys.stderr is err_spy
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    assert needle in got.lower()
    assert "no output" not in got.lower()
    assert out_spy.buf.getvalue() == ""
    assert err_spy.buf.getvalue() == ""


def test_run_slash_help_is_captured_not_leaked():
    _assert_no_stdio_swap("--help", "usage")


def test_run_slash_invalid_argument_is_captured_not_leaked():
    _assert_no_stdio_swap("not-a-real-subcommand", "error")


def test_second_review_is_busy_while_scheduler_holds_lock():
    from agent import curator as agent_curator

    assert agent_curator._curator_run_lock.acquire(blocking=False)
    try:
        result = agent_curator.run_curator_review(synchronous=True, dry_run=True)
    finally:
        agent_curator._curator_run_lock.release()
    assert result.get("busy") is True
    assert result.get("summary_so_far") == "already running"


def test_gateway_run_reports_busy_when_scheduler_holds_lock(monkeypatch):
    from agent import curator as agent_curator
    from hermes_cli.curator import run_slash

    monkeypatch.setattr(agent_curator, "is_enabled", lambda: True)
    monkeypatch.setattr(agent_curator, "get_consolidate", lambda: False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("review body ran while the scheduler held the lock")

    monkeypatch.setattr(agent_curator, "apply_automatic_transitions", _boom)
    assert agent_curator._curator_run_lock.acquire(blocking=False)
    try:
        out = run_slash("run")
    finally:
        agent_curator._curator_run_lock.release()
    assert "already running" in out
