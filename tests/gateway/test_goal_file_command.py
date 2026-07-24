"""Security tests: messaging platforms must reject ``/goal --file <path>``.

A remote chat command must never read a file from the Hermes backend host.
These tests prove the rejection happens BEFORE any file I/O, leaves the
existing goal state untouched, and enqueues no kickoff event — while plain
inline goals and inline contract parsing still work on the gateway.
"""
import pytest
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    session_id = "sid-goal-file-reject"


class _FakeSessionStore:
    def get_or_create_session(self, source):
        return _FakeSessionEntry()

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-file-reject"


class _FakeAdapter:
    """Minimal adapter stand-in so the goal handler's kickoff branch is
    reachable. The handler calls ``_enqueue_fifo(session_key, event, adapter)``
    only when ``adapters.get(platform)`` is truthy — an empty ``adapters``
    dict would silently mask a bug that wrongly reached the enqueue branch.
    A real-looking adapter here makes the "no kickoff on --file rejection"
    assertion meaningful: it proves the rejection short-circuited rather
    than that the enqueue path was simply unreachable."""


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    # A truthy adapter so the kickoff branch is genuinely exercised — this
    # is what separates "rejection worked" from "enqueue was unreachable".
    runner.adapters = {Platform.DISCORD: _FakeAdapter()}
    runner._queued_events = {}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-file",
            chat_type="channel",
            user_id="user-goal-file",
        ),
        message_id="msg-goal-file",
    )


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_goal_file_rejected_on_messaging_platform(hermes_home, tmp_path):
    runner = _make_runner()
    # Write a real file to prove it is never read (the rejection must fire
    # before any path resolution or file I/O).
    secret = tmp_path / "secret.txt"
    secret.write_text("SHOULD NEVER BE READ", encoding="utf-8")

    response = await GatewayRunner._handle_goal_command(
        runner, _make_event(f"/goal --file {secret}")
    )

    assert "messaging platforms" in response.lower()
    # No goal persisted.
    assert goals.GoalManager("sid-goal-file-reject").state is None


@pytest.mark.asyncio
async def test_goal_file_rejection_does_no_file_io(hermes_home, tmp_path):
    """The file must never be opened: monkeypatch read_text to fail loud so
    any accidental read surfaces as an assertion failure rather than passing
    silently."""
    runner = _make_runner()
    secret = tmp_path / "secret.txt"
    secret.write_text("leak", encoding="utf-8")

    import pathlib

    original_read_text = pathlib.Path.read_text

    def _fail_if_secret(self, *args, **kwargs):
        if self == secret:
            raise AssertionError("gateway read a backend file -- security boundary broken")
        return original_read_text(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pathlib.Path, "read_text", _fail_if_secret)
        response = await GatewayRunner._handle_goal_command(
            runner, _make_event(f"/goal --file {secret}")
        )
    assert "messaging platforms" in response.lower()


@pytest.mark.asyncio
async def test_goal_file_rejection_preserves_existing_goal(hermes_home):
    runner = _make_runner()
    # Establish an active goal first.
    await GatewayRunner._handle_goal_command(runner, _make_event("/goal original goal"))

    response = await GatewayRunner._handle_goal_command(
        runner, _make_event("/goal --file /etc/passwd")
    )
    assert "messaging platforms" in response.lower()

    state = goals.GoalManager("sid-goal-file-reject").state
    assert state is not None
    assert state.goal == "original goal"
    assert state.status == "active"


@pytest.mark.asyncio
async def test_goal_file_rejection_enqueues_no_kickoff(hermes_home, tmp_path):
    """``--file`` must be rejected before the kickoff branch, so
    ``_enqueue_fifo`` is never called. With a truthy adapter on the runner
    (see ``_make_runner``), the enqueue path is genuinely reachable — so
    asserting it was NOT called proves the rejection short-circuited, rather
    than that enqueue was unreachable. ``side_effect=AssertionError`` makes
    any accidental call fail loud instead of passing silently."""
    runner = _make_runner()
    secret = tmp_path / "secret.txt"
    secret.write_text("leak", encoding="utf-8")

    with patch.object(
        runner, "_enqueue_fifo", side_effect=AssertionError("must not enqueue on --file rejection")
    ) as mock_enqueue:
        response = await GatewayRunner._handle_goal_command(
            runner, _make_event(f"/goal --file {secret}")
        )
    assert "messaging platforms" in response.lower()
    mock_enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_inline_goal_does_enqueue_kickoff(hermes_home):
    """Control: a plain inline goal DOES call ``_enqueue_fifo`` (the kickoff
    branch is reachable on the happy path). This proves the no-kickoff
    assertion above is a real behavioral guard, not a tautology."""
    runner = _make_runner()
    with patch.object(runner, "_enqueue_fifo") as mock_enqueue:
        response = await GatewayRunner._handle_goal_command(
            runner, _make_event("/goal ship it")
        )
    assert "⊙ Goal set" in response
    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_goal_file_bare_token_rejected(hermes_home):
    runner = _make_runner()
    response = await GatewayRunner._handle_goal_command(
        runner, _make_event("/goal --file")
    )
    assert "messaging platforms" in response.lower() or "Usage" in response


@pytest.mark.asyncio
async def test_plain_inline_goal_still_works_on_gateway(hermes_home):
    runner = _make_runner()
    response = await GatewayRunner._handle_goal_command(
        runner, _make_event("/goal ship the benchmark")
    )
    assert "⊙ Goal set" in response
    state = goals.GoalManager("sid-goal-file-reject").state
    assert state is not None
    assert state.goal == "ship the benchmark"


@pytest.mark.asyncio
async def test_inline_contract_parsing_still_works_on_gateway(hermes_home):
    runner = _make_runner()
    response = await GatewayRunner._handle_goal_command(
        runner, _make_event("/goal migrate auth\nverify: pytest -q")
    )
    assert "⊙ Goal set" in response
    state = goals.GoalManager("sid-goal-file-reject").state
    assert state.goal == "migrate auth"
    assert state.contract.verification == "pytest -q"
