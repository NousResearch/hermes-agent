"""Regression tests for `/goal draft` word-boundary matching.

`/goal draft <objective>` expands plain text into a structured completion
contract via an aux-LLM call. Its detection guard must match ``draft`` as a
whole word — a bare ``lower.startswith("draft")`` misroutes any goal whose
first word merely *begins* with "draft" (``drafting``, ``drafts``,
``draftsman``) into the contract-draft path AND slices the first 5 chars off
the objective via ``args[len("draft"):]``.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    def __init__(self, session_id):
        self.session_id = session_id


class _FakeSessionStore:
    def __init__(self, session_id):
        self.entry = _FakeSessionEntry(session_id)

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-draft-prefix"


def _make_runner(session_id):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore(session_id)
    runner.adapters = {}
    runner._queued_events = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-draft-prefix",
            chat_type="channel",
            user_id="user-goal-draft-prefix",
        ),
        message_id="msg-goal-draft-prefix",
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
async def test_goal_starting_with_draft_word_is_not_truncated(hermes_home, monkeypatch):
    """`/goal drafting …` is a free-form goal, not a draft-contract request.

    It must be preserved verbatim (no ``draft`` prefix stripped) and must NOT
    invoke the aux-LLM ``draft_contract`` path.
    """
    called = {"draft": False}

    def _spy_draft_contract(objective, *args, **kwargs):
        called["draft"] = True
        return None

    monkeypatch.setattr(goals, "draft_contract", _spy_draft_contract)

    sid = "sid-goal-draft-prefix-truncate"
    runner = _make_runner(sid)
    event = _make_event("/goal drafting a response to the customer")

    response = await GatewayRunner._handle_goal_command(runner, event)

    assert called["draft"] is False, "free-form 'drafting…' goal must not hit draft_contract"

    state = goals.GoalManager(sid).state
    assert state is not None
    assert state.goal == "drafting a response to the customer"
    assert "drafting a response to the customer" in response


@pytest.mark.asyncio
async def test_goal_draft_still_routes_to_contract(hermes_home, monkeypatch):
    """A genuine `/goal draft <objective>` still routes to the draft path
    with the objective (minus the ``draft`` keyword) as the goal text."""
    captured = {}

    def _spy_draft_contract(objective, *args, **kwargs):
        captured["objective"] = objective
        return None

    monkeypatch.setattr(goals, "draft_contract", _spy_draft_contract)

    sid = "sid-goal-draft-prefix-contract"
    runner = _make_runner(sid)
    event = _make_event("/goal draft the quarterly report")

    response = await GatewayRunner._handle_goal_command(runner, event)

    assert captured.get("objective") == "the quarterly report"

    state = goals.GoalManager(sid).state
    assert state is not None
    assert state.goal == "the quarterly report"
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_goal_draft_empty_objective(hermes_home, monkeypatch):
    """Bare `/goal draft` still returns the usage message and does not set a goal."""
    monkeypatch.setattr(
        goals, "draft_contract", lambda *a, **k: pytest.fail("draft_contract must not run for bare draft")
    )

    sid = "sid-goal-draft-prefix-empty"
    runner = _make_runner(sid)
    event = _make_event("/goal draft")

    response = await GatewayRunner._handle_goal_command(runner, event)

    assert "Usage: /goal draft" in response
    assert goals.GoalManager(sid).state is None
