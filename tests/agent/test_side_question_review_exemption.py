"""A plugin ``ctx.side_question`` fork is outside background-review admission.

``agent/side_question_request.py`` answers a plugin's post-turn question on a cache-parity fork
of the session's own agent (``build_cache_parity_fork``), the same builder the automatic review
and ``/btw`` use. The builder stamps the fork ``_foreground_exempt_fork``, so it is never a
foreground turn: it registers with neither the review-admission registry nor the idle queue,
takes no durable session turn lease, writes no transcript and carries no aggregate review
budget. It therefore neither blocks nor is blocked by the automatic review of the turn it
asks about — the two may decode beside each other, as ``/btw`` already may — and what bounds it
is ``agent/side_question_lease.py`` (one lease per live agent, 60 s, text-only), not
``agent/review_admission.py``.

A real parent ``AIAgent`` runs a turn against a fake provider, the review slot is then occupied
as this turn's own automatic review would occupy it, and the side question is redeemed while
every foreground-registration seam is spied on.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.background_review as bg
from agent import review_admission, side_question_lease, turn_facade_lease
from agent.review_idle_queue import QUEUE as review_queue
from agent.side_question_lease import redeem_side_question
from agent.turn_facade_review import ForegroundReviewOwnership
from run_agent import AIAgent

SIDE_PROMPT = "Suggest the user's next prompt in one line."


def _usage(prompt=1200, completion=20, cached=1000):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _response(content="", finish_reason="stop", usage=None):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=usage,
    )


class _FakeProvider:
    """One fake client shared by the parent and every fork (``OpenAI(...)`` returns it)."""

    def __init__(self):
        self.client = MagicMock()
        self.requests: list[dict] = []
        self.replies: list = []
        self.client.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        return reply() if callable(reply) else reply


@pytest.fixture
def provider(monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    patches = [
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=fake.client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        # The consent gate has its own tests (test_side_question_lease.py).
        patch.object(
            side_question_lease, "_consented_plugin_loaded", return_value=True
        ),
    ]
    for p in patches:
        p.start()
    yield fake
    for p in reversed(patches):
        p.stop()
    side_question_lease._leases.clear()


@pytest.fixture
def issued(monkeypatch):
    """Tokens the finalizer leased, in order."""
    tokens: list = []
    real = side_question_lease.issue_side_question_token

    def _spy(*args, **kwargs):
        tokens.append(real(*args, **kwargs))
        return tokens[-1]

    monkeypatch.setattr(side_question_lease, "issue_side_question_token", _spy)
    return tokens


def _make_parent(**kwargs) -> AIAgent:
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **kwargs,
    )
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.skip_background_review = True
    return agent


def _run_parent_turn(parent, provider):
    provider.replies.append(_response("Found it: the answer is 42.", usage=_usage()))
    result = parent.run_conversation("look this up")
    assert result["final_response"] == "Found it: the answer is 42."


def test_plugin_side_question_answers_beside_the_turns_review_without_registering(
    provider,
    issued,
    monkeypatch,
):
    parent = _make_parent(session_id="sess-side")
    _run_parent_turn(parent, provider)
    profile_key = review_admission.current_profile_key()

    # The seams a foreground turn goes through, armed only now: the parent's own turn used them.
    entered: list = []
    monkeypatch.setattr(
        ForegroundReviewOwnership, "open", lambda self: entered.append("facade")
    )
    monkeypatch.setattr(
        review_admission,
        "note_turn_started",
        lambda *a, **k: entered.append("admission") or 0,
    )
    monkeypatch.setattr(
        review_queue, "note_turn_started", lambda: entered.append("idle_queue")
    )
    monkeypatch.setattr(
        turn_facade_lease,
        "admit_durable_turn_lease",
        lambda *a, **k: entered.append("turn_lease"),
    )
    forks: list = []
    real_build = bg.build_cache_parity_fork

    def _spy_build(*args, **kwargs):
        built = real_build(*args, **kwargs)
        forks.append(built[0])
        return built

    monkeypatch.setattr(bg, "build_cache_parity_fork", _spy_build)

    # This turn's automatic review owns the session's review slot while the plugin asks.
    occupant = bg.prepare_background_review_run(
        parent, session_id=parent.session_id, profile_key=profile_key
    )
    assert occupant is not None
    seen: dict = {}

    def _reply():
        # Sampled with the side request on the wire: the registry sees no live turn, so a review
        # may run (or keep running) beside it, and the occupant is still the slot's owner.
        seen.update(
            block=review_admission.foreground_block_reason(
                parent, None, profile_key, parent.session_id
            ),
            other_live=review_admission.other_live_turn(
                parent.session_id, None, profile_key
            ),
            occupant=review_admission.current_review_run(parent.session_id, profile_key)
            is occupant,
        )
        return _response("Ask: why 42?", usage=_usage())

    provider.replies.append(_reply)
    try:
        answer = redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt")
    finally:
        bg.finish_background_review_run(parent, occupant)

    assert answer == "Ask: why 42?"
    assert seen == {"block": None, "other_live": False, "occupant": True}
    assert entered == []
    (fork,) = forks
    assert fork._foreground_exempt_fork is True
    assert fork._persist_disabled is True and fork._session_db is None
    assert (
        fork._review_input_token_budget is None
    )  # the aggregate budget is the review's alone
    assert (
        getattr(fork, "_prompt_cache_fork_tag", None) is None
    )  # like /btw: the parent's slot
    assert fork._memory_write_origin == "side_question"
    assert review_admission.current_review_run(parent.session_id, profile_key) is None


def test_side_question_token_is_leased_before_the_turns_review_hand_off(
    provider, monkeypatch
):
    """The finalizer leases the token inside ``_apply_output_hooks`` (``post_llm_call``), before the
    turn's review candidate is handed off: a plugin redeeming synchronously in its hook runs on the
    turn thread while the turn still owns its session, and the turn's review still spawns exactly
    once, afterwards. Fork turns (the review, ``/btw``, the side question itself) lease nothing."""
    order: list = []
    real_issue = side_question_lease.issue_side_question_token
    monkeypatch.setattr(
        side_question_lease,
        "issue_side_question_token",
        lambda *a, **k: order.append("token") or real_issue(*a, **k),
    )
    monkeypatch.setattr(
        AIAgent, "_spawn_background_review", lambda self, **kw: order.append("review")
    )
    parent = _make_parent(session_id="sess-order")
    parent.skip_background_review = False
    parent._memory_nudge_interval = 1  # this turn trips the memory review
    parent._memory_store = object()
    parent.valid_tool_names = set(parent.valid_tool_names) | {"memory"}

    _run_parent_turn(parent, provider)

    assert order == ["token", "review"]
