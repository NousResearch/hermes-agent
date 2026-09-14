"""Pre-turn image pre-analysis must route as the conversation it belongs to.

An OpenCode target (opencode-go/zen) rejects a request without a non-empty
``x-opencode-session`` ("MissingSessionID"). The gateway's pre-analysis runs BEFORE the agent turn
binds its conversation scope, so the aux vision call used to go out headerless and every image
degraded to the "couldn't quite see it" hint. These are the behaviour contracts for the fix.
"""

from __future__ import annotations

import asyncio

from agent import auxiliary_client as aux
from agent.portal_tags import (
    get_affinity_scope,
    get_conversation_context,
    reset_conversation_context,
    set_conversation_context,
)
from gateway.config import Platform
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource

_BASE_URL = "https://opencode.ai/zen/go/v1"
_MSGS = [{"role": "user", "content": "describe this"}]


class _Runner(GatewayInboundMixin):
    """Minimal stand-in: text routing, a resolved OpenCode runtime, captured header."""

    def __init__(self) -> None:
        self.captured_header = None
        self.captured_scope = None

    async def _decide_image_input_mode(
        self, *, source=None, session_key=None, user_config=None, provider=None, model=None,
    ) -> str:
        return "text"

    def _resolve_session_agent_runtime(self, *, source=None, session_key=None, user_config=None):
        return "deepseek-flash", {"provider": "opencode-go", "base_url": _BASE_URL}

    async def _enrich_message_with_vision(self, user_text: str, image_paths: list) -> str:
        kwargs = aux._build_call_kwargs("opencode-go", "deepseek-flash", _MSGS, base_url=_BASE_URL)
        self.captured_header = (kwargs.get("extra_headers") or {}).get("x-opencode-session")
        self.captured_scope = get_conversation_context()
        return "enriched"


_SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", user_id="6789")


def _run(runner: _Runner) -> str:
    return asyncio.run(runner._enrich_inbound_images(
        source=_SOURCE, session_key="telegram:12345", message_text="hi", image_paths=["/tmp/x.jpg"],
    ))


def test_preanalysis_sends_a_routable_affinity_header():
    runner = _Runner()
    assert _run(runner) == "enriched"
    assert runner.captured_header, "pre-analysis aux call went out without x-opencode-session"


def test_preanalysis_scope_is_released_afterwards():
    runner = _Runner()
    _run(runner)
    assert get_conversation_context() is None and get_affinity_scope() is None


def test_preanalysis_does_not_clobber_a_live_scope():
    token = set_conversation_context("live-conversation")
    try:
        runner = _Runner()
        _run(runner)
        assert runner.captured_header == "live-conversation"
        assert get_conversation_context() == "live-conversation"
    finally:
        reset_conversation_context(token)
