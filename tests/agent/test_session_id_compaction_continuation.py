"""Session-id continuation across a context-compaction rotation.

Background: an OpenAI-wire custom provider (a LiteLLM gateway) forwards the
agent session id as request metadata so the upstream Langfuse trace carries it
as ``sessionId`` (see ``ChatCompletionsTransport._build_kwargs_from_profile`` /
``CodexResponsesTransport.build_kwargs``). When a long run crosses the
compression threshold, ``conversation_compression`` rotates ``agent.session_id``
to a fresh auto id and opens a child session. To keep every call in one logical
run attributed to the SAME root session, ``build_api_kwargs`` stamps
``agent._origin_session_id`` (captured once, before the first rotation) in
preference to the rotated ``agent.session_id`` — falling back to
``session_id`` when no compaction has occurred (a no-op for non-compacting runs).

These tests cover both halves of that contract:
  1. the custom-provider transport stamps whatever ``session_id`` it is given
     into ``extra_body.metadata.session_id`` (and not for non-custom providers);
  2. ``build_api_kwargs`` selects the origin id over a rotated ``session_id``.
"""

from agent.transports.chat_completions import ChatCompletionsTransport
from agent.chat_completion_helpers import build_api_kwargs
from providers import get_provider_profile


_MSGS = [{"role": "user", "content": "hi"}]
_UNSET = object()


def _build_chat_kwargs(transport, *, provider_profile, is_custom_provider, session_id):
    return transport.build_kwargs(
        model="openai/gpt-5.4-mini",
        messages=_MSGS,
        tools=None,
        provider_profile=provider_profile,
        max_tokens=None,
        max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
        timeout=300,
        reasoning_config=None,
        request_overrides=None,
        session_id=session_id,
        is_custom_provider=is_custom_provider,
        ollama_num_ctx=None,
    )


def _stamped_session_id(kwargs):
    extra_body = kwargs.get("extra_body") or {}
    return (extra_body.get("metadata") or {}).get("session_id")


class TestCustomProviderStampsSessionIdMetadata:
    """The transport forwards session_id as Langfuse metadata, gated to custom providers."""

    def test_profile_path_custom_provider_stamps(self):
        transport = ChatCompletionsTransport()
        kwargs = _build_chat_kwargs(
            transport,
            provider_profile=get_provider_profile("nvidia"),
            is_custom_provider=True,
            session_id="cron_root_123",
        )
        assert _stamped_session_id(kwargs) == "cron_root_123"

    def test_profile_path_non_custom_provider_does_not_stamp(self):
        transport = ChatCompletionsTransport()
        kwargs = _build_chat_kwargs(
            transport,
            provider_profile=get_provider_profile("nvidia"),
            is_custom_provider=False,
            session_id="cron_root_123",
        )
        assert _stamped_session_id(kwargs) is None

    def test_legacy_path_custom_provider_stamps(self):
        # provider_profile=None exercises the unregistered/custom legacy path.
        transport = ChatCompletionsTransport()
        kwargs = _build_chat_kwargs(
            transport,
            provider_profile=None,
            is_custom_provider=True,
            session_id="cron_root_123",
        )
        assert _stamped_session_id(kwargs) == "cron_root_123"

    def test_legacy_path_non_custom_provider_does_not_stamp(self):
        transport = ChatCompletionsTransport()
        kwargs = _build_chat_kwargs(
            transport,
            provider_profile=None,
            is_custom_provider=False,
            session_id="cron_root_123",
        )
        assert _stamped_session_id(kwargs) is None


class _RecordingTransport:
    """Captures the kwargs build_api_kwargs passes to the transport."""

    def __init__(self):
        self.captured = None

    def build_kwargs(self, **kwargs):
        self.captured = kwargs
        return {"_fake": True}


class _FakeCodexAgent:
    """Minimal agent that reaches the codex_responses branch of build_api_kwargs.

    The recording transport stands in for the real one, so the session-id
    SELECTION is exercised without importing run_agent / hitting the network.
    """

    api_mode = "codex_responses"
    tools = None
    base_url = "http://127.0.0.1:4000/v1"
    _base_url_lower = "http://127.0.0.1:4000/v1"
    _base_url_hostname = "127.0.0.1"
    provider = "custom"
    model = "openai/gpt-5.4-mini"
    reasoning_config = None
    max_tokens = None
    request_overrides = None

    def __init__(self, session_id, origin=_UNSET):
        self.session_id = session_id
        if origin is not _UNSET:
            self._origin_session_id = origin
        self._transport = _RecordingTransport()

    def _get_transport(self):
        return self._transport

    def _prepare_messages_for_non_vision_model(self, messages):
        return messages

    def _resolved_api_call_timeout(self):
        return 300


def _passed_session_id(agent):
    build_api_kwargs(agent, list(_MSGS))
    return agent._transport.captured["session_id"]


class TestOriginSessionIdSurvivesCompaction:
    """build_api_kwargs prefers the stable origin id over a rotated session_id."""

    def test_origin_wins_over_rotated_session_id(self):
        # After compaction: session_id has rotated to a child auto id, but the
        # origin id was captured first — the stamp must stay the origin id.
        agent = _FakeCodexAgent(
            session_id="20260620_070449_f190a4",
            origin="cron_6fab2e_20260620_070033",
        )
        assert _passed_session_id(agent) == "cron_6fab2e_20260620_070033"

    def test_threads_is_custom_provider(self):
        agent = _FakeCodexAgent(session_id="s", origin="root")
        build_api_kwargs(agent, list(_MSGS))
        assert agent._transport.captured["is_custom_provider"] is True

    def test_falls_back_to_session_id_without_origin(self):
        # No compaction has happened yet: no _origin_session_id attribute set,
        # so the stamp is identical to before (pure no-op).
        agent = _FakeCodexAgent(session_id="cron_6fab2e_20260620_070033")
        assert _passed_session_id(agent) == "cron_6fab2e_20260620_070033"

    def test_revert_case_origin_equals_current(self):
        # If a rotation is reverted on error, _origin_session_id equals the
        # still-current session_id, so origin-or-current resolves the same.
        agent = _FakeCodexAgent(session_id="cron_root", origin="cron_root")
        assert _passed_session_id(agent) == "cron_root"
