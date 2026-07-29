"""Nous Portal ``anthropic/*`` models route on the native Messages wire.

Portal serves its ``anthropic/*`` catalog at
``https://inference-api.nousresearch.com/v1/messages`` alongside the
OpenAI-compatible ``/v1/chat/completions`` used by everything else it proxies.
These tests pin the contracts that make that routing correct:

1. ``api_mode`` is derived from the model, not hardcoded per provider.
2. The Anthropic SDK's base URL + auth land on Portal's route with Bearer auth.
3. Portal catalog ids are forwarded verbatim (it routes on ``vendor/model``).
4. Portal's ``tags`` / ``session_id`` body fields survive onto the new wire.
5. Signed thinking blocks replay like native Anthropic (not strip-all third-party).
6. Auxiliary clients inherit the same dual-wire split (Claude → Messages).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.providers import nous_api_mode

PORTAL_URL = "https://inference-api.nousresearch.com/v1"
# Staging / preview hosts used via NOUS_INFERENCE_BASE_URL — not the prod
# hostname, so Portal behaviour must key off provider=nous.
STAGING_URL = "https://ai.wildebeest-newton.ts.net/v1"


# ── 1. api_mode is model-derived ─────────────────────────────────────────────


class TestApiModeRouting:
    """``nous_api_mode`` / ``determine_api_mode`` split the Portal catalog."""

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-haiku-4.5",
            "ANTHROPIC/Claude-Sonnet-5",  # case-insensitive
        ],
    )
    def test_anthropic_prefixed_models_use_the_messages_wire(self, model):
        assert nous_api_mode(model) == "anthropic_messages"


    def test_a_claude_model_without_the_vendor_prefix_is_not_rerouted(self):
        """Portal ids carry the vendor prefix. A bare ``claude-*`` slug is not a
        Portal Anthropic id, so it must not be pushed onto the native wire."""
        assert nous_api_mode("claude-opus-4.8") == "chat_completions"

    def test_determine_api_mode_honors_the_model_for_nous(self):
        """Callers that skip resolve_runtime_provider (fallback, switch_model
        empty-mode path) must still land Claude on Messages — the Hermes
        overlay alone advertises openai_chat for every Nous model."""
        from hermes_cli.providers import determine_api_mode

        assert (
            determine_api_mode(
                "nous",
                PORTAL_URL,
                model="anthropic/claude-opus-4.8",
            )
            == "anthropic_messages"
        )
        assert (
            determine_api_mode("nous", PORTAL_URL, model="hermes-4-405b")
            == "chat_completions"
        )
        # No model → historical OpenAI-wire default (safer than guessing).
        assert determine_api_mode("nous", PORTAL_URL) == "chat_completions"


class TestRuntimeResolution:
    """The resolved runtime dict carries the model-derived mode end to end."""

    @pytest.fixture(autouse=True)
    def _stub_portal_credentials(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {})
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "nous")
        monkeypatch.setattr(rp, "load_pool", lambda p: SimpleNamespace(
            has_credentials=lambda: False,
        ))
        monkeypatch.setattr(
            rp,
            "resolve_nous_runtime_credentials",
            lambda **kw: {
                "base_url": PORTAL_URL,
                "api_key": "portal-invoke-jwt",
                "source": "portal",
                "expires_at": None,
            },
        )

    def test_anthropic_model_resolves_to_the_messages_wire(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "nous"})

        resolved = rp.resolve_runtime_provider(
            requested="nous", target_model="anthropic/claude-opus-5"
        )

        assert resolved["provider"] == "nous"
        assert resolved["api_mode"] == "anthropic_messages"
        # base_url keeps its /v1; the adapter strips it before the SDK
        # re-appends /v1/messages (see TestClientShape).
        assert resolved["base_url"] == PORTAL_URL


    def test_target_model_wins_over_the_persisted_default(self, monkeypatch):
        """A mid-session ``/model`` switch passes the model it is switching TO;
        deriving the mode from the stale config default would leave the session
        on the wrong wire."""
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"provider": "nous", "default": "hermes-4-405b"},
        )

        resolved = rp.resolve_runtime_provider(
            requested="nous", target_model="anthropic/claude-opus-5"
        )

        assert resolved["api_mode"] == "anthropic_messages"



class TestPoolRuntimeResolution:
    """Portal credential pools take a separate resolution path."""

    def _pool(self, entry):
        return SimpleNamespace(
            has_credentials=lambda: True,
            select=lambda: entry,
        )

    @pytest.fixture
    def portal_entry(self):
        return SimpleNamespace(
            provider="nous",
            source="device_code",
            runtime_api_key="pool-invoke-jwt",
            agent_key="pool-invoke-jwt",
            agent_key_expires_at="2099-01-01T00:00:00+00:00",
            scope="inference:invoke",
            runtime_base_url=PORTAL_URL,
        )

    def test_pool_entry_honors_the_model_derived_mode(self, monkeypatch, portal_entry):
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "nous")
        monkeypatch.setattr(rp, "_agent_key_is_usable", lambda *a, **k: True)
        monkeypatch.setattr(rp, "load_pool", lambda p: self._pool(portal_entry))
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "nous"})

        resolved = rp.resolve_runtime_provider(
            requested="nous", target_model="anthropic/claude-opus-4.8"
        )

        assert resolved["api_mode"] == "anthropic_messages"



# ── 2. Client shape: endpoint + auth ────────────────────────────────────────


class TestClientShape:


    def test_lookalike_host_does_not_get_portal_treatment(self):
        """Substring matching would hand a spoofed host the Portal JWT as a
        Bearer token. Hostname matching must reject it."""
        from agent.anthropic_adapter import (
            _is_nous_portal_endpoint,
            _requires_bearer_auth,
        )

        spoofed = "https://inference-api.nousresearch.com.attacker.test/v1"
        assert not _is_nous_portal_endpoint(spoofed)
        assert not _requires_bearer_auth(spoofed)


    def test_portal_jwt_authenticates_with_bearer_not_x_api_key(self):
        """Portal validates the OAuth invoke JWT as a Bearer credential, the
        same way its /chat/completions route does. Sending it as x-api-key
        (the adapter's third-party default) 401s."""
        from agent.anthropic_adapter import build_anthropic_client

        client = build_anthropic_client("portal-invoke-jwt", PORTAL_URL)

        assert client.auth_token == "portal-invoke-jwt"
        assert client.api_key is None

    def test_portal_bearer_does_not_also_send_env_anthropic_api_key(
        self, monkeypatch
    ):
        """The Anthropic SDK fills api_key from ANTHROPIC_API_KEY when the
        constructor omits it. Hermes loads that env from ~/.hermes/.env, so
        without an explicit clear every Portal request would dual-auth as
        X-Api-Key: sk-ant-… + Authorization: Bearer portal.jwt."""
        from agent.anthropic_adapter import build_anthropic_client

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
        client = build_anthropic_client("portal-invoke-jwt", PORTAL_URL)

        assert client.auth_token == "portal-invoke-jwt"
        assert client.api_key is None
        assert "X-Api-Key" not in client.auth_headers
        assert client.auth_headers.get("Authorization", "").startswith(
            "Bearer portal-invoke-jwt"
        )



# ── 3. Portal catalog ids are forwarded verbatim ─────────────────────────────


class TestModelIdPassthrough:
    def _kwargs(self, model, base_url):
        from agent.anthropic_adapter import build_anthropic_kwargs

        return build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            base_url=base_url,
        )

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-5",
            "anthropic/claude-haiku-4.5",
        ],
    )
    def test_portal_receives_the_catalog_id_unchanged(self, model):
        """Portal routes on its own ``vendor/model`` ids. Stripping the prefix
        or hyphenating the dots makes the model unresolvable there."""
        assert self._kwargs(model, PORTAL_URL)["model"] == model



    def test_native_anthropic_still_gets_the_normalized_slug(self):
        """The Portal carve-out must not leak into real Anthropic, which needs
        the bare hyphenated slug."""
        kwargs = self._kwargs("anthropic/claude-opus-4.8", "https://api.anthropic.com")
        assert kwargs["model"] == "claude-opus-4-8"



# ── 4. Portal body fields survive onto the Messages wire ─────────────────────


class TestPortalBodyFields:
    """``tags`` and ``session_id`` are top-level Portal body fields.

    They are produced by the Nous provider profile, which only the OpenAI-wire
    transport consults — so the Anthropic branch has to merge them in itself.
    """

    def _build(self, provider="nous", session_id="sess-abc123"):
        from agent.chat_completion_helpers import build_api_kwargs
        from agent.transports.anthropic import AnthropicTransport

        transport = AnthropicTransport()
        agent = SimpleNamespace(
            api_mode="anthropic_messages",
            provider=provider,
            model="anthropic/claude-opus-4.8",
            session_id=session_id,
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            request_overrides={},
            context_compressor=None,
            _ephemeral_max_output_tokens=None,
            _is_anthropic_oauth=False,
            _anthropic_base_url=PORTAL_URL,
            _oauth_1m_beta_disabled=False,
            _get_transport=lambda: transport,
            _prepare_anthropic_messages_for_api=lambda msgs: msgs,
            _anthropic_preserve_dots=lambda: False,
        )
        return build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    def test_portal_tags_reach_the_messages_request(self):
        from agent.portal_tags import hermes_client_tag

        tags = self._build()["extra_body"]["tags"]

        assert "product=hermes-agent" in tags
        assert hermes_client_tag() in tags
        assert all(isinstance(tag, str) for tag in tags), (
            "Portal skips non-string tag entries unpredictably"
        )

    def test_session_id_reaches_the_messages_request(self):
        extra_body = self._build(session_id="sess-abc123")["extra_body"]

        assert extra_body["session_id"] == "sess-abc123"
        assert "conversation=sess-abc123" in extra_body["tags"]





    def test_helper_merge_is_a_no_op_for_non_nous(self):
        from agent.chat_completion_helpers import (
            _merge_nous_portal_messages_extra_body,
        )

        kwargs = {"model": "claude-opus-4-8", "messages": []}
        agent = SimpleNamespace(provider="anthropic", session_id="s")
        assert _merge_nous_portal_messages_extra_body(agent, kwargs) is kwargs
        assert "extra_body" not in kwargs


# ── 5. Thinking signatures replay like native Anthropic ──────────────────────


class TestPortalThinkingReplay:
    """Portal proxies Claude to Anthropic-family backends that validate signed
    thinking blocks. The generic third-party strip would drop them and 400 the
    first tool-loop turn; Portal must take the native Anthropic replay path.
    """

    def _messages(self):
        return [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"cmd":"ls"}',
                        },
                    }
                ],
                "anthropic_content_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "plan the listing",
                        "signature": "sig-portal-abc",
                    },
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "terminal",
                        "input": {"cmd": "ls"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "name": "terminal",
                "content": "done",
            },
        ]

    def _assert_thinking_kept(self, base_url):
        from agent.anthropic_adapter import convert_messages_to_anthropic

        _system, converted = convert_messages_to_anthropic(
            self._messages(),
            base_url=base_url,
            model="anthropic/claude-opus-4.8",
        )
        assistant = next(m for m in converted if m["role"] == "assistant")
        thinking = [
            b
            for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]

        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "plan the listing"
        assert thinking[0]["signature"] == "sig-portal-abc"
        assert any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in assistant["content"]
        )

    def test_portal_keeps_signed_thinking_on_the_latest_assistant_turn(self):
        self._assert_thinking_kept(PORTAL_URL)

    def test_staging_host_with_env_override_keeps_signed_thinking(
        self, monkeypatch
    ):
        monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", STAGING_URL)
        self._assert_thinking_kept(STAGING_URL)

    def test_other_third_party_gateways_still_strip_thinking(self):
        """The Portal carve-out must not leak into MiniMax-style proxies."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        _system, converted = convert_messages_to_anthropic(
            self._messages(),
            base_url="https://api.minimax.io/anthropic",
            model="MiniMax-M2.7",
        )
        assistant = next(m for m in converted if m["role"] == "assistant")
        thinking = [
            b
            for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]

        assert thinking == []


# ── 6. Auxiliary clients inherit the dual-wire split ─────────────────────────


class TestAuxiliaryDualWire:
    """``resolve_provider_client('nous', …)`` must wrap Claude onto Messages."""


    def test_non_anthropic_catalog_model_stays_on_chat_completions(self):
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        plain = MagicMock(name="openai-client")
        plain.api_key = "portal-invoke-jwt"
        plain.base_url = PORTAL_URL

        with (
            patch(
                "agent.auxiliary_client._try_nous",
                return_value=(plain, "hermes-4-405b"),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            client, model = resolve_provider_client("nous", "hermes-4-405b")

        assert model == "hermes-4-405b"
        assert client is plain
        assert not isinstance(client, AnthropicAuxiliaryClient)



    def test_build_call_kwargs_includes_sticky_session_id(self):
        """Aux Messages calls must pin session_id, not just tags."""
        from agent.auxiliary_client import _build_call_kwargs
        from agent.portal_tags import (
            reset_conversation_context,
            set_conversation_context,
        )

        token = set_conversation_context("sess-sticky-aux")
        try:
            # Alias spelling + profile stubbed out so the fallback path is the
            # one under test (profile success already covers session_id).
            with patch(
                "providers.get_provider_profile",
                side_effect=ImportError("no profile"),
            ):
                kwargs = _build_call_kwargs(
                    "nous-portal",
                    "anthropic/claude-opus-4.8",
                    [{"role": "user", "content": "hi"}],
                    max_tokens=64,
                    base_url=PORTAL_URL,
                )
        finally:
            reset_conversation_context(token)

        extra = kwargs.get("extra_body", {})
        assert "tags" in extra
        assert extra.get("session_id") == "sess-sticky-aux"


    def test_aux_create_forwards_portal_catalog_id_verbatim(self):
        """Regression: adapter must pass base_url into build_anthropic_kwargs.

        Without it the Portal carve-out never fires and
        ``anthropic/claude-opus-4.8`` is normalized to ``claude-opus-4-8``,
        which Portal's Messages route cannot resolve.
        """
        from agent.auxiliary_client import AnthropicAuxiliaryClient

        captured = {}

        def _fake_create(client, api_kwargs, **kwargs):
            captured["model"] = api_kwargs.get("model")
            return SimpleNamespace(
                content=[],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=1, output_tokens=1, total_tokens=2
                ),
            )

        client = AnthropicAuxiliaryClient(
            MagicMock(name="anthropic-sdk"),
            "anthropic/claude-opus-4.8",
            "portal-invoke-jwt",
            PORTAL_URL,
        )
        with patch(
            "agent.anthropic_adapter.create_anthropic_message",
            side_effect=_fake_create,
        ):
            client.chat.completions.create(
                model="anthropic/claude-opus-4.8",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert captured["model"] == "anthropic/claude-opus-4.8"
