"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"

    def test_anthropic_host_custom_provider_uses_anthropic_messages(self):
        """A custom provider on the native api.anthropic.com host (no
        "/anthropic" path suffix, name != "anthropic") must resolve to the
        anthropic_messages wire protocol — not default to chat_completions,
        which POSTs /v1/chat/completions and 404s. Mirrors the primary-path
        determine_api_mode() host check."""
        fbs = [
            {
                "provider": "cron-anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="https://api.anthropic.com"),
                    "claude-sonnet-4-6",
                ),
            ),
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.api_mode == "anthropic_messages"


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False

    def test_single_credential_returns_false(self):
        """With one credential that just 429'd, rotation has nowhere to go.

        The pool may still report has_available() True once cooldown expires,
        but retrying against the same entry will hit the same daily-quota
        429 and burn the retry budget.  Must fall back.
        """
        assert _pool_may_recover_from_rate_limit(_pool(1)) is False

    def test_single_credential_in_cooldown_returns_false(self):
        assert _pool_may_recover_from_rate_limit(_pool(1, has_available=False)) is False

    def test_two_credentials_available_returns_true(self):
        """With >1 credentials and at least one available, rotate instead of fallback."""
        assert _pool_may_recover_from_rate_limit(_pool(2)) is True

    def test_multiple_credentials_all_in_cooldown_returns_false(self):
        """All credentials cooling down — fall back rather than wait."""
        assert _pool_may_recover_from_rate_limit(_pool(3, has_available=False)) is False

    def test_many_credentials_available_returns_true(self):
        assert _pool_may_recover_from_rate_limit(_pool(10)) is True


# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )

    def test_skips_entry_matching_current_base_url_and_model(self):
        """Two custom_providers entries pointing at the same shim URL
        with the same model should dedup even if their provider names differ."""
        fbs = [
            # Different provider name but same shim URL + model — same backend.
            {"provider": "claude-cli-alt", "model": "claude-opus-4.7",
             "base_url": "http://127.0.0.1:7891/v1"},
            # Real different fallback.
            {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "claude-cli"
        agent.model = "claude-opus-4.7"
        agent.base_url = "http://127.0.0.1:7891/v1"

        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # Same shim/base_url+model entry skipped, second one used.
        assert called == [("openrouter", "anthropic/claude-opus-4.7")], (
            f"expected base_url-aware dedup, got call order: {called}"
        )

    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()


# ── Gemini thought-signature fallback guard ────────────────────────────────

GEMINI_OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

_NO_EXTRA = object()


def _tool_call_history(extra_content=_NO_EXTRA):
    """One assistant tool call; ``extra_content`` present only when passed."""
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "terminal", "arguments": "{}"},
    }
    if extra_content is not _NO_EXTRA:
        tool_call["extra_content"] = extra_content
    return [{"role": "assistant", "tool_calls": [tool_call]}]


def _gemini_routing_resolver(called, created=None):
    """Fake resolve_provider_client mirroring the real gemini routing:
    a GeminiNativeClient is returned only when the resolved base URL speaks
    Gemini's native REST API (agent/auxiliary_client.py gemini branch).
    """
    from agent.gemini_native_adapter import (
        DEFAULT_GEMINI_BASE_URL,
        GeminiNativeClient,
        is_native_gemini_base_url,
    )

    def _resolve(provider, model=None, raw_codex=False, **kwargs):
        called.append((provider, model))
        if provider in ("gemini", "google"):
            base_url = kwargs.get("explicit_base_url") or DEFAULT_GEMINI_BASE_URL
            if is_native_gemini_base_url(base_url):
                client = GeminiNativeClient(api_key="test-key")
                if created is not None:
                    created.append(client)
                return client, model
            return _mock_client(base_url=base_url), model
        return _mock_client(), model

    return _resolve


def _activate(agent, resolver):
    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolver):
        with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
            return agent._try_activate_fallback()


class TestGeminiFallbackToolHistoryGuard:
    def _agent_with_history(self, fbs, history):
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "anthropic"
        agent.model = "claude-opus-4"
        agent.context_compressor = None  # keep activation tail offline
        agent._last_api_messages_for_fallback = history
        return agent

    def test_skips_gemini_after_unsigned_tool_calls_and_uses_next_fallback(self):
        fbs = [
            {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        ]
        agent = self._agent_with_history(fbs, _tool_call_history())

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        # Compatibility is decided from the *resolved* transport, so gemini is
        # resolved first, then skipped in favor of the next entry.
        assert called == [
            ("gemini", "gemini-3.1-flash-lite"),
            ("openrouter", "anthropic/claude-sonnet-4"),
        ]
        assert agent.provider == "openrouter"
        assert agent.model == "anthropic/claude-sonnet-4"

    def test_allows_gemini_before_tool_history(self):
        from agent.gemini_native_adapter import GeminiNativeClient

        agent = self._agent_with_history(
            [{"provider": "gemini", "model": "gemini-3.1-flash-lite"}], [])

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        assert called == [("gemini", "gemini-3.1-flash-lite")]
        assert agent.provider == "gemini"
        assert isinstance(agent.client, GeminiNativeClient)

    def test_gemini_openai_compat_endpoint_not_skipped(self):
        """provider: gemini + Google /openai endpoint is NOT native — it can
        replay unsigned tool calls and must not be screened (#62332 review)."""
        fbs = [{
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "base_url": GEMINI_OPENAI_COMPAT_URL,
        }]
        agent = self._agent_with_history(fbs, _tool_call_history())

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        assert called == [("gemini", "gemini-3.1-flash-lite")]
        assert agent.provider == "gemini"
        assert agent.base_url == GEMINI_OPENAI_COMPAT_URL

    def test_gemini_named_aggregator_model_not_skipped(self):
        """A Gemini-named model on an aggregator is not a native transport
        and must not be skipped merely for its name (#62332 review)."""
        fbs = [{"provider": "openrouter", "model": "google/gemini-2.5-pro"}]
        agent = self._agent_with_history(fbs, _tool_call_history())

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        assert called == [("openrouter", "google/gemini-2.5-pro")]
        assert agent.provider == "openrouter"

    def test_google_provider_alias_screened_like_gemini(self):
        fbs = [
            {"provider": "google", "model": "gemini-3.1-flash-lite"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        ]
        agent = self._agent_with_history(fbs, _tool_call_history())

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        assert called[0] == ("google", "gemini-3.1-flash-lite")
        assert agent.provider == "openrouter"

    def test_unsigned_extra_content_shapes_still_skip(self):
        """Metadata presence alone is not a signature (#62332 review)."""
        for extra in (
            {},                                     # empty metadata
            {"unrelated": {"x": 1}},                # non-signature metadata
            {"google": {"thought_signature": ""}},  # empty signature value
            "raw-string",                           # non-dict extra_content
        ):
            fbs = [
                {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            ]
            agent = self._agent_with_history(fbs, _tool_call_history(extra))

            called = []
            ok = _activate(agent, _gemini_routing_resolver(called))

            assert ok is True, f"extra_content={extra!r}"
            assert agent.provider == "openrouter", (
                f"extra_content={extra!r} must not count as a Gemini signature"
            )

    def test_valid_signature_shapes_allow_native_gemini(self):
        for extra in (
            {"google": {"thought_signature": "sig-abc"}},
            {"google": {"thoughtSignature": "sig-abc"}},
            {"google": "sig-abc"},
            {"thought_signature": "sig-abc"},
        ):
            agent = self._agent_with_history(
                [{"provider": "gemini", "model": "gemini-3.1-flash-lite"}],
                _tool_call_history(extra))

            called = []
            ok = _activate(agent, _gemini_routing_resolver(called))

            assert ok is True, f"extra_content={extra!r}"
            assert agent.provider == "gemini", (
                f"extra_content={extra!r} is a valid signature and must not skip"
            )

    def test_tool_role_tool_calls_do_not_block(self):
        """The native adapter never emits functionCall parts for system or
        tool/function roles, so residual unsigned ``tool_calls`` there (e.g.
        a provider echo) must not skip a viable native Gemini fallback."""
        history = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "sig"}},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "ok",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }],
            },
        ]
        agent = self._agent_with_history(
            [{"provider": "gemini", "model": "gemini-3.1-flash-lite"}], history)

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        assert agent.provider == "gemini"

    def test_malformed_history_entries_do_not_crash_or_block(self):
        history = [
            "not-a-dict",
            {"role": "assistant", "tool_calls": "not-a-list"},
            {"role": "assistant", "tool_calls": [None, 42, "call"]},
        ]
        agent = self._agent_with_history(
            [{"provider": "gemini", "model": "gemini-3.1-flash-lite"}], history)

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is True
        # The native adapter emits no functionCall part for non-dict tool
        # calls, so none of these can trigger the missing-signature 400.
        assert agent.provider == "gemini"

    def test_gemini_only_chain_exhausts_without_permanent_blacklist(self):
        agent = self._agent_with_history(
            [{"provider": "gemini", "model": "gemini-3.1-flash-lite"}],
            _tool_call_history())
        original_client = agent.client

        called = []
        ok = _activate(agent, _gemini_routing_resolver(called))

        assert ok is False
        assert agent.provider == "anthropic"
        assert agent.model == "claude-opus-4"
        assert agent.client is original_client
        # Transcript-dependent skip must not permanently blacklist the entry.
        assert getattr(agent, "_unavailable_fallback_keys", set()) == set()

    def test_skipped_native_client_is_closed(self):
        agent = self._agent_with_history(
            [
                {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            ],
            _tool_call_history())

        called, created = [], []
        ok = _activate(agent, _gemini_routing_resolver(called, created))

        assert ok is True
        assert len(created) == 1
        assert created[0].is_closed is True
