"""Provider / credential / failover tests for run_agent.AIAgent.

Split out of the former monolithic ``test_run_agent.py`` (8,699 lines) so the
per-file CI shard runner (``scripts/run_tests_parallel.py`` spawns one
subprocess per file and cannot split within a file) is not pinned by a single
file. Pure move: test bodies are byte-identical to the original. Shared
fixtures live in ``conftest.py``; shared mock builders in
``_run_agent_helpers.py``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.error_classifier import FailoverReason

from tests.run_agent._run_agent_helpers import _make_tool_defs


class TestProviderModelNormalization:
    def test_aiagent_strips_matching_native_provider_prefix(self):
        with (
            patch(
                "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                model="zai/glm-5.1",
                provider="zai",
                base_url="https://api.z.ai/api/paas/v4",
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        assert agent.model == "glm-5.1"

    def test_aiagent_keeps_aggregator_vendor_slug(self):
        with (
            patch(
                "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                model="anthropic/claude-sonnet-4.6",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        assert agent.model == "anthropic/claude-sonnet-4.6"

class TestNousCredentialRefresh:
    """Verify Nous credential refresh rebuilds the runtime client."""

    def test_try_refresh_nous_client_credentials_rebuilds_client(
        self, agent, monkeypatch
    ):
        agent.provider = "nous"
        agent.api_mode = "chat_completions"

        closed = {"value": False}
        retired = {"value": False}
        rebuilt = {"kwargs": None}
        captured = {}

        class _ExistingClient:
            def close(self):
                closed["value"] = True

        class _RebuiltClient:
            pass

        def _fake_resolve(**kwargs):
            captured.update(kwargs)
            return {
                "api_key": "new-nous-key",
                "base_url": "https://inference-api.nousresearch.com/v1",
            }

        def _fake_openai(**kwargs):
            rebuilt["kwargs"] = kwargs
            return _RebuiltClient()

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_nous_runtime_credentials", _fake_resolve
        )

        existing = _ExistingClient()
        agent.client = existing

        _orig_retire = agent._retire_shared_openai_client

        def _spy_retire(client, *, reason):
            if client is existing:
                retired["value"] = True
            return _orig_retire(client, reason=reason)

        monkeypatch.setattr(agent, "_retire_shared_openai_client", _spy_retire)

        with patch("run_agent.OpenAI", side_effect=_fake_openai):
            ok = agent._try_refresh_nous_client_credentials(force=True)

        assert ok is True
        # #70773: the replaced shared client is RETIRED (sockets shutdown,
        # FD release deferred to GC), never hard-closed from the refreshing
        # thread — close() releasing pool FDs cross-thread was the
        # TLS-FD→SQLite corruption vector.
        assert retired["value"] is True
        assert closed["value"] is False
        assert captured["force_refresh"] is True
        assert rebuilt["kwargs"]["api_key"] == "new-nous-key"
        assert (
            rebuilt["kwargs"]["base_url"] == "https://inference-api.nousresearch.com/v1"
        )
        assert "default_headers" not in rebuilt["kwargs"]
        assert isinstance(agent.client, _RebuiltClient)

class TestCredentialPoolRecovery:
    def test_recover_with_pool_rotates_on_402(self, agent):
        current = SimpleNamespace(label="primary")
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def current(self):
                return current

            def mark_exhausted_and_rotate(
                self,
                *,
                status_code,
                error_context=None,
                api_key_hint=None,
            ):
                assert status_code == 402
                assert error_context is None
                assert api_key_hint == agent.api_key
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=402,
            has_retried_429=False,
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_rotates_on_billing_reason_even_with_http_400(self, agent):
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def mark_exhausted_and_rotate(
                self,
                *,
                status_code,
                error_context=None,
                api_key_hint=None,
            ):
                assert status_code == 400
                assert error_context == {"reason": "out_of_extra_usage"}
                assert api_key_hint == agent.api_key
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=400,
            has_retried_429=False,
            classified_reason=FailoverReason.billing,
            error_context={"reason": "out_of_extra_usage"},
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_retries_first_429_then_rotates(self, agent):
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def current(self):
                return SimpleNamespace(label="primary")

            def entries(self):
                return []

            def mark_exhausted_and_rotate(
                self, *, status_code, error_context=None, api_key_hint=None
            ):
                assert status_code == 429
                assert error_context is None
                assert api_key_hint == agent.api_key
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
        )
        assert recovered is False
        assert retry_same is True
        agent._swap_credential.assert_not_called()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=True,
        )
        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)


    def test_recover_with_pool_refreshes_on_401(self, agent):
        """401 with successful refresh should swap to refreshed credential."""
        refreshed_entry = SimpleNamespace(label="refreshed-primary", id="abc")

        class _Pool:
            def try_refresh_matching(self, api_key_hint=None):
                return refreshed_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is True
        agent._swap_credential.assert_called_once_with(refreshed_entry)

    def test_recover_with_pool_401_same_entry_refreshes_stop_after_two(self, agent):
        """Repeated same-entry auth refreshes must eventually fall through.

        A single-entry OAuth pool re-mints a fresh token on every 401, so
        ``try_refresh_matching()`` reports success forever. The cap (#26080)
        must let the third consecutive same-entry refresh fall through
        (return not-recovered) so the fallback chain can activate instead of
        looping on the same dead credential.
        """
        refreshed_entry = SimpleNamespace(label="primary", id="abc")

        class _Pool:
            def try_refresh_matching(self, api_key_hint=None):
                return refreshed_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()
        agent._auth_pool_refresh_counts = {}

        first = agent._recover_with_credential_pool(status_code=401, has_retried_429=False)
        second = agent._recover_with_credential_pool(status_code=401, has_retried_429=False)
        third = agent._recover_with_credential_pool(status_code=401, has_retried_429=False)

        assert first == (True, False)
        assert second == (True, False)
        # Third same-entry refresh exceeds the cap → not recovered, fall through.
        assert third == (False, False)
        assert agent._swap_credential.call_count == 2

    def test_recover_with_pool_401_cap_is_per_entry(self, agent):
        """Rotating to a different entry resets the per-entry refresh tally."""
        entry_a = SimpleNamespace(label="primary", id="aaa")
        entry_b = SimpleNamespace(label="secondary", id="bbb")
        sequence = [entry_a, entry_a, entry_b, entry_b]

        class _Pool:
            def try_refresh_matching(self, api_key_hint=None):
                return sequence.pop(0)

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()
        agent._auth_pool_refresh_counts = {}

        # Two refreshes of entry_a, then two of entry_b — neither hits the cap,
        # so all four recover. The (provider, id) key isolates the tallies.
        results = [
            agent._recover_with_credential_pool(status_code=401, has_retried_429=False)
            for _ in range(4)
        ]

        assert all(r == (True, False) for r in results)
        assert agent._swap_credential.call_count == 4

    def test_recover_with_pool_rotates_on_401_when_refresh_fails(self, agent):
        """401 with failed refresh should rotate to next credential."""
        next_entry = SimpleNamespace(label="secondary", id="def")

        class _Pool:
            def try_refresh_matching(self, api_key_hint=None):
                return None  # refresh failed

            def mark_exhausted_and_rotate(
                self, *, status_code, error_context=None, api_key_hint=None
            ):
                assert status_code == 401
                assert error_context is None
                assert api_key_hint == agent.api_key
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_401_refresh_fails_no_more_credentials(self, agent):
        """401 with failed refresh and no other credentials returns not recovered."""

        class _Pool:
            def try_refresh_matching(self, api_key_hint=None):
                return None

            def mark_exhausted_and_rotate(
                self, *, status_code, error_context=None, api_key_hint=None
            ):
                assert error_context is None
                return None  # no more credentials

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is False
        agent._swap_credential.assert_not_called()

    def test_extract_api_error_context_uses_reset_timestamp_and_reason(self, agent):
        response = SimpleNamespace(headers={})
        error = SimpleNamespace(
            body={
                "error": {
                    "code": "device_code_exhausted",
                    "message": "Weekly credits exhausted.",
                    "resets_at": "2026-04-12T10:30:00Z",
                }
            },
            response=response,
        )

        context = agent._extract_api_error_context(error)

        assert context["reason"] == "device_code_exhausted"
        assert context["message"] == "Weekly credits exhausted."
        assert context["reset_at"] == "2026-04-12T10:30:00Z"

    def test_extract_api_error_context_uses_type_as_reason(self, agent):
        error = SimpleNamespace(
            body={
                "error": {
                    "type": "usage_limit_reached",
                    "message": "The usage limit has been reached",
                }
            },
            response=SimpleNamespace(headers={}),
        )

        context = agent._extract_api_error_context(error)

        assert context["reason"] == "usage_limit_reached"
        assert context["message"] == "The usage limit has been reached"

    def test_extract_api_error_context_parses_resets_in_hours_and_minutes(self, agent, monkeypatch):
        from agent import agent_runtime_helpers

        monkeypatch.setattr(agent_runtime_helpers.time, "time", lambda: 1_000.0)
        error = SimpleNamespace(
            body={
                "error": {
                    "type": "GoUsageLimitError",
                    "message": "Weekly usage limit reached. Resets in 6hr 29min.",
                }
            },
            response=SimpleNamespace(headers={}),
        )

        context = agent._extract_api_error_context(error)

        assert context["reason"] == "GoUsageLimitError"
        assert context["reset_at"] == 1_000.0 + (6 * 60 * 60) + (29 * 60)

    def test_recover_with_pool_passes_error_context_on_rotated_429(self, agent):
        next_entry = SimpleNamespace(label="secondary")
        captured = {}

        class _Pool:
            def current(self):
                return SimpleNamespace(label="primary")

            def entries(self):
                return []

            def mark_exhausted_and_rotate(
                self, *, status_code, error_context=None, api_key_hint=None
            ):
                captured["status_code"] = status_code
                captured["error_context"] = error_context
                captured["api_key_hint"] = api_key_hint
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=True,
            error_context={"reason": "device_code_exhausted", "reset_at": "2026-04-12T10:30:00Z"},
        )

        assert recovered is True
        assert retry_same is False
        assert captured["status_code"] == 429
        assert captured["error_context"]["reason"] == "device_code_exhausted"

class TestAnthropicImageFallback:
    def test_build_api_kwargs_converts_multimodal_user_image_to_text(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None

        api_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you see this now?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }]

        with (
            patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=json.dumps({"success": True, "analysis": "A cat sitting on a chair."}))),
            patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        kwargs = mock_build.call_args.kwargs or dict(zip(
            ["model", "messages", "tools", "max_tokens", "reasoning_config"],
            mock_build.call_args.args,
        ))
        transformed = kwargs["messages"]
        assert isinstance(transformed[0]["content"], str)
        assert "A cat sitting on a chair." in transformed[0]["content"]
        assert "Can you see this now?" in transformed[0]["content"]
        assert "vision_analyze with image_url: https://example.com/cat.png" in transformed[0]["content"]

    def test_build_api_kwargs_reuses_cached_image_analysis_for_duplicate_images(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None
        data_url = "data:image/png;base64,QUFBQQ=="

        api_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ]

        mock_vision = AsyncMock(return_value=json.dumps({"success": True, "analysis": "A small test image."}))
        with (
            patch("tools.vision_tools.vision_analyze_tool", new=mock_vision),
            patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        assert mock_vision.await_count == 1

class TestFallbackAnthropicProvider:
    """Bug fix: _try_activate_fallback had no case for anthropic provider."""

    def test_fallback_to_anthropic_sets_api_mode(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-test"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            mock_build.return_value = MagicMock()
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "anthropic_messages"
        assert agent._anthropic_client is not None
        assert agent.client is None

    def test_fallback_to_anthropic_enables_prompt_caching(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-test"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            agent._try_activate_fallback()

        assert agent._use_prompt_caching is True

    def test_fallback_to_openrouter_uses_openai_client(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://openrouter.ai/api/v1"
        mock_client.api_key = "sk-or-test"

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "chat_completions"
        assert agent.client is mock_client

class TestAnthropicBaseUrlPassthrough:
    """Bug fix: base_url was filtered with 'anthropic in base_url', blocking proxies."""

    def test_custom_proxy_base_url_passed_through(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant-api03-test1234567890",
                base_url="https://llm-proxy.company.com/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # base_url should be passed through, not filtered out
            assert call_args[0][1] == "https://llm-proxy.company.com/v1"

    def test_none_base_url_passed_as_none(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant...7890",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # No base_url provided, should be default empty string or None
            passed_url = call_args[0][1]
            assert not passed_url or passed_url is None

class TestAnthropicCredentialRefresh:
    def test_try_refresh_anthropic_client_credentials_rebuilds_client(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            old_client = MagicMock()
            new_client = MagicMock()
            mock_build.side_effect = [old_client, new_client]
            agent = AIAgent(
                api_key="sk-ant-oat01-stale-token",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant-oat01-stale-token"
        agent._anthropic_base_url = "https://api.anthropic.com"
        agent.provider = "anthropic"

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat01-fresh-token"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=new_client) as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is True

        old_client.close.assert_called_once()
        rebuild.assert_called_once_with(
            "sk-ant-oat01-fresh-token", "https://api.anthropic.com", timeout=None,
        )
        assert agent._anthropic_client is new_client
        assert agent._anthropic_api_key == "sk-ant-oat01-fresh-token"

    def test_try_refresh_anthropic_client_credentials_returns_false_when_token_unchanged(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-same-token",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        old_client = MagicMock()
        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant-oat01-same-token"

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat01-same-token"),
            patch("agent.anthropic_adapter.build_anthropic_client") as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is False

        old_client.close.assert_not_called()
        rebuild.assert_not_called()

    def test_anthropic_messages_create_preflights_refresh(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._anthropic_client = MagicMock()
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value.get_final_message.return_value = response
        agent._anthropic_client.messages.stream.return_value = stream_cm

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=True) as refresh:
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        refresh.assert_called_once_with()
        agent._anthropic_client.messages.stream.assert_called_once_with(model="claude-sonnet-4-20250514")
        agent._anthropic_client.messages.create.assert_not_called()
        assert result is response

    def test_anthropic_messages_create_falls_back_when_stream_unavailable(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = RuntimeError(
            "stream is not supported by this provider"
        )
        agent._anthropic_client.messages.create.return_value = response

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False):
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        agent._anthropic_client.messages.stream.assert_called_once_with(model="claude-sonnet-4-20250514")
        agent._anthropic_client.messages.create.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is response

    def test_anthropic_messages_create_honors_disable_streaming(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._disable_streaming = True
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.create.return_value = response

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False):
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        agent._anthropic_client.messages.stream.assert_not_called()
        agent._anthropic_client.messages.create.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is response

    def test_anthropic_messages_create_does_not_mask_bedrock_stream_validation_errors(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        exc = RuntimeError("ValidationException: InvokeModelWithResponseStream input malformed")
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = exc

        with (
            patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False),
            pytest.raises(RuntimeError, match="input malformed"),
        ):
            agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        agent._anthropic_client.messages.create.assert_not_called()

    def test_anthropic_messages_create_falls_back_for_bedrock_stream_access_denied(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = RuntimeError(
            "User is not authorized to perform: bedrock:InvokeModelWithResponseStream"
        )
        agent._anthropic_client.messages.create.return_value = response

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False):
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        agent._anthropic_client.messages.create.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is response

class TestOAuthFlagAfterCredentialRefresh:
    """_is_anthropic_oauth must update when token type changes during refresh."""

    def test_oauth_flag_updates_api_key_to_oauth(self, agent):
        """Refreshing from API key to OAuth token must set flag to True."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "sk-ant-api-old"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="sk-ant-setup-oauth-token"),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_oauth_flag_updates_oauth_to_api_key(self, agent):
        """Refreshing from OAuth to API key must set flag to False."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "sk-ant-setup-old"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="sk-ant-api03-new-key"),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is False

class TestFallbackSetsOAuthFlag:
    """_try_activate_fallback must set _is_anthropic_oauth for Anthropic fallbacks."""

    def test_fallback_to_anthropic_oauth_sets_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-setup-oauth-token"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_fallback_to_anthropic_api_key_clears_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-regular-key"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is False
