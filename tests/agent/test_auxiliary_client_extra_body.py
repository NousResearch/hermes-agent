"""Auxiliary task request configuration and Anthropic extra-body forwarding."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm

pytestmark = pytest.mark.usefixtures("auxiliary_client_state")


class TestAuxiliaryTaskExtraBody:
    def test_task_reasoning_disable_uses_deepseek_thinking_wire(self, monkeypatch):
        """Task-level ``none`` must reach an always-toggle profile as its native disable shape."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(
            aux,
            "_get_auxiliary_task_config",
            lambda _task: {"reasoning_effort": "none"},
        )

        kwargs = aux._build_call_kwargs(
            provider="deepseek",
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            extra_body=aux._get_task_extra_body("compression"),
            task="compression",
        )

        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning" not in kwargs["extra_body"]

    def test_explicit_deepseek_thinking_disable_beats_profile_default(self):
        """An explicit vendor control is authoritative when no normalized config is present."""
        import agent.auxiliary_client as aux

        kwargs = aux._build_call_kwargs(
            provider="deepseek",
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            extra_body={"thinking": {"type": "disabled"}},
            task="compression",
        )

        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_disabled_caller_reasoning_suppresses_task_reasoning_for_profile_wire(
        self, monkeypatch
    ):
        """A profile-owned ``reasoning_effort=none`` must not ship with task reasoning."""
        import agent.auxiliary_client as aux

        projection = aux._ProfileProjection({}, {}, {"reasoning_effort": "none"}, True)
        monkeypatch.setattr(aux, "_project_provider_profile", lambda *_args: projection)
        monkeypatch.setattr(
            aux,
            "_get_auxiliary_task_config",
            lambda _task: {
                "reasoning_effort": "low",
                "extra_body": {"metadata": {"task": "title"}},
            },
        )

        kwargs = aux._build_call_kwargs(
            provider="custom",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            extra_body=aux._get_task_extra_body("title_generation"),
            reasoning_config={"enabled": False},
            task="title_generation",
        )

        assert kwargs["reasoning_effort"] == "none"
        assert "reasoning" not in kwargs["extra_body"]
        assert kwargs["extra_body"]["metadata"] == {"task": "title"}

    def test_disabled_caller_reasoning_keeps_profile_owned_disable_shape(
        self, monkeypatch
    ):
        """Control: a profile whose disabled shape IS ``extra_body.reasoning`` (OpenRouter) keeps it —
        the caller's thinking-off replaces the task effort, it never deletes the profile's own field."""
        import agent.auxiliary_client as aux

        projection = aux._ProfileProjection(
            {}, {"reasoning": {"enabled": False}}, {}, True
        )
        monkeypatch.setattr(aux, "_project_provider_profile", lambda *_args: projection)
        monkeypatch.setattr(
            aux, "_get_auxiliary_task_config", lambda _task: {"reasoning_effort": "low"}
        )

        kwargs = aux._build_call_kwargs(
            provider="openrouter",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            extra_body=aux._get_task_extra_body("title_generation"),
            reasoning_config={"enabled": False},
            task="title_generation",
        )

        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
        assert "reasoning_effort" not in kwargs

    @pytest.mark.parametrize(
        "task", ["session_search", "moa_reference", "moa_aggregator"]
    )
    def test_generic_reasoning_fallback_clamps_ultra_for_auxiliary_and_moa_calls(
        self, task, monkeypatch
    ):
        """The OpenAI-compatible fallback must never put Hermes-only ``ultra`` on the wire."""
        from agent.auxiliary_client import _ProfileProjection, _build_call_kwargs

        monkeypatch.setattr(
            "agent.auxiliary_client._project_provider_profile",
            lambda *_args: _ProfileProjection({}, {}, {}, False),
        )

        kwargs = _build_call_kwargs(
            provider="custom",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            reasoning_config={"enabled": True, "effort": "ultra"},
            task=task,
        )

        assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}

    def test_task_extra_body_reasoning_effort_ultra_is_clamped(self, monkeypatch):
        """``auxiliary.<task>.reasoning_effort: ultra`` folds into ``extra_body.reasoning`` (the path
        compression/title/vision/... use, with no reasoning_config) and must take the same clamp."""
        import agent.auxiliary_client as aux
        from agent.auxiliary_client import _build_call_kwargs, _get_task_extra_body

        monkeypatch.setattr(
            aux,
            "_get_auxiliary_task_config",
            lambda task: {"reasoning_effort": "ultra"},
        )
        monkeypatch.setattr(
            aux,
            "_project_provider_profile",
            lambda *_args: aux._ProfileProjection({}, {}, {}, False),
        )

        kwargs = _build_call_kwargs(
            provider="custom",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            extra_body=_get_task_extra_body("session_search"),
            reasoning_config=None,
            task="session_search",
        )

        assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}

    def test_profile_projection_receives_wire_clamped_effort(self, monkeypatch):
        """Profiles clamp only against their own narrower sets (or a catalog that may be cold), so
        ``ultra`` must already be a wire level when the projection sees it — the MoA aggregator on
        an OpenRouter/Nous slot 400'd otherwise (#112010)."""
        import agent.auxiliary_client as aux

        seen = {}
        real = aux._project_provider_profile

        def spy(provider, provider_norm, model, effective_base, reasoning_config):
            seen["config"] = reasoning_config
            return real(
                provider, provider_norm, model, effective_base, reasoning_config
            )

        monkeypatch.setattr(aux, "_project_provider_profile", spy)
        kwargs = aux._build_call_kwargs(
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hello"}],
            reasoning_config={"enabled": True, "effort": "ultra"},
            task="moa_aggregator",
        )

        assert seen["config"] == {"enabled": True, "effort": "max"}
        assert (
            "ultra" not in json.dumps(kwargs.get("extra_body"))
            and kwargs.get("reasoning_effort") != "ultra"
        )

    def test_sync_call_merges_task_extra_body_from_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {
                        "enable_thinking": False,
                        "reasoning": {"effort": "none"},
                    }
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "glm-4.5-air"),
            ),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"metadata": {"source": "test"}},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is False
        assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}
        assert kwargs["extra_body"]["metadata"] == {"source": "test"}

    @pytest.mark.asyncio
    async def test_async_call_explicit_extra_body_overrides_task_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        config = {
            "auxiliary": {"session_search": {"extra_body": {"enable_thinking": False}}}
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "glm-4.5-air"),
            ),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"enable_thinking": True},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is True

    @pytest.mark.parametrize("moa_task", ["moa_reference", "moa_aggregator"])
    def test_moa_tasks_reject_task_level_reasoning_effort(self, moa_task, caplog):
        """MoA reasoning is per-slot in the preset — the auxiliary-task
        shorthand is ignored with a warning pointing at the preset config."""
        from agent.auxiliary_client import _get_task_extra_body

        config = {"auxiliary": {moa_task: {"reasoning_effort": "xhigh"}}}
        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"),
        ):
            result = _get_task_extra_body(moa_task)

        assert "reasoning" not in result
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_anthropic_aux_client_forwards_extra_body_reasoning(self):
        """_AnthropicCompletionsAdapter passes extra_body.reasoning into
        build_anthropic_kwargs as reasoning_config."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(
            MagicMock(), "claude-sonnet-4-6", is_oauth=False
        )

        with (
            patch(
                "agent.anthropic_adapter.build_anthropic_kwargs",
                return_value={
                    "model": "claude-sonnet-4-6",
                    "messages": [],
                    "max_tokens": 64,
                },
            ) as mock_bak,
            patch(
                "agent.anthropic_adapter.create_anthropic_message",
                return_value=SimpleNamespace(usage=None),
            ) as mock_create,
            patch("agent.transports.get_transport") as mock_gt,
        ):
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok",
                tool_calls=None,
                reasoning=None,
                finish_reason="stop",
                usage=None,
                provider_data=None,
            )
            adapter.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"reasoning": {"enabled": True, "effort": "low"}},
            )

        assert mock_bak.call_args.kwargs["reasoning_config"] == {
            "enabled": True,
            "effort": "low",
        }
        mock_create.assert_called_once()

    def _run_anthropic_adapter(self, *, call_extra_body=None, bak_result=None):
        """Drive _AnthropicCompletionsAdapter.create() with mocked SDK layers;
        return the api_kwargs handed to create_anthropic_message."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(
            MagicMock(), "claude-sonnet-4-6", is_oauth=False
        )
        bak_result = bak_result or {
            "model": "claude-sonnet-4-6",
            "messages": [],
            "max_tokens": 64,
        }
        with (
            patch(
                "agent.anthropic_adapter.build_anthropic_kwargs",
                return_value=dict(bak_result),
            ),
            patch(
                "agent.anthropic_adapter.create_anthropic_message",
                return_value=SimpleNamespace(usage=None),
            ) as mock_create,
            patch("agent.transports.get_transport") as mock_gt,
        ):
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok",
                tool_calls=None,
                reasoning=None,
                finish_reason="stop",
                usage=None,
                provider_data=None,
            )
            kwargs = {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
            }
            if call_extra_body is not None:
                kwargs["extra_body"] = call_extra_body
            adapter.create(**kwargs)
        return mock_create.call_args.args[1]

    def test_anthropic_aux_extra_body_passthrough(self):
        """Bug B (#37217): vendor fields in extra_body reach the Anthropic SDK."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={
                "thinking": {"type": "disabled"},
                "metadata": {"user_id": "u1"},
            },
        )
        assert api_kwargs["extra_body"] == {
            "thinking": {"type": "disabled"},
            "metadata": {"user_id": "u1"},
        }

    def test_bare_custom_auth_error_does_not_fall_back_to_env_base_url(
        self, monkeypatch
    ):
        """Bare 'custom' with nothing configured: the main resolver raises AuthError; aux must
        return no endpoint rather than route to a stale env OPENAI_BASE_URL with a placeholder key."""
        from hermes_cli.auth import AuthError
        from agent.auxiliary_client import _resolve_custom_runtime

        monkeypatch.setenv("OPENAI_BASE_URL", "https://old-proxy.example/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError(
                "no creds", provider="custom", code="missing_api_key"
            ),
        ):
            assert _resolve_custom_runtime() == (None, None, None)
