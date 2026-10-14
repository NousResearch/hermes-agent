"""Regression tests for provider-scoped ``ollama_num_ctx`` resolution.

Covers the num_ctx half of #123398: ``custom_providers[].models.<id>.ollama_num_ctx``
was never read — only the top-level ``model:`` block reached the request, so a
per-provider pin left the window at the /api/show probe (or Ollama's 2048 default)
regardless of config.
"""

from __future__ import annotations

from unittest.mock import patch

# Imported at collection time on purpose: run_agent's module body runs _early_recovery's
# marker probe against the checkout's git dir, which the per-test home-io guard refuses
# in a worktree whose git dir lives under the real HERMES_HOME.
import run_agent  # noqa: F401
from hermes_cli.config_providers import get_custom_provider_ollama_num_ctx


class TestGetCustomProviderOllamaNumCtx:
    def test_per_model_override_on_matching_route(self):
        custom = [
            {
                "base_url": "http://localhost:11434/v1",
                "models": {"mistral-small3.1:24b": {"ollama_num_ctx": 131072}},
            }
        ]
        assert (
            get_custom_provider_ollama_num_ctx(
                "mistral-small3.1:24b", "http://localhost:11434/v1", custom
            )
            == 131072
        )

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "http://localhost:11434/v1/",
                "models": {"m": {"ollama_num_ctx": 65536}},
            }
        ]
        assert (
            get_custom_provider_ollama_num_ctx("m", "http://localhost:11434/v1", custom)
            == 65536
        )

    def test_other_route_or_model_returns_none(self):
        custom = [
            {
                "base_url": "https://elsewhere.invalid/v1",
                "models": {"m": {"ollama_num_ctx": 65536}},
            },
            {
                "base_url": "http://localhost:11434/v1",
                "models": {"other-model": {"ollama_num_ctx": 65536}},
            },
        ]
        assert (
            get_custom_provider_ollama_num_ctx("m", "http://localhost:11434/v1", custom)
            is None
        )

    def test_invalid_or_non_positive_values_are_skipped(self):
        custom = [
            {
                "base_url": "http://localhost:11434/v1",
                "models": {"m": {"ollama_num_ctx": "131K"}},
            }
        ]
        assert (
            get_custom_provider_ollama_num_ctx("m", "http://localhost:11434/v1", custom)
            is None
        )
        assert (
            get_custom_provider_ollama_num_ctx(
                "m",
                "http://localhost:11434/v1",
                [
                    {
                        "base_url": "http://localhost:11434/v1",
                        "models": {"m": {"ollama_num_ctx": 0}},
                    }
                ],
            )
            is None
        )

    def test_empty_inputs_return_none(self):
        assert (
            get_custom_provider_ollama_num_ctx(
                "", "http://x", [{"base_url": "http://x"}]
            )
            is None
        )
        assert get_custom_provider_ollama_num_ctx("m", "", [{"base_url": ""}]) is None
        assert get_custom_provider_ollama_num_ctx("m", "http://x", None) is None


def _build_agent(
    cfg,
    probed_ctx,
    base_url="http://localhost:11434/v1",
    model="gemma3:27b",
    probed_num_ctx=None,
):
    import agent.context_compressor as cc_mod

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=probed_ctx,
        ),
        patch.object(
            cc_mod,
            "get_model_context_length",
            return_value=probed_ctx,
        ),
        patch(
            "agent.agent_init.query_ollama_num_ctx", return_value=probed_num_ctx
        ) as probe,
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model=model,
            api_key="ollama",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent, probe


class TestProviderScopedNumCtxReachesTheAgent:
    """A custom_providers[].models.<model>.ollama_num_ctx pin must behave exactly
    like the top-level model.ollama_num_ctx: honoured, no probe, not capped."""

    def test_per_model_pin_is_honoured_without_a_probe(self):
        cfg = {
            "agent": {},
            "custom_providers": [
                {
                    "name": "ollama-local",
                    "base_url": "http://localhost:11434/v1",
                    "models": {"gemma3:27b": {"ollama_num_ctx": 131072}},
                }
            ],
        }
        agent, probe = _build_agent(cfg, probed_ctx=262144)
        assert agent._ollama_num_ctx == 131072
        probe.assert_not_called()

    def test_keyed_providers_form_is_honoured_too(self):
        cfg = {
            "agent": {},
            "providers": {
                "ollama-local": {
                    "api": "http://localhost:11434/v1",
                    "models": {"gemma3:27b": {"ollama_num_ctx": 98304}},
                }
            },
        }
        agent, _probe = _build_agent(cfg, probed_ctx=262144)
        assert agent._ollama_num_ctx == 98304

    def test_top_level_model_block_wins_over_the_provider_pin(self):
        cfg = {
            "agent": {},
            "model": {"ollama_num_ctx": 65536},
            "custom_providers": [
                {
                    "name": "ollama-local",
                    "base_url": "http://localhost:11434/v1",
                    "models": {"gemma3:27b": {"ollama_num_ctx": 131072}},
                }
            ],
        }
        agent, _probe = _build_agent(cfg, probed_ctx=262144)
        assert agent._ollama_num_ctx == 65536

    def test_mismatched_route_falls_back_to_the_probe(self):
        cfg = {
            "agent": {},
            "custom_providers": [
                {
                    "name": "ollama-elsewhere",
                    "base_url": "http://127.0.0.1:9999/v1",
                    "models": {"gemma3:27b": {"ollama_num_ctx": 131072}},
                }
            ],
        }
        agent, probe = _build_agent(cfg, probed_ctx=262144, probed_num_ctx=4096)
        assert agent._ollama_num_ctx == 4096
        probe.assert_called_once()

    def test_provider_pin_is_not_capped_by_context_length(self):
        # An explicit pin keeps the "never override an explicit num_ctx" contract the
        # top-level model.ollama_num_ctx already has, even when a route-scoped
        # context_length is smaller.
        cfg = {
            "agent": {},
            "custom_providers": [
                {
                    "name": "ollama-local",
                    "base_url": "http://localhost:11434/v1",
                    "context_length": 65536,
                    "models": {"gemma3:27b": {"ollama_num_ctx": 131072}},
                }
            ],
        }
        agent, _probe = _build_agent(cfg, probed_ctx=262144)
        assert agent._ollama_num_ctx == 131072
