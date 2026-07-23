"""Regression tests for provider-derived overrides across runtime switches."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


_ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
_ZAI_ALT_BASE_URL = "https://alt.z.ai/api/coding/paas/v4"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
_STALE_OVERRIDES = {"extra_body": {"thinking": {"type": "enabled"}}}


def _make_zai_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "glm-5.2"
    agent.provider = "custom:zai-coding-plan"
    agent.base_url = _ZAI_BASE_URL
    agent.api_key = "zai-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalZaiClient")
    agent._client_kwargs = {"api_key": "zai-key", "base_url": _ZAI_BASE_URL}
    agent.request_overrides = {"extra_body": {"thinking": {"type": "enabled"}}}
    agent._custom_providers = [
        {
            "provider_key": "zai-coding-plan",
            "name": "Z.AI Coding Plan",
            "base_url": _ZAI_BASE_URL,
            "model": "glm-5.2",
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    ]
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None
    agent._credential_pool = None
    return agent


def _make_fallback_agent(chain: list[dict]) -> AIAgent:
    """Build the smallest real fallback runtime needed by the activator."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = _make_zai_agent()
    primary_overrides = copy.deepcopy(agent.request_overrides)
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": agent.api_key,
        "client_kwargs": dict(agent._client_kwargs),
        "request_overrides": copy.deepcopy(primary_overrides),
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "compressor_model": agent.model,
        "compressor_base_url": agent.base_url,
        "compressor_api_key": agent.api_key,
        "compressor_provider": agent.provider,
        "compressor_context_length": 131072,
        "compressor_api_mode": agent.api_mode,
        "compressor_threshold_tokens": 100000,
    }
    agent._fallback_chain = list(chain)
    agent._fallback_index = 0
    agent._fallback_activated = False
    agent._unavailable_fallback_keys = set()
    agent._rate_limited_until = 0
    agent._transport_cache = {}
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.log_prefix = ""
    agent.reasoning_config = None
    agent._consecutive_stale_streams = 0
    agent._is_azure_openai_url = lambda _url: False
    agent._is_direct_openai_url = lambda _url: False
    agent._provider_model_requires_responses_api = lambda *_args, **_kwargs: False
    agent._anthropic_prompt_cache_policy = lambda **_kwargs: (False, False)
    agent._ensure_lmstudio_runtime_loaded = lambda: None
    agent._buffer_status = MagicMock()
    agent._replace_primary_openai_client = MagicMock()
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._close_openai_client = MagicMock()
    agent._try_activate_fallback = (
        lambda reason=None: try_activate_fallback(agent, reason)
    )
    return agent


def _fallback_client(base_url: str, api_key: str = "fallback-key") -> SimpleNamespace:
    return SimpleNamespace(
        api_key=api_key,
        base_url=base_url,
        _custom_headers={},
    )


class _StatefulCompressor:
    def __init__(self, agent: AIAgent, *, fail_model: str | None = None):
        self.model = agent.model
        self.context_length = 131072
        self.base_url = agent.base_url
        self.api_key = agent.api_key
        self.provider = agent.provider
        self.api_mode = agent.api_mode
        self.threshold_tokens = 100000
        self.fail_model = fail_model

    def update_model(
        self,
        *,
        model,
        context_length,
        base_url,
        api_key,
        provider,
        api_mode,
    ):
        self.model = model
        self.context_length = context_length
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_tokens = context_length // 2
        if model == self.fail_model:
            raise RuntimeError(f"compressor rejected {model}")


def _fallback_patches(*, clients, custom_providers=None, fallback_pool=None):
    return (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=list(clients),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, _provider: model,
        ),
        patch("agent.credential_pool.load_pool", return_value=fallback_pool),
        patch(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            return_value=None,
        ),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=list(custom_providers or []),
        ),
    )


def test_switch_to_openai_codex_clears_custom_provider_extra_body():
    agent = _make_zai_agent()
    new_client = MagicMock(name="CodexClient")
    agent._create_openai_client = lambda *_args, **_kwargs: new_client

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="gpt-5.6-luna",
            new_provider="openai-codex",
            api_key="codex-key",
            base_url=_CODEX_BASE_URL,
            api_mode="codex_responses",
        )

    assert agent.request_overrides == {}
    assert agent._primary_runtime["request_overrides"] == {}
    assert agent.client is new_client


def test_switch_endpoint_rebuilds_extra_body_from_new_custom_provider_config():
    agent = _make_zai_agent()
    new_provider_config = {
        "provider_key": "zai-coding-plan",
        "name": "Z.AI Coding Plan",
        "base_url": _ZAI_ALT_BASE_URL,
        "model": "glm-5.3",
        "extra_body": {"enable_thinking": False},
    }
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[new_provider_config],
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="glm-5.3",
            new_provider="custom:zai-coding-plan",
            api_key="new-zai-key",
            base_url=_ZAI_ALT_BASE_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {"extra_body": {"enable_thinking": False}}
    assert agent._primary_runtime["request_overrides"] == {
        "extra_body": {"enable_thinking": False}
    }
    agent.request_overrides["extra_body"]["enable_thinking"] = True
    assert agent._primary_runtime["request_overrides"] == {
        "extra_body": {"enable_thinking": False}
    }


def test_switch_same_endpoint_rebuilds_extra_body_for_new_model():
    agent = _make_zai_agent()
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    configs = [
        {
            "provider_key": "zai-coding-plan",
            "base_url": _ZAI_BASE_URL,
            "model": "glm-5.2",
            "extra_body": {"candidate": "a"},
        },
        {
            "provider_key": "zai-coding-plan",
            "base_url": _ZAI_BASE_URL,
            "model": "glm-5.3",
            "extra_body": {"candidate": "b"},
        },
    ]

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=configs,
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="glm-5.3",
            new_provider="custom:zai-coding-plan",
            api_key="new-zai-key",
            base_url=_ZAI_BASE_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {"extra_body": {"candidate": "b"}}


def test_switch_model_rebuild_matches_target_config_with_url_case_and_slash():
    agent = _make_zai_agent()
    upper_runtime_url = "HTTPS://API.Z.AI/api/coding/paas/v4/"
    agent.base_url = upper_runtime_url
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    configs = [
        {
            "provider_key": "zai-coding-plan",
            "base_url": _ZAI_BASE_URL,
            "model": "glm-5.3",
            "extra_body": {"candidate": "b"},
        }
    ]

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=configs,
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="glm-5.3",
            new_provider="custom:zai-coding-plan",
            api_key="new-zai-key",
            base_url=upper_runtime_url,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {"extra_body": {"candidate": "b"}}


def test_switch_same_endpoint_clears_old_model_extra_body_when_target_has_none():
    agent = _make_zai_agent()
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    target_without_overrides = {
        "provider_key": "zai-coding-plan",
        "base_url": _ZAI_BASE_URL,
        "model": "glm-5.3",
    }

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[target_without_overrides],
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="glm-5.3",
            new_provider="custom:zai-coding-plan",
            api_key="new-zai-key",
            base_url=_ZAI_BASE_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {}


def test_failed_switch_restores_original_request_overrides():
    agent = _make_zai_agent()
    overrides_seen_during_rebuild = []

    def fail_client_rebuild(*_args, **_kwargs):
        overrides_seen_during_rebuild.append(dict(agent.request_overrides))
        raise RuntimeError("simulated client build failure")

    agent._create_openai_client = fail_client_rebuild

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="simulated client build failure"),
    ):
        agent.switch_model(
            new_model="gpt-5.6-luna",
            new_provider="openai-codex",
            api_key="codex-key",
            base_url=_CODEX_BASE_URL,
            api_mode="codex_responses",
        )

    assert overrides_seen_during_rebuild == [{}]
    assert agent.request_overrides == _STALE_OVERRIDES


def test_tui_one_turn_switch_isolates_then_restores_request_overrides():
    """A temporary provider must not leak overrides in either direction."""
    from tui_gateway import server

    agent = _make_zai_agent()
    original_overrides = {
        "extra_body": {
            "thinking": {"type": "enabled"},
            "route_local_marker": "keep-after-restore",
        }
    }
    agent.request_overrides = original_overrides
    agent._fallback_chain = [
        {
            "provider": "custom:backup",
            "model": "backup-model",
            "nested": {"owner": "primary"},
        }
    ]
    agent._fallback_model = copy.deepcopy(agent._fallback_chain[0])
    agent._fallback_index = 3
    agent._consecutive_stale_streams = 7
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    restore_snapshot = server._snapshot_agent_model_runtime(agent)

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers", return_value=[]
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="gpt-5.6-luna",
            new_provider="openai-codex",
            api_key="codex-key",
            base_url=_CODEX_BASE_URL,
            api_mode="codex_responses",
        )

        # The temporary request must not carry the original provider's body.
        assert agent.request_overrides == {}

        server._restore_agent_model_runtime(agent, restore_snapshot)

    assert agent.provider == "custom:zai-coding-plan"
    assert agent.base_url == _ZAI_BASE_URL
    assert agent.request_overrides == original_overrides
    assert agent._fallback_chain == [
        {
            "provider": "custom:backup",
            "model": "backup-model",
            "nested": {"owner": "primary"},
        }
    ]
    assert agent._fallback_model == agent._fallback_chain[0]
    assert agent._fallback_model is not agent._fallback_chain[0]
    assert agent._fallback_index == 3
    assert agent._consecutive_stale_streams == 7


def test_fallback_success_rebuilds_overrides_for_target_runtime():
    agent = _make_fallback_agent(
        [{"provider": "openai-codex", "model": "gpt-5.6-luna"}]
    )
    patches = _fallback_patches(
        clients=[
            (
                _fallback_client(_CODEX_BASE_URL, "codex-key"),
                "gpt-5.6-luna",
            )
        ]
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is True

    assert agent.provider == "openai-codex"
    assert agent.request_overrides == {}


def test_fallback_exact_runtime_and_canonical_endpoint_keeps_nested_overrides():
    from agent.agent_runtime_helpers import _rebuild_request_overrides_for_runtime

    agent = _make_zai_agent()
    agent.base_url = "HTTPS://API.Z.AI/api/coding/paas/v4/"
    agent.request_overrides["route_local"] = {"nested": {"owner": "same-runtime"}}
    original = copy.deepcopy(agent.request_overrides)
    original_identity = agent.request_overrides

    changed = _rebuild_request_overrides_for_runtime(
        agent,
        previous_provider="CUSTOM:ZAI-CODING-PLAN",
        previous_base_url=f"{_ZAI_BASE_URL}/",
        previous_model="GLM-5.2",
    )

    assert changed is False
    assert agent.request_overrides == original
    assert agent.request_overrides is original_identity


def test_runtime_override_rebuild_recomputes_fast_mode_for_target_model():
    from agent.agent_runtime_helpers import _rebuild_request_overrides_for_runtime

    agent = _make_zai_agent()
    agent.model = "claude-opus-4-6"
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.service_tier = "priority"

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.config.get_compatible_custom_providers", return_value=[]
        ),
    ):
        changed = _rebuild_request_overrides_for_runtime(
            agent,
            previous_provider="custom:zai-coding-plan",
            previous_base_url=_ZAI_BASE_URL,
            previous_model="glm-5.2",
        )

    assert changed is True
    assert agent.request_overrides == {"speed": "fast"}


def test_fallback_chain_rebuilds_overrides_for_the_candidate_that_succeeds():
    target_base_url = "https://target.example/v1"
    target_config = {
        "provider_key": "target",
        "name": "Target",
        "base_url": target_base_url,
        "model": "target-model",
        "extra_body": {"target_flag": True},
    }
    agent = _make_fallback_agent(
        [
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
            {
                "provider": "custom:target",
                "model": "target-model",
                "base_url": target_base_url,
            },
        ]
    )
    prompt_cache_calls = 0

    def fail_first_candidate(**_kwargs):
        nonlocal prompt_cache_calls
        prompt_cache_calls += 1
        if prompt_cache_calls == 1:
            raise RuntimeError("first candidate failed after runtime swap")
        return False, False

    agent._anthropic_prompt_cache_policy = fail_first_candidate
    patches = _fallback_patches(
        clients=[
            (_fallback_client(_CODEX_BASE_URL), "gpt-5.6-luna"),
            (_fallback_client(target_base_url), "target-model"),
        ],
        custom_providers=[target_config],
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is True

    assert agent.provider == "custom:target"
    assert agent.request_overrides == {"extra_body": {"target_flag": True}}


def test_failed_candidate_does_not_confuse_same_endpoint_successor_ownership():
    target_base_url = "https://target.example/v1"
    target_configs = [
        {
            "provider_key": "target",
            "base_url": target_base_url,
            "model": "candidate-a",
            "extra_body": {"candidate": "a"},
        },
        {
            "provider_key": "target",
            "base_url": target_base_url,
            "model": "candidate-b",
            "extra_body": {"candidate": "b"},
        },
    ]
    agent = _make_fallback_agent(
        [
            {
                "provider": "custom:target",
                "model": "candidate-a",
                "base_url": target_base_url,
            },
            {
                "provider": "custom:target",
                "model": "candidate-b",
                "base_url": target_base_url,
            },
        ]
    )
    prompt_cache_calls = 0

    def fail_first_candidate(**_kwargs):
        nonlocal prompt_cache_calls
        prompt_cache_calls += 1
        if prompt_cache_calls == 1:
            raise RuntimeError("candidate-a failed after runtime swap")
        return False, False

    agent._anthropic_prompt_cache_policy = fail_first_candidate
    patches = _fallback_patches(
        clients=[
            (_fallback_client(target_base_url), "candidate-a"),
            (_fallback_client(target_base_url), "candidate-b"),
        ],
        custom_providers=target_configs,
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is True

    assert agent.model == "candidate-b"
    assert agent.request_overrides == {"extra_body": {"candidate": "b"}}


def test_successful_fallback_rebuilds_overrides_for_same_endpoint_next_model():
    target_base_url = "https://target.example/v1"
    target_configs = [
        {
            "provider_key": "target",
            "base_url": target_base_url,
            "model": "candidate-a",
            "extra_body": {"candidate": "a"},
        },
        {
            "provider_key": "target",
            "base_url": target_base_url,
            "model": "candidate-b",
            "extra_body": {"candidate": "b"},
        },
    ]
    agent = _make_fallback_agent(
        [
            {
                "provider": "custom:target",
                "model": "candidate-a",
                "base_url": target_base_url,
            },
            {
                "provider": "custom:target",
                "model": "candidate-b",
                "base_url": target_base_url,
            },
        ]
    )
    patches = _fallback_patches(
        clients=[
            (_fallback_client(target_base_url), "candidate-a"),
            (_fallback_client(target_base_url), "candidate-b"),
        ],
        custom_providers=target_configs,
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        assert agent._try_activate_fallback() is True
        assert agent.request_overrides == {"extra_body": {"candidate": "a"}}
        assert agent._try_activate_fallback() is True

    assert agent.model == "candidate-b"
    assert agent.request_overrides == {"extra_body": {"candidate": "b"}}


def test_failed_candidate_transaction_does_not_pollute_successor_runtime():
    a_base_url = "https://candidate-a.example/v1"
    b_base_url = "https://candidate-b.example/v1"
    configs = [
        {
            "provider_key": "candidate-a",
            "base_url": a_base_url,
            "model": "model-a",
            "extra_body": {"candidate": "a"},
        },
        {
            "provider_key": "candidate-b",
            "base_url": b_base_url,
            "model": "model-b",
            "extra_body": {"candidate": "b"},
        },
    ]
    agent = _make_fallback_agent(
        [
            {
                "provider": "custom:candidate-a",
                "model": "model-a",
                "base_url": a_base_url,
            },
            {
                "provider": "custom:candidate-b",
                "model": "model-b",
                "base_url": b_base_url,
            },
        ]
    )
    agent.context_compressor = _StatefulCompressor(agent, fail_model="model-a")
    candidate_a = _fallback_client(a_base_url, "key-a")
    candidate_b = _fallback_client(b_base_url, "key-b")
    cache_policies = [(True, False), (False, True)]
    agent._anthropic_prompt_cache_policy = lambda **_kwargs: cache_policies.pop(0)
    agent._cached_system_prompt = (
        "Stable prefix\n"
        f"Model: {agent.model}\n"
        f"Provider: {agent.provider}\n"
    )
    patches = _fallback_patches(
        clients=[
            (candidate_a, "model-a"),
            (candidate_b, "model-b"),
        ],
        custom_providers=configs,
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patch("agent.model_metadata.get_model_context_length", return_value=65536),
    ):
        assert agent._try_activate_fallback() is True

    assert agent.model == "model-b"
    assert agent.provider == "custom:candidate-b"
    assert agent.base_url == b_base_url
    assert agent.api_key == "key-b"
    assert agent.client is candidate_b
    assert agent._client_kwargs["api_key"] == "key-b"
    assert agent._client_kwargs["base_url"] == b_base_url
    assert agent.request_overrides == {"extra_body": {"candidate": "b"}}
    assert agent._credential_pool is None
    assert agent._transport_cache == {}
    assert agent._use_prompt_caching is False
    assert agent._use_native_cache_layout is True
    assert agent._cached_system_prompt.endswith(
        "Model: model-b\nProvider: custom:candidate-b\n"
    )
    assert agent._fallback_activated is True
    assert agent.context_compressor.model == "model-b"
    assert agent.context_compressor.provider == "custom:candidate-b"
    assert agent.context_compressor.base_url == b_base_url
    assert agent.context_compressor.api_key == "key-b"
    assert agent.context_compressor.context_length == 65536
    agent._close_openai_client.assert_any_call(
        candidate_a,
        reason="fallback_activation_rollback",
        shared=True,
    )


def test_failed_fallback_candidate_restores_entry_overrides_before_exhaustion():
    agent = _make_fallback_agent(
        [{"provider": "openai-codex", "model": "gpt-5.6-luna"}]
    )
    original = copy.deepcopy(agent.request_overrides)

    def mutate_then_fail(**_kwargs):
        agent.request_overrides = {"extra_body": {"candidate_only": True}}
        raise RuntimeError("candidate activation failed")

    agent._anthropic_prompt_cache_policy = mutate_then_fail
    patches = _fallback_patches(
        clients=[(_fallback_client(_CODEX_BASE_URL), "gpt-5.6-luna")]
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is False

    assert agent.request_overrides == original


def test_terminal_fallback_failure_restores_atomic_entry_runtime():
    target_base_url = "https://target.example/v1"
    target_config = {
        "provider_key": "target",
        "base_url": target_base_url,
        "model": "candidate-a",
        "extra_body": {"candidate": "a"},
    }
    agent = _make_fallback_agent(
        [
            {
                "provider": "custom:target",
                "model": "candidate-a",
                "base_url": target_base_url,
            }
        ]
    )
    primary_client = agent.client
    primary_pool = MagicMock()
    primary_pool.provider = agent.provider
    agent._credential_pool = primary_pool
    agent._client_kwargs["default_headers"] = {
        "X-Primary": {"nested": "keep"}
    }
    agent._transport_cache = {"primary": object()}
    primary_transport_cache = dict(agent._transport_cache)
    agent._config_context_length = 77777
    agent._cached_system_prompt = (
        "Stable prefix\n"
        f"Model: {agent.model}\n"
        f"Provider: {agent.provider}\n"
    )
    primary_prompt = agent._cached_system_prompt
    primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": agent.api_key,
        "client": primary_client,
        "client_kwargs": copy.deepcopy(agent._client_kwargs),
        "request_overrides": copy.deepcopy(agent.request_overrides),
        "credential_pool": primary_pool,
        "transport_cache": primary_transport_cache,
        "config_context_length": agent._config_context_length,
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "cached_system_prompt": primary_prompt,
        "fallback_activated": agent._fallback_activated,
    }
    agent._anthropic_prompt_cache_policy = lambda **_kwargs: (True, True)
    agent._buffer_status = MagicMock(
        side_effect=RuntimeError("candidate failed after prompt rewrite")
    )
    patches = _fallback_patches(
        clients=[(_fallback_client(target_base_url), "candidate-a")],
        custom_providers=[target_config],
        fallback_pool=None,
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is False

    assert agent.model == primary_runtime["model"]
    assert agent.provider == primary_runtime["provider"]
    assert agent.base_url == primary_runtime["base_url"]
    assert agent.api_mode == primary_runtime["api_mode"]
    assert agent.api_key == primary_runtime["api_key"]
    assert agent.client is primary_runtime["client"]
    assert agent._client_kwargs == primary_runtime["client_kwargs"]
    assert agent.request_overrides == primary_runtime["request_overrides"]
    assert agent._credential_pool is primary_runtime["credential_pool"]
    assert agent._transport_cache == primary_runtime["transport_cache"]
    assert agent._config_context_length == primary_runtime["config_context_length"]
    assert agent._use_prompt_caching is primary_runtime["use_prompt_caching"]
    assert agent._use_native_cache_layout is primary_runtime["use_native_cache_layout"]
    assert agent._cached_system_prompt == primary_runtime["cached_system_prompt"]
    assert agent._fallback_activated is primary_runtime["fallback_activated"]
    agent._client_kwargs["default_headers"]["X-Primary"]["nested"] = "mutated"
    agent.request_overrides["extra_body"]["thinking"]["type"] = "mutated"
    assert primary_runtime["client_kwargs"]["default_headers"] == {
        "X-Primary": {"nested": "keep"}
    }
    assert primary_runtime["request_overrides"] == {
        "extra_body": {"thinking": {"type": "enabled"}}
    }


def test_failed_candidate_after_successful_fallback_restores_that_fallback():
    x_base_url = "https://fallback-x.example/v1"
    y_base_url = "https://fallback-y.example/v1"
    configs = [
        {
            "provider_key": "fallback-x",
            "base_url": x_base_url,
            "model": "model-x",
            "extra_body": {"candidate": "x"},
        },
        {
            "provider_key": "fallback-y",
            "base_url": y_base_url,
            "model": "model-y",
            "extra_body": {"candidate": "y"},
        },
    ]
    agent = _make_fallback_agent(
        [
            {
                "provider": "custom:fallback-x",
                "model": "model-x",
                "base_url": x_base_url,
            },
            {
                "provider": "custom:fallback-y",
                "model": "model-y",
                "base_url": y_base_url,
            },
        ]
    )
    candidate_x = _fallback_client(x_base_url, "key-x")
    candidate_y = _fallback_client(y_base_url, "key-y")
    agent._cached_system_prompt = (
        "Stable prefix\n"
        f"Model: {agent.model}\n"
        f"Provider: {agent.provider}\n"
    )
    agent._buffer_status = MagicMock(
        side_effect=[None, RuntimeError("fallback-y failed late")]
    )
    patches = _fallback_patches(
        clients=[
            (candidate_x, "model-x"),
            (candidate_y, "model-y"),
        ],
        custom_providers=configs,
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is True
        fallback_x_runtime = {
            "model": agent.model,
            "provider": agent.provider,
            "base_url": agent.base_url,
            "api_mode": agent.api_mode,
            "api_key": agent.api_key,
            "client": agent.client,
            "client_kwargs": copy.deepcopy(agent._client_kwargs),
            "request_overrides": copy.deepcopy(agent.request_overrides),
            "cached_system_prompt": agent._cached_system_prompt,
            "pending_notice": agent._pending_fallback_notice,
            "fallback_activated": agent._fallback_activated,
        }
        assert agent._try_activate_fallback() is False

    assert agent.model == fallback_x_runtime["model"]
    assert agent.provider == fallback_x_runtime["provider"]
    assert agent.base_url == fallback_x_runtime["base_url"]
    assert agent.api_mode == fallback_x_runtime["api_mode"]
    assert agent.api_key == fallback_x_runtime["api_key"]
    assert agent.client is fallback_x_runtime["client"]
    assert agent._client_kwargs == fallback_x_runtime["client_kwargs"]
    assert agent.request_overrides == fallback_x_runtime["request_overrides"]
    assert agent._cached_system_prompt == fallback_x_runtime["cached_system_prompt"]
    assert agent._pending_fallback_notice == fallback_x_runtime["pending_notice"]
    assert agent._fallback_activated is fallback_x_runtime["fallback_activated"]
    agent._close_openai_client.assert_any_call(
        candidate_y,
        reason="fallback_activation_rollback",
        shared=True,
    )


def test_restore_primary_runtime_deep_copies_primary_request_overrides():
    agent = _make_fallback_agent(
        [{"provider": "openai-codex", "model": "gpt-5.6-luna"}]
    )
    primary_snapshot = copy.deepcopy(agent._primary_runtime["request_overrides"])
    agent.context_compressor = MagicMock()
    patches = _fallback_patches(
        clients=[(_fallback_client(_CODEX_BASE_URL), "gpt-5.6-luna")]
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert agent._try_activate_fallback() is True
    assert agent.request_overrides == {}

    assert agent._restore_primary_runtime() is True
    assert agent.request_overrides == primary_snapshot
    agent.request_overrides["extra_body"]["mutated_after_restore"] = True
    assert agent._primary_runtime["request_overrides"] == primary_snapshot


def test_initial_primary_runtime_deep_copies_request_overrides():
    original = {"extra_body": {"thinking": {"type": "enabled"}}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=_ZAI_BASE_URL,
            provider="custom:zai-coding-plan",
            model="glm-5.2",
            request_overrides=original,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._primary_runtime["request_overrides"] == original
    original["extra_body"]["thinking"]["type"] = "mutated"
    assert agent._primary_runtime["request_overrides"] == {
        "extra_body": {"thinking": {"type": "enabled"}}
    }


def test_primary_transport_recovery_restores_request_overrides_from_snapshot():
    from agent.agent_runtime_helpers import try_recover_primary_transport

    agent = _make_fallback_agent([])
    primary_snapshot = copy.deepcopy(agent._primary_runtime["request_overrides"])
    agent.request_overrides = {"extra_body": {"candidate_only": True}}
    agent._fallback_activated = False
    agent._is_openrouter_url = lambda: False
    agent._vprint = MagicMock()

    class ReadTimeout(Exception):
        pass

    with patch("agent.agent_runtime_helpers.time.sleep", return_value=None):
        assert try_recover_primary_transport(
            agent,
            ReadTimeout(),
            retry_count=0,
            max_retries=1,
        ) is True

    assert agent.request_overrides == primary_snapshot
    agent.request_overrides["extra_body"]["after_recovery"] = True
    assert agent._primary_runtime["request_overrides"] == primary_snapshot
