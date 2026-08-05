from types import SimpleNamespace

from agent.agent_init import _merge_custom_provider_extra_body




def test_custom_provider_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={
            "extra_body": {
                "reasoning_effort": "low",
                "caller_only": True,
            }
        },
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
        ],
    )

    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "caller_only": True,
    }




def test_named_custom_provider_extra_body_matches_provider_key():
    agent = SimpleNamespace(
        provider="custom:zai-coding-plan",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "provider_key": "other-provider",
                "name": "Other Provider",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": True},
            },
            {
                "provider_key": "zai-coding-plan",
                "name": "Z.AI Coding Plan",
                "base_url": "https://api.z.ai/api/coding/paas/v4/",
                "model": "glm-5.2",
                "extra_body": {"enable_thinking": False},
            },
        ],
    )

    assert agent.request_overrides == {"extra_body": {"enable_thinking": False}}


def test_bare_custom_provider_disambiguates_via_requested_provider():
    """Two providers: entries at the SAME (base_url, model) — one with
    extra_body, one without — must not cross-contaminate.

    ``agent.provider`` is always the bare canonical "custom" for named
    providers:/custom_providers: entries (the runtime resolver never emits
    "custom:<name>"); only ``agent.requested_provider`` carries the actual
    selected entry's identity ("vllm" vs "vllm-no-think"). Without filtering
    on it, the matcher fell through to "first entry with a non-empty
    extra_body" and applied the WRONG entry's extra_body regardless of which
    provider was actually selected (repro: a vLLM endpoint listed twice with
    the same model id, only "vllm-no-think" carrying
    extra_body.chat_template_kwargs.enable_thinking)."""
    custom_providers = [
        {
            "name": "vLLM",
            "provider_key": "vllm",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        },
        {
            "name": "vLLM No-Think",
            "provider_key": "vllm-no-think",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]

    agent_plain = SimpleNamespace(
        provider="custom",
        requested_provider="vllm",
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        base_url="http://192.168.15.115:8000/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(agent_plain, custom_providers)
    assert agent_plain.request_overrides == {}

    agent_no_think = SimpleNamespace(
        provider="custom",
        requested_provider="vllm-no-think",
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        base_url="http://192.168.15.115:8000/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(agent_no_think, custom_providers)
    assert agent_no_think.request_overrides == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_switching_away_from_extra_body_provider_clears_stale_value():
    """A second call for a provider WITHOUT extra_body must clear whatever
    the previous call (e.g. at agent init, before a live /model switch)
    applied — mirrors what switch_model() now does on every live switch."""
    custom_providers = [
        {
            "name": "vLLM",
            "provider_key": "vllm",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        },
        {
            "name": "vLLM No-Think",
            "provider_key": "vllm-no-think",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]
    agent = SimpleNamespace(
        provider="custom",
        requested_provider="vllm-no-think",
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        base_url="http://192.168.15.115:8000/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }

    # Simulate a live /model switch to "vllm" (no extra_body).
    agent.requested_provider = "vllm"
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides == {}


def test_bare_raw_entry_name_as_provider_resolves_via_known_identities():
    """hermes_cli/model_switch.py's live /model-switch path sets
    agent.provider to the entry's OWN raw identity (e.g. "vllm-no-think"),
    not "custom" or "custom:vllm-no-think" — a different convention than
    agent-init resolution. Matching must still work by recognizing the bare
    name against the configured custom_providers identities."""
    custom_providers = [
        {
            "name": "vLLM",
            "provider_key": "vllm",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        },
        {
            "name": "vLLM No-Think",
            "provider_key": "vllm-no-think",
            "base_url": "http://192.168.15.115:8000/v1",
            "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]

    agent_vllm = SimpleNamespace(
        provider="vllm",
        requested_provider="vllm",
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        base_url="http://192.168.15.115:8000/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(agent_vllm, custom_providers)
    assert agent_vllm.request_overrides == {}

    agent_no_think = SimpleNamespace(
        provider="vllm-no-think",
        requested_provider="vllm-no-think",
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        base_url="http://192.168.15.115:8000/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(agent_no_think, custom_providers)
    assert agent_no_think.request_overrides == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_bare_provider_name_matching_builtin_never_leaks_custom_extra_body():
    """A builtin provider name (never present in custom_providers) must not
    accidentally match — the known-identities fallback only engages for
    names that actually belong to a configured custom_providers entry."""
    agent = SimpleNamespace(
        provider="openai",
        requested_provider="openai",
        model="gpt-5.5",
        base_url="https://api.openai.com/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "vLLM No-Think",
                "provider_key": "vllm-no-think",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.5",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        ],
    )
    assert agent.request_overrides == {}


def test_switching_preserves_unrelated_caller_override_across_switch():
    """A caller-set override key unrelated to the custom-provider config
    (e.g. an explicit fast-mode service_tier) must survive a switch that
    also changes the custom-provider-derived extra_body."""
    custom_providers = [
        {
            "name": "A",
            "provider_key": "a",
            "base_url": "https://proxy.example.com/v1",
            "model": "m",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
        {
            "name": "B",
            "provider_key": "b",
            "base_url": "https://proxy.example.com/v1",
            "model": "m",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]
    agent = SimpleNamespace(
        provider="custom",
        requested_provider="a",
        model="m",
        base_url="https://proxy.example.com/v1",
        request_overrides={"extra_body": {"service_tier": "flex"}},
    )
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "service_tier": "flex",
        }
    }

    agent.requested_provider = "b"
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "service_tier": "flex",
        }
    }
