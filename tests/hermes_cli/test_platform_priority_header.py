"""Per-run LLM-gateway priority: ``platform_priority`` → ``X-Platform-Priority``.

A single run may raise the priority it asks the gateway for. The profile's static
``extra_headers`` value stays the default for every run that asks for nothing, and a per-run
value must survive a client rebuild (credential swap / route change) instead of silently
falling back to the profile default.
"""

from pathlib import Path

import pytest

from hermes_cli.config_providers import (
    PLATFORM_PRIORITY_HEADER,
    apply_platform_priority_header_to_client_kwargs,
    normalize_platform_priority,
)


class _StubAgent:
    """Just enough agent for a header build: model, priority, user-header merge hook."""

    base_url = "https://llm-gateway.internal.example.com/v1"

    def __init__(self, priority=None, model="test-model"):
        self.platform_priority = priority
        self.model = model

    def _apply_user_default_headers(self):
        pass


def _write_gateway_config(base_url: str, priority: str) -> None:
    """Profile config with the static per-provider ``extra_headers`` value."""
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    (home / "config.yaml").write_text(
        "providers:\n"
        "  platform-gateway:\n"
        "    name: Platform LLM gateway\n"
        f"    base_url: {base_url}\n"
        "    extra_headers:\n"
        f"      {PLATFORM_PRIORITY_HEADER}: {priority}\n"
    )


def test_normalize_accepts_normal_and_high_case_insensitively():
    assert normalize_platform_priority("normal") == "normal"
    assert normalize_platform_priority("HIGH") == "high"
    assert normalize_platform_priority("  high ") == "high"


@pytest.mark.parametrize("bad", ["critical", "low", "", None, 42, ["high"], True])
def test_normalize_rejects_every_other_value(bad):
    """``critical`` is a policy review gate — a client may never ask for it."""
    with pytest.raises(ValueError):
        normalize_platform_priority(bad)


def test_apply_overrides_profile_value_and_keeps_sibling_headers():
    kwargs = {
        "default_headers": {
            PLATFORM_PRIORITY_HEADER: "normal",
            "CF-Access-Client-Id": "client-id.access",
        }
    }
    apply_platform_priority_header_to_client_kwargs(kwargs, "high")
    assert kwargs["default_headers"] == {
        PLATFORM_PRIORITY_HEADER: "high",
        "CF-Access-Client-Id": "client-id.access",
    }


def test_apply_without_a_run_value_leaves_the_profile_header_alone():
    kwargs = {"default_headers": {PLATFORM_PRIORITY_HEADER: "normal"}}
    apply_platform_priority_header_to_client_kwargs(kwargs, None)
    assert kwargs["default_headers"] == {PLATFORM_PRIORITY_HEADER: "normal"}
    apply_platform_priority_header_to_client_kwargs({}, "")
    assert {} == {}


def test_init_header_build_lifts_the_run_priority_over_the_profile_config():
    """Real config loader: profile says normal, this run says high."""
    from agent.agent_init import _apply_openai_header_policy

    _write_gateway_config(_StubAgent.base_url, "normal")
    client_kwargs = {"base_url": _StubAgent.base_url}
    _apply_openai_header_policy(_StubAgent("high"), client_kwargs)
    assert client_kwargs["default_headers"][PLATFORM_PRIORITY_HEADER] == "high"


def test_init_header_build_without_a_run_priority_keeps_the_profile_value():
    from agent.agent_init import _apply_openai_header_policy

    _write_gateway_config(_StubAgent.base_url, "normal")
    client_kwargs = {"base_url": _StubAgent.base_url}
    _apply_openai_header_policy(_StubAgent(None), client_kwargs)
    assert client_kwargs["default_headers"][PLATFORM_PRIORITY_HEADER] == "normal"


def test_client_rebuild_keeps_the_run_priority():
    """Credential swap / route change rebuilds the headers — the run's value must survive."""
    from agent.client_lifecycle import ClientLifecycleMixin

    _write_gateway_config(_StubAgent.base_url, "normal")

    class _RebuildAgent:
        api_mode = "chat_completions"
        provider = "custom"
        platform_priority = "high"
        _client_kwargs = {"base_url": _StubAgent.base_url}
        _apply_user_default_headers = _StubAgent._apply_user_default_headers

    agent = _RebuildAgent()
    ClientLifecycleMixin._apply_client_headers_for_base_url(agent, _StubAgent.base_url)
    assert agent._client_kwargs["default_headers"][PLATFORM_PRIORITY_HEADER] == "high"
