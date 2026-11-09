"""External-process providers keep a local server's stall patience, keyed on auth_type (#149555).

The ``process://`` URL used to earn that patience by accident, because ``is_local_endpoint`` reads a
dot-less scheme host as a LAN name (#129153). The fixture's URL has a dot on purpose: the profile's
``auth_type`` decides now, and the budget no longer changes with the model name.
"""

from __future__ import annotations

import pytest

PROVIDER = "patience-fixture"


@pytest.fixture
def process_agent(monkeypatch):
    from hermes_constants import get_hermes_home
    from providers import get_provider_profile
    from run_agent import AIAgent

    for name in ("HERMES_API_CALL_STALE_TIMEOUT", "HERMES_STREAM_STALE_TIMEOUT", "HERMES_LOCAL_STREAM_STALE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    plugin = get_hermes_home() / "plugins" / PROVIDER
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        f"name: {PROVIDER}\nkind: model-provider\nversion: 0.1.0\ndescription: test fixture\n")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"register_provider(ProviderProfile(name='{PROVIDER}', auth_type='external_process', "
        "base_url='process://patience.fixture', supports_streaming=True))\n")
    profile = get_provider_profile(PROVIDER)
    monkeypatch.setattr(profile, "create_client", lambda **kwargs: object())
    return AIAgent(provider=PROVIDER, model="claude-opus-5-5[1m]", api_key="fixture", base_url=profile.base_url,
                   enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True)


def _budgets(agent, model: str) -> tuple[float, float]:
    from agent.chat_completion_helpers import _StreamingCall

    agent.model = model
    payload = {"model": model, "messages": [{"role": "user", "content": "x" * 1_300_000}]}
    call = _StreamingCall(agent, payload, None)
    call._resolve_stale_timeout()
    return agent._compute_non_stream_stale_timeout(payload), call._stream_stale_timeout


@pytest.mark.parametrize("model", ["claude-opus-5-5[1m]", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_external_process_stall_budget_is_the_local_ceiling_for_every_model(process_agent, model):
    """A ~325K-token request to a child process: no wall-clock kill on a buffered call and the 900s
    local ceiling on a stream, the same for every model (opus was cut at 240s while haiku waited)."""
    assert _budgets(process_agent, model) == (float("inf"), 900.0)


@pytest.mark.parametrize("seconds", [180, 1200])
def test_explicit_stale_timeout_wins_on_both_paths(process_agent, seconds):
    """An explicit value is the operator's deadline as-is; 180 is not mistaken for the unset default."""
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(f"providers:\n  {PROVIDER}:\n    stale_timeout_seconds: {seconds}\n")
    assert _budgets(process_agent, "claude-haiku-5-5") == (float(seconds), float(seconds))
