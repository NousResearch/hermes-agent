"""The "no credentials" error for an explicit provider names a remedy that can actually work.

Deriving the env var from the provider id (``f"{id.upper()}_API_KEY"``) invents names nothing reads:
``MINIMAX-OAUTH_API_KEY`` for OAuth ids (#114405, #78996) and ``ALIBABA_API_KEY`` where the registry
reads ``DASHSCOPE_API_KEY``. Both the auxiliary ladder and main-agent init share one helper.
"""
import pytest
import yaml

from agent.auxiliary_unavailable import missing_provider_credentials_message
from hermes_cli.auth import PROVIDER_REGISTRY


@pytest.mark.parametrize("provider, expected, forbidden", [
    ("minimax-oauth", "hermes auth add minimax-oauth", "MINIMAX-OAUTH_API_KEY"),
    ("alibaba", "Set the DASHSCOPE_API_KEY environment variable", "ALIBABA_API_KEY"),
])
def test_aux_ladder_names_registry_remedy_for_explicit_provider(tmp_path, monkeypatch, provider, expected, forbidden):
    """Real call_llm → ladder with the compression provider pinned and no credentials anywhere."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("MINIMAX_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": provider, "default": "test-model"},
        "auxiliary": {"compression": {"provider": provider, "model": "test-model"}},
    }), encoding="utf-8")
    from agent.auxiliary_client import call_llm
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    with pytest.raises(AuxiliaryClientUnavailable) as excinfo:
        call_llm(task="compression", messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    assert expected in str(excinfo.value)
    assert forbidden not in str(excinfo.value)


def test_main_init_shares_helper_and_no_registry_provider_gets_an_invented_env_var(tmp_path, monkeypatch):
    """agent_init raises the same text as the helper; no registry id yields a made-up ``<ID>_API_KEY``."""
    from types import SimpleNamespace

    from agent.agent_init import _routed_client_kwargs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(provider="minimax-oauth", model="m", base_url=None, api_key=None,
                            _fallback_activated=False, _explicit_provider="minimax-oauth")
    with pytest.raises(RuntimeError, match=r"hermes auth add minimax-oauth"):
        _routed_client_kwargs(agent, None, 60)

    for pid, pconfig in PROVIDER_REGISTRY.items():
        invented = f"{pid.upper()}_API_KEY"
        message = missing_provider_credentials_message(pid)
        if invented in message:
            assert invented in pconfig.api_key_env_vars, (pid, message)
        if not pconfig.api_key_env_vars:
            assert "_API_KEY environment variable" not in message, (pid, message)
