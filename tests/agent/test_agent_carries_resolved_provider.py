"""Task 9 (cpf-zkw.9): the agent carries the authoritative ResolvedProvider.

Resolution provenance (base_url_source / key_source) is computed once and rides
along on the constructed agent (value-object carry, plan §2/§4), so logging and
`hermes doctor` can explain *why* an endpoint/key was chosen — and so Task 10
can drop the legacy dict shim and read the typed object directly. The carry is
optional/backward-compatible: callers still passing individual base_url/api_key/
provider fields leave it unset and the provenance attrs read "".
"""

from __future__ import annotations

import pytest

from hermes_cli.provider_resolution import ResolvedProvider


@pytest.fixture(autouse=True)
def _reset_aux_global_state():
    """Constructing an AIAgent exercises the auxiliary client and can leave
    process-global aux state behind (e.g. the unhealthy-provider cache). Reset
    it around each test so this file can't pollute later tests that assume a
    clean aux client (e.g. tests/run_agent/test_provider_parity.py)."""
    import agent.auxiliary_client as ax
    import hermes_cli.runtime_provider as rp

    def _clean():
        try:
            ax._reset_aux_unhealthy_cache()
        except Exception:
            pass
        try:
            ax.clear_runtime_main()
        except Exception:
            pass
        rp.clear_resolution_memo()

    _clean()
    yield
    _clean()


def _make_resolved() -> ResolvedProvider:
    return ResolvedProvider(
        provider="custom",
        requested_provider="custom",
        api_mode="chat_completions",
        base_url="http://localhost:1234/v1",
        api_key="sk-local",
        base_url_source="config.base_url",
        key_source="env:DEEPSEEK_API_KEY",
    )


def test_agent_carries_resolved_provider_and_provenance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from run_agent import AIAgent

    resolved = _make_resolved()
    agent = AIAgent(
        model="my-model",
        provider="custom",
        api_key="sk-local",
        base_url="http://localhost:1234/v1",
        resolved_provider=resolved,
    )
    assert agent._resolved_provider is resolved
    assert agent.base_url_source == "config.base_url"
    assert agent.key_source == "env:DEEPSEEK_API_KEY"


def test_agent_without_resolved_provider_has_blank_provenance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from run_agent import AIAgent

    agent = AIAgent(
        model="my-model",
        provider="custom",
        api_key="sk-local",
        base_url="http://localhost:1234/v1",
    )
    assert agent._resolved_provider is None
    assert agent.base_url_source == ""
    assert agent.key_source == ""
