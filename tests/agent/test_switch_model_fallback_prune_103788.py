"""Regression tests for #103788: switch_model()'s fallback prune must drop
entries by backend ENDPOINT identity, not raw provider-string equality.

Bare `custom` is a category, not a deployment — `custom:antigravity` and a
local custom proxy at 127.0.0.1:8001 are different backends. The old prune
compared ``entry.provider in {old_norm, new_norm}`` verbatim, so any switch
whose old or new provider label was bare ``custom`` wiped every bare-custom
fallback entry (chain emptied -> eager-failover never entered -> a 429 on the
new primary retried to failure instead of falling back). The prune now reuses
agent.backend_identity.same_endpoint (FailureScope.ENDPOINT): explicit
base_urls decide; a shared provider label only implies the same endpoint when
URLs are absent.
"""

import agent.agent_runtime_helpers as arh


class _Agent:
    pass


def _run_finish_switch(*, chain, old_provider, old_base_url, old_model,
                       new_provider, new_base_url, new_model):
    """Drive _finish_switch on a bare agent with the given pre-switch identity.

    Falls back to the legacy 4-arg signature on unfixed main (the RED run) so
    the test demonstrates the behavioral bug (chain emptied), not a TypeError.
    """
    a = _Agent()
    a._fallback_activated = True
    a._provider_fallback_active = True
    a._provider_fallback_route = ("old-model", old_provider)
    a._fallback_index = 1
    a._fallback_chain = [dict(e) for e in chain]
    try:
        arh._finish_switch(
            a, new_provider,
            (old_provider or "").strip().lower(), (new_provider or "").strip().lower(),
            old_provider=old_provider, old_model=old_model, old_base_url=old_base_url,
            new_model=new_model, new_base_url=new_base_url,
        )
    except TypeError:
        # Pre-fix signature: _finish_switch(agent, new_provider, old_norm, new_norm)
        arh._finish_switch(
            a, new_provider,
            (old_provider or "").strip().lower(), (new_provider or "").strip().lower(),
        )
    return a


def test_bare_custom_chain_survives_custom_to_named_custom_switch():
    """#103788 repro: 4 local bare-custom fallback entries must survive a
    switch from a bare-custom primary (antigravity host) to custom:antigravity."""
    chain = [
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"},
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8002/v1"},
        {"provider": "custom", "model": "glm-5.3-flash", "base_url": "http://127.0.0.1:8001/v1"},
        {"provider": "custom", "model": "glm-5.3-flash", "base_url": "http://127.0.0.1:8002/v1"},
    ]
    a = _run_finish_switch(
        chain=chain,
        old_provider="custom", old_base_url="http://antigravity-host/v1",
        old_model="gemini-3.8-flash-high",
        new_provider="custom:antigravity", new_base_url="http://antigravity-host/v1",
        new_model="claude-opus-4.6",
    )
    assert len(a._fallback_chain) == 4, (
        "bare-custom entries at different endpoints must survive the prune; "
        f"got {len(a._fallback_chain)}"
    )
    assert a._fallback_model == chain[0]


def test_entry_at_old_primary_endpoint_is_pruned():
    """The prune still works for its original purpose: an entry resolving to the
    old primary's endpoint (same explicit base_url) is dropped on switch-away."""
    old_url = "http://antigravity-host/v1"
    chain = [
        {"provider": "custom", "model": "gemini-3.8-flash-high", "base_url": old_url},
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"},
    ]
    a = _run_finish_switch(
        chain=chain,
        old_provider="custom", old_base_url=old_url, old_model="gemini-3.8-flash-high",
        new_provider="custom:codebuddy", new_base_url="http://codebuddy-host/v1",
        new_model="claude-opus-4.6",
    )
    assert len(a._fallback_chain) == 1
    assert a._fallback_chain[0]["base_url"] == "http://127.0.0.1:8001/v1"


def test_named_provider_entries_pruned_by_label_when_url_absent():
    """Named providers without explicit URLs still prune by provider label
    (shared label = shared default endpoint) — deepseek chain entry must not
    survive a deliberate switch away from deepseek."""
    chain = [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "anthropic", "model": "claude-sonnet-4.6"},
    ]
    a = _run_finish_switch(
        chain=chain,
        old_provider="deepseek", old_base_url="", old_model="deepseek-v4-flash",
        new_provider="openai", new_base_url="", new_model="gpt-5.2",
    )
    assert [e["provider"] for e in a._fallback_chain] == ["anthropic"]


def test_same_provider_switch_does_not_prune():
    """old_norm == new_norm (e.g. credential refresh re-selects the same
    provider): the prune is skipped entirely — unchanged behavior."""
    chain = [{"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"}]
    a = _run_finish_switch(
        chain=chain,
        old_provider="custom", old_base_url="http://antigravity-host/v1",
        old_model="gemini-3.8-flash-high",
        new_provider="custom", new_base_url="http://antigravity-host/v1",
        new_model="gemini-3.8-flash-high",
    )
    assert len(a._fallback_chain) == 1
