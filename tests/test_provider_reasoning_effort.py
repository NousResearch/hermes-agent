"""Provider-scoped reasoning effort: per-model > providers entry > global.

Regression for #119681: ``reasoning_effort`` declared on a bare named provider
(``providers:`` entry) was silently dropped — resolution only looked at the model
string, so ``delegate_task`` children pinned to that provider and desktop sessions
built from it fell back to the global level.
"""
from hermes_constants import (
    resolve_reasoning_config,
    resolve_specific_reasoning_config,
)


def _cfg():
    return {
        "agent": {"reasoning_effort": "medium"},
        "providers": {"myProv": {"reasoning_effort": "high"}},
    }


def test_provider_entry_supersedes_global_but_not_per_model():
    assert resolve_reasoning_config(_cfg(), "some-model", "myProv") == {
        "enabled": True, "effort": "high"}
    # ``custom:`` runtime identities resolve to the same entry.
    assert resolve_reasoning_config(_cfg(), "some-model", "custom:myProv") == {
        "enabled": True, "effort": "high"}
    # No provider -> global still applies.
    assert resolve_reasoning_config(_cfg(), "some-model") == {
        "enabled": True, "effort": "medium"}
    # Per-model override stays most specific.
    cfg = _cfg()
    cfg["agent"] = {**cfg["agent"], "reasoning_overrides": {"some-model": "low"}}
    assert resolve_reasoning_config(cfg, "some-model", "myProv") == {
        "enabled": True, "effort": "low"}


def test_specific_resolution_leaves_unpinned_inheritance_alone():
    # Pinned route: provider entry is found, so a delegation child adopts it.
    assert resolve_specific_reasoning_config(_cfg(), "some-model", "myProv") == {
        "enabled": True, "effort": "high"}
    # Unpinned route: nothing model/provider-specific -> None, caller keeps inherit.
    assert resolve_specific_reasoning_config(_cfg(), "some-model", None) is None
    assert resolve_specific_reasoning_config(
        {"agent": {"reasoning_effort": "medium"}}, "some-model", "unknown") is None
