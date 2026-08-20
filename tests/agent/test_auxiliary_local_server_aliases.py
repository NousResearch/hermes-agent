"""Regression test for fallback router local-server alias drift (#74280).

Issue #74280 reports that ``fallback_providers`` entries with
``provider: ollama`` (and other local servers like ``vllm``,
``llamacpp``) can never activate because ``agent.auxiliary_client`` keeps
its own copy of ``_PROVIDER_ALIASES`` that is missing the local-server
aliases present in the canonical map in ``hermes_cli/auth.py``.

The fallback router normalizes via ``_normalize_aux_provider`` →
``_PROVIDER_ALIASES.get(...)``. With the alias absent, ``ollama`` stays
as ``ollama``, fails to match any provider branch, and ``resolve_provider_client``
returns ``(None, None)`` → ``provider not configured``.

These tests pin the alias map and the normalization + provider-resolution
end-to-end for the canonical local-server names.
"""
from __future__ import annotations

import pytest


# Aliases that MUST exist in agent.auxiliary_client._PROVIDER_ALIASES to
# keep the fallback router in sync with hermes_cli/auth.py.
LOCAL_SERVER_ALIASES = {
    "ollama": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
    "llama-cpp": "custom",
}


def test_local_server_aliases_present():
    """Every local-server alias must map to 'custom'."""
    from agent.auxiliary_client import _PROVIDER_ALIASES
    for alias, target in LOCAL_SERVER_ALIASES.items():
        assert _PROVIDER_ALIASES.get(alias) == target, (
            f"_PROVIDER_ALIASES is missing {alias!r} → {target!r} "
            f"(see #74280)"
        )


def test_ollama_normalizes_to_custom():
    """_normalize_aux_provider('ollama') must return 'custom'."""
    from agent.auxiliary_client import _normalize_aux_provider
    assert _normalize_aux_provider("ollama") == "custom"


@pytest.mark.parametrize("alias", ["ollama", "vllm", "llamacpp", "llama.cpp", "llama-cpp"])
def test_local_server_aliases_normalize_to_custom_in_canonical_map(alias):
    """Cross-check: hermes_cli.auth.resolve_provider maps the same alias
    to the same target.

    Source of truth for the alias is ``hermes_cli.auth.resolve_provider``;
    ``agent.auxiliary_client`` must agree.
    """
    from agent.auxiliary_client import _PROVIDER_ALIASES
    from hermes_cli.auth import resolve_provider

    # Canonical target via hermes_cli.auth (the live resolver).
    # resolve_provider accepts an alias by passing it as `requested`.
    canonical = resolve_provider(requested=alias, explicit_base_url="http://localhost:11434/v1")
    # The resolver may return 'custom' or 'ollama-cloud' / etc; we only care
    # that local-server aliases land in 'custom' (the route that honors
    # explicit_base_url).
    assert canonical == "custom", (
        f"hermes_cli.auth.resolve_provider({alias!r}) returned {canonical!r}; "
        f"expected 'custom' (see #74280)"
    )
    # And agent.auxiliary_client must agree.
    assert _PROVIDER_ALIASES.get(alias) == "custom"
