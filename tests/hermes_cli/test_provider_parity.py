"""End-to-end provider parity contract: the desktop Providers tabs must show
the SAME provider universe as ``hermes model`` (the CLI/TUI picker).

This is the single load-bearing invariant of the unified provider catalog:

    keys(/api/env provider rows) ∪ ids(/api/providers/oauth) ⊇ CANONICAL_PROVIDERS

i.e. every provider the CLI picker offers is configurable from the desktop app,
on one of the two Providers sub-tabs (API keys or Accounts). It is asserted as
an invariant against the real FastAPI endpoints (not a snapshot / count), so it
can never silently drift again when a provider plugin is added.
"""

from fastapi.testclient import TestClient

from hermes_cli.models import CANONICAL_PROVIDERS
from hermes_cli.provider_catalog import provider_catalog
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

# `custom` is the bring-your-own-endpoint pseudo-provider configured inline via
# the model picker's local-endpoint flow, not a fixed credential card. It is in
# the CLI picker's universe but intentionally has no dedicated Providers-tab
# card. Exempt it from the union check.
_EXEMPT = {"custom"}

# Providers that legitimately offer BOTH auth methods and so intentionally
# appear on both desktop tabs (an API-key card AND an account sign-in card).
# Anthropic supports a direct API key (Keys tab) and a subscription OAuth /
# Claude Code login (Accounts tab); surfacing both is correct, not a bug.
_DUAL_TAB = {"anthropic"}


def _keys_tab_providers() -> set[str]:
    """Provider slugs that have at least one card on the desktop API-keys tab."""
    data = client.get("/api/env", headers=HEADERS).json()
    return {
        info.get("provider")
        for info in data.values()
        if info.get("category") == "provider" and info.get("provider")
    }


def _accounts_tab_providers() -> set[str]:
    """Provider slugs offered on the desktop Accounts tab."""
    data = client.get("/api/providers/oauth", headers=HEADERS).json()
    return {p["id"] for p in data["providers"]}


def test_every_hermes_model_provider_is_configurable_in_desktop():
    """PARITY CONTRACT: GUI (keys ∪ accounts) ⊇ `hermes model` universe."""
    gui = _keys_tab_providers() | _accounts_tab_providers()
    missing = [
        e.slug
        for e in CANONICAL_PROVIDERS
        if e.slug not in _EXEMPT and e.slug not in gui
    ]
    assert not missing, (
        "providers shown in `hermes model` but not configurable in the desktop "
        f"Providers tabs: {missing}"
    )


def test_each_provider_lands_on_the_tab_its_auth_type_dictates():
    """A keys-tab provider must surface under /api/env; an accounts-tab provider
    under /api/providers/oauth. Cross-checks the catalog's tab routing against
    where each provider actually renders.
    """
    keys = _keys_tab_providers()
    accounts = _accounts_tab_providers()
    for d in provider_catalog():
        if d.slug in _EXEMPT:
            continue
        if d.tab == "keys" and d.api_key_env_vars:
            assert d.slug in keys, f"{d.slug} (keys tab) missing from /api/env"
        elif d.tab == "accounts":
            assert d.slug in accounts, f"{d.slug} (accounts tab) missing from /api/providers/oauth"


def test_no_provider_appears_on_both_tabs():
    """A provider should be configured exactly one way — not duplicated across
    both tabs (which would confuse users about where to put credentials).

    Exception: genuinely dual-auth providers (see ``_DUAL_TAB``) intentionally
    appear on both tabs.
    """
    overlap = (_keys_tab_providers() & _accounts_tab_providers()) - _EXEMPT - _DUAL_TAB
    assert not overlap, f"providers appearing on BOTH desktop tabs: {sorted(overlap)}"
