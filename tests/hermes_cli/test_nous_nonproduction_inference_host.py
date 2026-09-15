"""Nous inference-host validation for operators pointed at a non-production Portal.

Portal-returned inference URLs go through a host allowlist before the user's bearer is sent
there. The Portal override (``HERMES_PORTAL_BASE_URL``) is a per-profile ``.env`` value, so it
is read through the profile secret scope; and when that trusted override names a Portal outside
the production allowlist, the Portal's own inference host is accepted as long as it is a Nous
host — instead of being refused and healed onto the production gateway, which 401s a token the
production Portal never issued. No environment is named in code: the rule is "operator-trusted
non-production Portal ⇒ any ``*.nousresearch.com`` inference host", nothing narrower.
"""
from __future__ import annotations

import pytest

from hermes_cli import auth_nous

PROD_PORTAL = "https://portal.nousresearch.com"
NONPROD_PORTAL = "https://portal.example-env.nousresearch.com"
PROD_INFERENCE = "https://inference-api.nousresearch.com/v1"
NONPROD_INFERENCE = "https://inference.example-env.nousresearch.com/v1"


def test_portal_env_override_is_read_through_the_profile_scope(monkeypatch):
    """Under multiplexing a routed turn resolves the Portal override from its own profile scope:
    the scoped value wins over the process env and a scoped miss stays a miss (the default
    profile's Portal is never borrowed). Unscoped (CLI / multiplex off) keeps reading the
    environment — the contract ``_nous_inference_env_override`` already has."""
    from agent import secret_scope as ss

    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", NONPROD_PORTAL)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    assert auth_nous._nous_portal_env_override() == NONPROD_PORTAL  # unscoped: environ

    ss.set_multiplex_active(True)
    try:
        token = ss.set_secret_scope({"HERMES_PORTAL_BASE_URL": "https://portal.other.example"})
        try:
            assert auth_nous._nous_portal_env_override() == "https://portal.other.example"
        finally:
            ss.reset_secret_scope(token)
        token = ss.set_secret_scope({})
        try:
            assert auth_nous._nous_portal_env_override() is None, "scoped miss must not read environ"
        finally:
            ss.reset_secret_scope(token)
    finally:
        ss.set_multiplex_active(False)


@pytest.mark.parametrize(
    "portal_override, url, expected",
    [
        # The bug: the operator's non-production Portal returns its own inference host; it survives.
        (NONPROD_PORTAL, NONPROD_INFERENCE, NONPROD_INFERENCE),
        # The protection: with no override, or a production override, only production hosts pass.
        (None, NONPROD_INFERENCE, None),
        (PROD_PORTAL, NONPROD_INFERENCE, None),
        # Production stays valid under any override; the rule widens, never narrows.
        (None, PROD_INFERENCE, PROD_INFERENCE),
        (NONPROD_PORTAL, PROD_INFERENCE, PROD_INFERENCE),
    ],
)
def test_inference_host_follows_the_operator_selected_portal(monkeypatch, portal_override, url, expected):
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    if portal_override is None:
        monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("HERMES_PORTAL_BASE_URL", portal_override)
    assert auth_nous._validate_nous_inference_url_from_network(url) == expected


def test_non_production_portal_never_admits_a_foreign_or_non_https_host(monkeypatch):
    """The widening is bounded to Nous-domain https hosts: a look-alike domain, a bare
    ``nousresearch.com`` suffix without the dot, and plain http are all still refused."""
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", NONPROD_PORTAL)
    for url in ("https://attacker.example/v1", "https://evilnousresearch.com/v1",
                "https://nousresearch.com.attacker.example/v1", "http://inference.example-env.nousresearch.com/v1",
                "https://.nousresearch.com/v1", "https://a..nousresearch.com/v1"):
        assert auth_nous._validate_nous_inference_url_from_network(url) is None, url
    # A malformed override (no parseable host) must not select the wider set either.
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "localhost:8000")
    assert auth_nous._validate_nous_inference_url_from_network(NONPROD_INFERENCE) is None


def test_stored_portal_alone_does_not_widen(monkeypatch):
    """A poisoned auth.json cannot select the wider set: the stored portal_base_url is a
    network-provenance value, only the operator override counts. ``_healed_nous_inference_url``
    therefore heals a foreign host to production unless the operator chose that environment."""
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    state = {"portal_base_url": NONPROD_PORTAL, "inference_base_url": NONPROD_INFERENCE, "client_id": "cid"}
    _portal, stored_inference, _effective, _ = auth_nous._nous_effective_routing(state)
    assert stored_inference == auth_nous.DEFAULT_NOUS_INFERENCE_URL
    refreshed = {"inference_base_url": NONPROD_INFERENCE}
    assert auth_nous._healed_nous_inference_url(refreshed) == auth_nous.DEFAULT_NOUS_INFERENCE_URL
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", NONPROD_PORTAL)
    assert auth_nous._healed_nous_inference_url(refreshed) == NONPROD_INFERENCE


def test_widening_follows_the_profile_scope_under_multiplex(monkeypatch):
    """Under multiplexing the decision uses the routed profile's own override: a secondary whose
    scope lacks it stays on the strict set even when the process env carries a non-production
    Portal (the default profile's), and a scope that carries one gets its environment's host."""
    from agent import secret_scope as ss

    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", NONPROD_PORTAL)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    ss.set_multiplex_active(True)
    try:
        token = ss.set_secret_scope({})
        try:
            assert auth_nous._validate_nous_inference_url_from_network(NONPROD_INFERENCE) is None
        finally:
            ss.reset_secret_scope(token)
        token = ss.set_secret_scope({"HERMES_PORTAL_BASE_URL": NONPROD_PORTAL})
        try:
            assert auth_nous._validate_nous_inference_url_from_network(NONPROD_INFERENCE) == NONPROD_INFERENCE
        finally:
            ss.reset_secret_scope(token)
    finally:
        ss.set_multiplex_active(False)
