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
