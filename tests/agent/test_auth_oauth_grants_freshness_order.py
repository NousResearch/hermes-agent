"""The fork-heal comparator must read the same mint clock as the seeding gate.

``_write_hermes_oauth_credentials`` stamps ``lastRefresh`` (a mint time) on the
Anthropic singleton because the access token's ``expiresAt`` is not a freshness
key. The heal in ``hermes_cli/auth_oauth_grants.py`` reads that singleton too;
it must carry ``lastRefresh`` through and rank on it when both sides have one.
"""
from __future__ import annotations

import json

from hermes_cli import auth_oauth_grants as g

NEWER_MINT = "2026-09-28T05:00:00Z"
OLDER_MINT = "2026-09-28T04:00:00Z"


def _singleton(tmp_path, **extra):
    p = tmp_path / ".anthropic_oauth.json"
    p.write_text(json.dumps({"accessToken": "NEWER-PAIR", "refreshToken": "rt-new",
                             "expiresAt": 1700000000000, **extra}))
    return p


def _older_row():
    # Later expiry, earlier mint: the case an expiry-ranked comparison gets wrong.
    return {"id": "r1", "auth_type": "oauth", "source": "hermes_pkce",
            "access_token": "OLDER-PAIR", "refresh_token": "rt-old",
            "expires_at_ms": 1800000000000, "last_refresh": OLDER_MINT}


def test_singleton_row_carries_last_refresh(tmp_path):
    row = g._singleton_as_row(_singleton(tmp_path, lastRefresh=NEWER_MINT))
    assert row["last_refresh"] == NEWER_MINT


def test_mint_time_decides_when_both_sides_have_one(tmp_path):
    single = g._singleton_as_row(_singleton(tmp_path, lastRefresh=NEWER_MINT))
    assert g._freshness_order(single, _older_row()) == 1
    assert g._freshness_order(_older_row(), single) == -1
    assert g._adopt_if_fresher(single, _older_row()) is None


def test_expiry_still_decides_without_a_mint_stamp_on_both_sides(tmp_path):
    single = g._singleton_as_row(_singleton(tmp_path))  # pre-stamp singleton on disk
    assert g._freshness_order(single, _older_row()) == -1


def test_sync_keeps_the_newer_singleton_and_rewrites_the_row(tmp_path):
    run = object.__new__(g._HealPass)
    run.summary = {"adopted": ["r1"]}
    run.root_singleton = tmp_path / ".anthropic_oauth.json"
    run.root_singleton_row = g._singleton_as_row(_singleton(tmp_path, lastRefresh=NEWER_MINT))
    run.r_rows = [_older_row()]
    run.root_changed = False
    run.sync_root_singleton_with_pkce_row()
    assert run.root_singleton_row["access_token"] == "NEWER-PAIR"
    assert run.r_rows[0]["access_token"] == "NEWER-PAIR"
    assert run.r_rows[0]["last_refresh"] == NEWER_MINT
    assert run.root_changed is True
