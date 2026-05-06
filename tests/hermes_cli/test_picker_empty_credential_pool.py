"""Regression test: section 1 of list_authenticated_providers must require
real credentials, not a bare key in the auth store's ``credential_pool`` map.

Background
----------
The auth store's ``credential_pool`` dict is keyed by provider id, and a key
can exist with an *empty list* value (e.g. after credentials were rotated
out, or after early seeding of an entry that was never populated).  The old
section-1 check::

    if store and hermes_id in store.get("credential_pool", {}):
        has_creds = True

returned True for empty entries, surfacing providers in the /model picker
that the user never authenticated to (huggingface, opencode-go,
minimax(-cn), zai, ...).  Selecting one routes every turn at a dead
endpoint and 401s.

A sister fix (commit 46072425f) plugged the same class of bug in sections
2 and 2b but missed section 1 — this is the focused fix.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


class _StubPool:
    """Minimal stand-in for ``agent.credential_pool.CredentialPool``."""

    def __init__(self, has_creds: bool = False) -> None:
        self._has_creds = has_creds

    def has_credentials(self) -> bool:  # pragma: no cover - trivial
        return self._has_creds


def test_section1_skips_provider_with_empty_credential_pool_list(monkeypatch):
    """huggingface entry with ``credential_pool['huggingface'] = []`` must NOT appear.

    This is the exact pathological case observed in the wild — the auth
    store contains the key but the value is an empty list, meaning no
    credentials are actually available.  The picker must skip the row.
    """
    # Make sure no env-var path can let huggingface in via the front door.
    for ev in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(ev, raising=False)

    fake_store = {"credential_pool": {"huggingface": []}}

    def _fake_load_pool(slug, *args, **kwargs):
        # The bug-trigger: store key exists, but no real credentials behind it.
        return _StubPool(has_creds=False)

    with patch("hermes_cli.auth._load_auth_store", return_value=fake_store), \
         patch("agent.credential_pool.load_pool", side_effect=_fake_load_pool):
        providers = list_authenticated_providers(current_provider="")

    slugs = [p["slug"] for p in providers]
    assert "huggingface" not in slugs, (
        "huggingface appeared in /model picker despite having no real "
        f"credentials; pool-key-only check leaked through. Got: {slugs}"
    )


def test_section1_includes_provider_when_pool_has_real_credentials(monkeypatch):
    """Sanity: when the credential pool reports real creds, the row appears."""
    for ev in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(ev, raising=False)

    fake_store = {"credential_pool": {"huggingface": []}}

    def _fake_load_pool(slug, *args, **kwargs):
        if slug == "huggingface":
            return _StubPool(has_creds=True)
        return _StubPool(has_creds=False)

    with patch("hermes_cli.auth._load_auth_store", return_value=fake_store), \
         patch("agent.credential_pool.load_pool", side_effect=_fake_load_pool):
        providers = list_authenticated_providers(current_provider="")

    slugs = [p["slug"] for p in providers]
    assert "huggingface" in slugs, (
        "huggingface should appear when the credential pool reports real "
        f"credentials; got: {slugs}"
    )
