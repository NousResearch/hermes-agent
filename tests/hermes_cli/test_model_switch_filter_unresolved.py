"""Regression tests for picker filtering of unresolved providers.

Bug — ``list_authenticated_providers()`` would emit ``/model`` picker rows for
    providers that ``PROVIDER_TO_MODELS_DEV`` knows about but that the runtime
    ``resolve_provider()`` rejects as ``Unknown provider``. The reporter set
    ``MISTRAL_API_KEY`` and saw a ``Mistral (72)`` row in the Telegram picker;
    selecting any model under it produced
    ``Could not resolve credentials for provider 'Mistral': Unknown provider 'mistral'``
    (#57503).

The fix adds a resolve-gate in section 1 of ``list_authenticated_providers``:
if a slug has no ``PROVIDER_REGISTRY`` entry, skip it. The picker now only
shows providers that can actually be selected at runtime.

These tests pin that behavior in two directions:

1. With ``MISTRAL_API_KEY`` set, ``mistral`` MUST NOT appear in the picker
   (it has no ``PROVIDER_REGISTRY`` entry at the time of this fix).
2. A provider that IS in ``PROVIDER_REGISTRY`` (e.g. ``deepseek``) and has
   its API key set MUST still appear, so the fix does not regress existing
   picker coverage.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


def _reload():
    """Force a fresh import of model_switch so module-level caches reset.

    The picker caches ``PROVIDER_REGISTRY`` at import time. Each test must
    see a fresh module to avoid bleeding env vars between cases.
    """
    for mod_name in [m for m in list(sys.modules) if m.startswith("hermes_cli.model_switch")]:
        sys.modules.pop(mod_name, None)
    return importlib.import_module("hermes_cli.model_switch")


def _slug(rows):
    """Return the set of provider slugs from picker rows."""
    return {row.get("slug") for row in rows}


def test_mistral_filtered_when_unregistered_but_api_key_set(monkeypatch):
    """#57503 — MISTRAL_API_KEY set, mistral has no PROVIDER_REGISTRY entry.

    Picker must NOT emit a 'mistral' row. Users with a working
    custom_providers entry for Mistral no longer see the duplicate
    broken-from-models.dev entry cluttering the list.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key-shouldnt-crash")

    model_switch = _reload()
    rows = model_switch.list_authenticated_providers(max_models=5)

    slugs = _slug(rows)
    assert "mistral" not in slugs, (
        f"mistral leaked into /model picker despite being unregistered: {sorted(slugs)}"
    )


def test_resolveable_provider_still_appears(monkeypatch):
    """Sanity — a registered provider with a key must still surface.

    Guards against the new resolve-gate being too broad (e.g. nuking the
    happy path of every other provider while filtering mistral).
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key-shouldnt-crash")

    model_switch = _reload()
    rows = model_switch.list_authenticated_providers(max_models=5)

    slugs = _slug(rows)
    assert "deepseek" in slugs, (
        "deepseek disappeared from /model picker — resolve-gate is too broad. "
        f"slugs seen: {sorted(slugs)}"
    )


def test_resolve_gate_skips_models_dev_only_provider_without_creds(monkeypatch):
    """A models-dev provider with no key and no PROVIDER_REGISTRY entry has
    no way to surface in the picker — verify we never emit such a row.

    Keeps the fix idempotent: the previous ``if not has_creds: continue``
    gate already handled the no-credential case; we add a second gate on
    top that catches the WITH-credentials-but-no-registry case.
    """
    # No MISTRAL_API_KEY, no other mistral-shaped env vars set
    for k in ("MISTRAL_API_KEY", "MISTRAL_BASE_URL"):
        monkeypatch.delenv(k, raising=False)

    model_switch = _reload()
    rows = model_switch.list_authenticated_providers(max_models=5)

    slugs = _slug(rows)
    assert "mistral" not in slugs


def test_picker_skips_pconfig_none_does_not_break_other_section1_providers(monkeypatch, tmp_path):
    """Coverage guard — providers that share a models.dev id with their
    canonical Hermes slug (e.g. ``gemini``, ``xai``, ``cohere``) must keep
    appearing. These ARE in ``PROVIDER_REGISTRY`` so the new gate should
    not touch them.
    """
    # Pick a registry-backed provider that exists in models.dev and is
    # unlikely to be subject to the env-var duck-typing the picker uses
    # for unnamespaced keys.
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key-shouldnt-crash")

    model_switch = _reload()
    rows = model_switch.list_authenticated_providers(max_models=5)

    slugs = _slug(rows)
    assert "xai" in slugs, (
        "xai disappeared — the resolve-gate ended up filtering registry-backed "
        "providers too. slugs seen: " + repr(sorted(slugs))
    )
