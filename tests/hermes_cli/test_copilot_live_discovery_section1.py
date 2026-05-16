"""Regression tests for GitHub Copilot live-catalog discovery in /model.

These complement ``test_copilot_in_model_list.py``. Those tests mock
``fetch_models_dev`` to ``{}``, which makes ``list_authenticated_providers``
skip Section 1 (``PROVIDER_TO_MODELS_DEV`` loop) for ``copilot`` because no
``pdata`` is found, and Section 2 (``HERMES_OVERLAYS`` loop) emits the row
via its existing live-discovery branch.

In production, however, ``fetch_models_dev`` returns the real models.dev
manifest, which **does** include ``github-copilot``. Section 1 then emits
the Copilot row *first*, adds the slug to ``seen_slugs``, and Section 2's
live-discovery branch is skipped — so without the fix the picker shows the
stale ``_PROVIDER_MODELS["copilot"]`` snapshot and new GitHub-Copilot models
never appear.

These tests pin that real-data path: ``fetch_models_dev`` returns a manifest
that includes ``github-copilot``, and the picker is expected to reflect live
discovery instead of the curated stub.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


# Minimal models.dev payload containing a github-copilot entry. The real
# manifest has many more providers, but only this entry is load-bearing for
# the Section 1 dispatch — anything else just gets skipped for lack of creds.
_MDEV_WITH_COPILOT = {
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "env": ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"],
        "models": {},
    },
}


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_uses_live_catalog_when_models_dev_includes_copilot():
    """Section 1 must route Copilot through live discovery.

    Reproducer for the bug where ``list_authenticated_providers`` emitted
    Copilot from Section 1 with the stale curated catalog whenever models.dev
    knew about ``github-copilot`` — which is the production path.
    """
    live_models = [
        "gpt-5.4",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "brand-new-live-only-model",
    ]

    with patch(
        "agent.models_dev.fetch_models_dev",
        return_value=_MDEV_WITH_COPILOT,
    ), patch(
        "hermes_cli.models._resolve_copilot_catalog_api_key",
        return_value="gh-token",
    ), patch(
        "hermes_cli.models._fetch_github_models",
        return_value=live_models,
    ):
        providers = list_authenticated_providers(
            current_provider="openrouter", max_models=50
        )

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)
    assert copilot is not None, "copilot row missing from picker"
    # The live-only entry must appear — proves live discovery ran instead of
    # the static ``_PROVIDER_MODELS["copilot"]`` snapshot.
    assert "brand-new-live-only-model" in copilot["models"]
    assert copilot["models"] == live_models
    assert copilot["total_models"] == len(live_models)


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_falls_back_to_curated_when_live_unreachable():
    """``provider_model_ids`` returns ``None`` on offline / auth failures.

    When the live catalog can't be reached, the picker must still emit a
    Copilot row populated from the curated ``_PROVIDER_MODELS["copilot"]``
    snapshot — otherwise offline users would see an empty model list.
    """
    with patch(
        "agent.models_dev.fetch_models_dev",
        return_value=_MDEV_WITH_COPILOT,
    ), patch(
        "hermes_cli.models._resolve_copilot_catalog_api_key",
        return_value="gh-token",
    ), patch(
        "hermes_cli.models._fetch_github_models",
        return_value=None,
    ):
        providers = list_authenticated_providers(
            current_provider="openrouter", max_models=50
        )

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)
    assert copilot is not None, "copilot row missing from picker"
    # Curated stub must still be present for offline users.
    assert "gpt-5.4" in copilot["models"]
    assert "claude-sonnet-4.6" in copilot["models"]
    # And it must not be empty.
    assert copilot["total_models"] > 0
