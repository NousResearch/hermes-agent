"""Regression tests for claude-agent-sdk /model picker visibility (#65982).

Two symptoms, one root — the picker's authenticated-provider detection for a
self-authenticating runtime (``oauth_external``) only sees
``CLAUDE_CODE_OAUTH_TOKEN`` in the environment; a macOS-Keychain-only
``claude`` login is invisible to it:

- Token in env: the row appeared but carried a 1-model catalog, because the
  unified-pathway curated fallbacks in ``list_authenticated_providers()``
  looked up ``curated["claude-agent-sdk"]`` (which does not exist) without
  the ``_PROVIDER_CATALOG_DELEGATES`` (claude-agent-sdk -> anthropic) mapping.
- No token (Keychain-only login, lane serving turns fine): the row vanished
  from the interactive TUI picker entirely, because the current-provider
  fallback in ``build_models_payload()`` ran only under ``explicit_only``,
  and the TUI call site passes neither ``explicit_only`` nor
  ``include_unconfigured``.

Reported with repro and patches by 5tevebaker on PR #65982.
"""

import os
from unittest.mock import patch

from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import _PROVIDER_MODELS


def _sdk_ctx() -> ConfigContext:
    return ConfigContext(
        current_provider="claude-agent-sdk",
        current_model="claude-opus-5",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=[],
    )


def test_authenticated_row_serves_the_delegate_catalog():
    """Token in env -> the row carries anthropic's full curated catalog."""
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=[]),
        patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-dummy"}),
    ):
        rows = list_authenticated_providers(current_provider="claude-agent-sdk")

    row = next((r for r in rows if r["slug"] == "claude-agent-sdk"), None)
    assert row is not None
    curated = list(_PROVIDER_MODELS["anthropic"])
    assert len(curated) > 1  # sanity: the delegate catalog is a real list
    assert row["total_models"] == len(curated)
    assert set(row["models"]).issubset(set(curated))


def test_keychain_only_current_provider_stays_visible_in_tui_payload(monkeypatch):
    """No env token, row omitted upstream -> the TUI-shape payload keeps it."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers", return_value=[]
    ):
        payload = build_models_payload(
            _sdk_ctx(),
            probe_custom_providers=False,
            probe_current_custom_provider=True,
        )

    rows = [r for r in payload["providers"] if r["slug"] == "claude-agent-sdk"]
    assert len(rows) == 1
    row = rows[0]
    assert row["is_current"] is True
    assert row["authenticated"] is True  # self-authenticating, not "missing key"
    assert row["auth_type"] == "oauth_external"
    assert row["warning"] == ""
    # Saved model leads so the picker preselects it; delegate catalog follows.
    assert row["models"][0] == "claude-opus-5"
    assert len(row["models"]) > 1


def test_current_provider_fallback_does_not_duplicate_an_existing_row():
    """When the row is already present, the fallback append is a no-op."""
    existing = [
        {
            "slug": "claude-agent-sdk",
            "name": "Claude Agent SDK",
            "is_current": True,
            "is_user_defined": False,
            "models": ["claude-opus-5"],
            "total_models": 1,
            "source": "built-in",
        }
    ]
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=existing,
    ):
        payload = build_models_payload(
            _sdk_ctx(),
            probe_custom_providers=False,
            probe_current_custom_provider=True,
        )

    rows = [r for r in payload["providers"] if r["slug"] == "claude-agent-sdk"]
    assert len(rows) == 1
