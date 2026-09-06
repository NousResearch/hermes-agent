"""Model pickers under a Nous free-tier identity.

A guest identity carrying inference shows one row, "Nous · free tier", with the single model
``nous/welcome``; the same guest with ``nous.guest: false`` shows no Nous row at all. The rule lives
in one helper (``_free_tier_nous_row``) and this file exercises it through the real row builders
against a temp ``HERMES_HOME`` with the network catalog fetch stubbed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.auth import _save_auth_store

GUEST_STATE = {
    "auth_method": "anonymous",
    "account_tier": "anonymous",
    "anon_token": "anon_t",
    "access_token": "aaa.bbb.ccc",
    "expires_at": "2030-01-01T00:00:00+00:00",
    "inference_base_url": "https://welcome-api.nousresearch.com/v1",
}


@pytest.fixture
def guest_home(monkeypatch, tmp_path):
    """Seed a guest identity as the only Nous state and keep every row builder offline."""
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.delenv("HERMES_FORCE_GUEST", raising=False)
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY", "LM_API_KEY", "LM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    _save_auth_store({"active_provider": "nous", "providers": {"nous": dict(GUEST_STATE)}})

    # No network from any lap: models.dev, the Portal catalog, and Ollama Cloud all stubbed.
    from agent import models_dev
    from hermes_cli import models as models_mod
    from hermes_cli import model_switch_providers as msp
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda *a, **k: ["anthropic/claude-x", "openai/gpt-y"])
    monkeypatch.setattr(models_mod, "fetch_ollama_cloud_models", lambda *a, **k: [])
    monkeypatch.setattr(msp, "_nous_picker_model_ids", lambda *a, **k: pytest.fail("guest must not fetch the Portal catalog"))
    return Path(os.environ["HERMES_HOME"])


def _write_config(monkeypatch, home: Path, text: str) -> None:
    (home / "config.yaml").write_text(text)
    from hermes_cli import config as cfg_mod
    for attr in ("_config_cache", "_cached_config"):
        if hasattr(cfg_mod, attr):
            monkeypatch.setattr(cfg_mod, attr, None, raising=False)


def _nous_rows(rows):
    return [r for r in rows if str(r.get("slug", "")).lower() == "nous"]


def _cli_nous_rows(config):
    from hermes_cli.main_provider_setup import _build_provider_picker_rows
    ordered, _ = _build_provider_picker_rows(config, "nous", {}, {})
    return [(key, label) for key, label, _members in ordered if key == "nous"]


def test_guest_identity_shows_free_tier_row_with_only_welcome_model(guest_home, monkeypatch):
    from hermes_cli.model_switch_providers import list_picker_providers
    rows = _nous_rows(list_picker_providers("nous", "", None, None, 50, "nous/welcome"))
    assert len(rows) == 1
    row = rows[0]
    assert "free tier" in row["name"]
    assert row["models"] == ["nous/welcome"]
    assert row["total_models"] == 1
    rendered = repr(row).lower()
    assert "guest" not in rendered and "anonymous" not in rendered

    # The `hermes model` provider picker applies the same rule from the same helper.
    cli_rows = _cli_nous_rows({})
    assert len(cli_rows) == 1
    assert "free tier" in cli_rows[0][1]
    assert "guest" not in cli_rows[0][1].lower() and "anonymous" not in cli_rows[0][1].lower()


def test_guest_identity_with_guest_off_hides_the_nous_row(guest_home, monkeypatch):
    _write_config(monkeypatch, guest_home, "nous:\n  guest: false\n")
    from hermes_cli import anon_auth
    assert anon_auth.has_guest() and not anon_auth.guest_enabled()

    from hermes_cli.model_switch_providers import list_picker_providers
    assert _nous_rows(list_picker_providers("nous", "", None, None, 50, "nous/welcome")) == []
    assert _cli_nous_rows({"nous": {"guest": False}}) == []
