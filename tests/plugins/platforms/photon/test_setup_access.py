"""Tests for `hermes photon setup`'s access auto-configuration.

`_autoconfigure_access` allowlists the operator and points the cron home
channel at their DM, writing to the per-test ~/.hermes/.env (the hermetic
HERMES_HOME fixture isolates this). It must fill only unset keys so a re-run
never clobbers a hand-tuned allowlist.
"""
from __future__ import annotations

import pytest

from hermes_cli.config import get_env_value, save_env_value
from plugins.platforms.photon.adapter import _env_enablement
from plugins.platforms.photon import cli


def test_autoconfigure_access_fills_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_HOME_CHANNEL", raising=False)

    cli._autoconfigure_access("+15551234567")

    assert get_env_value("PHOTON_ALLOWED_USERS") == "+15551234567"
    assert get_env_value("PHOTON_HOME_CHANNEL") == "+15551234567"


def test_autoconfigure_access_preserves_existing_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_HOME_CHANNEL", raising=False)
    # A hand-tuned allowlist already in place must survive a setup re-run.
    save_env_value("PHOTON_ALLOWED_USERS", "+19998887777,+15551112222")

    cli._autoconfigure_access("+15551234567")

    assert get_env_value("PHOTON_ALLOWED_USERS") == "+19998887777,+15551112222"
    # The still-unset home channel is filled.
    assert get_env_value("PHOTON_HOME_CHANNEL") == "+15551234567"


def test_env_enablement_seeds_home_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "project_123")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret_123")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL", "+15551234567")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL_NAME", "Primary DM")

    seed = _env_enablement()

    assert seed is not None
    assert seed["home_channel"] == {
        "chat_id": "+15551234567",
        "name": "Primary DM",
    }


def test_env_enablement_home_channel_defaults_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "project_123")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret_123")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL", "+15551234567")
    monkeypatch.delenv("PHOTON_HOME_CHANNEL_NAME", raising=False)

    seed = _env_enablement()

    assert seed is not None
    assert seed["home_channel"] == {
        "chat_id": "+15551234567",
        "name": "Home",
    }
