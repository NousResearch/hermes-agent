"""Tests for `hermes photon setup`'s access auto-configuration.

`_autoconfigure_access` allowlists the operator and points the cron home
channel at their DM, writing to the per-test ~/.hermes/.env (the hermetic
HERMES_HOME fixture isolates this). It must fill only unset keys so a re-run
never clobbers a hand-tuned allowlist.
"""
from __future__ import annotations

import pytest

from hermes_cli.config import get_env_value, save_env_value
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
