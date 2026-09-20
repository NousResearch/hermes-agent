"""Regression tests for auxiliary client invalidation after dotenv credential reload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from gateway import run as gateway_run
import agent.auxiliary_client as auxiliary_client


def test_reload_runtime_env_clears_auxiliary_cache_when_credentials_change(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    fingerprints = iter(("old", "new"))
    monkeypatch.setattr(auxiliary_client, "credential_env_fingerprint", lambda: next(fingerprints))
    clear = patch.object(auxiliary_client, "clear_cached_clients")
    with clear as clear_cache, patch.object(gateway_run, "load_hermes_dotenv"):
        gateway_run._reload_runtime_env_preserving_config_authority()
    clear_cache.assert_called_once_with()


def test_reload_runtime_env_preserves_auxiliary_cache_when_credentials_unchanged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(auxiliary_client, "credential_env_fingerprint", lambda: "same")
    with patch.object(auxiliary_client, "clear_cached_clients") as clear_cache:
        gateway_run._reload_runtime_env_preserving_config_authority()
    clear_cache.assert_not_called()
