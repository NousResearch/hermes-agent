"""Tests for Bitwarden env alias mapping in Hermes env loader."""

from __future__ import annotations

import os

from agent.secret_sources.bitwarden import FetchResult
from hermes_cli import env_loader
from agent.secret_sources import bitwarden as bw


def test_bitwarden_env_map_materializes_target_and_prunes_aliases(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    project_id: project-123\n"
        "    env_map:\n"
        "      DISCORD_BOT_TOKEN_MEOS_DEV: DISCORD_BOT_TOKEN\n"
        "    prune_env_prefixes:\n"
        "      - DISCORD_BOT_TOKEN_\n",
        encoding="utf-8",
    )

    def fake_apply_bitwarden_secrets(**kwargs):
        assert kwargs["project_id"] == "project-123"
        secrets = {
            "DISCORD_BOT_TOKEN_MEOS_DEV": "dev-token",
            "DISCORD_BOT_TOKEN_MEOS_ACADEMIC": "academic-token",
            "OPENAI_API_KEY": "openai-key",
        }
        for key, value in secrets.items():
            os.environ[key] = value
        return FetchResult(
            secrets=secrets,
            applied=list(secrets),
        )

    monkeypatch.setattr(bw, "apply_bitwarden_secrets", fake_apply_bitwarden_secrets)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "access")
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_BOT_TOKEN_MEOS_DEV", raising=False)
    monkeypatch.delenv("DISCORD_BOT_TOKEN_MEOS_ACADEMIC", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_loader.reset_secret_source_cache()
    env_loader._SECRET_SOURCES.clear()

    env_loader._apply_external_secret_sources(home)

    assert env_loader.os.environ["DISCORD_BOT_TOKEN"] == "dev-token"
    assert env_loader.os.environ["OPENAI_API_KEY"] == "openai-key"
    assert "DISCORD_BOT_TOKEN_MEOS_DEV" not in env_loader.os.environ
    assert "DISCORD_BOT_TOKEN_MEOS_ACADEMIC" not in env_loader.os.environ
    assert env_loader.get_secret_source("DISCORD_BOT_TOKEN") == "bitwarden"
    assert env_loader.get_secret_source("OPENAI_API_KEY") == "bitwarden"
    assert env_loader.get_secret_source("DISCORD_BOT_TOKEN_MEOS_DEV") is None


def test_bitwarden_env_map_respects_existing_target_without_override(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    project_id: project-123\n"
        "    override_existing: false\n"
        "    env_map:\n"
        "      DISCORD_BOT_TOKEN_MEOS_DEV: DISCORD_BOT_TOKEN\n",
        encoding="utf-8",
    )

    def fake_apply_bitwarden_secrets(**kwargs):
        os.environ["DISCORD_BOT_TOKEN_MEOS_DEV"] = "dev-token"
        return FetchResult(
            secrets={"DISCORD_BOT_TOKEN_MEOS_DEV": "dev-token"},
            applied=["DISCORD_BOT_TOKEN_MEOS_DEV"],
        )

    monkeypatch.setattr(bw, "apply_bitwarden_secrets", fake_apply_bitwarden_secrets)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "access")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "existing-token")
    env_loader.reset_secret_source_cache()
    env_loader._SECRET_SOURCES.clear()

    env_loader._apply_external_secret_sources(home)

    assert env_loader.os.environ["DISCORD_BOT_TOKEN"] == "existing-token"
    assert "DISCORD_BOT_TOKEN_MEOS_DEV" not in env_loader.os.environ
    assert env_loader.get_secret_source("DISCORD_BOT_TOKEN_MEOS_DEV") is None
    assert env_loader.get_secret_source("DISCORD_BOT_TOKEN") is None
