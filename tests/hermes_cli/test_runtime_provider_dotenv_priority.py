"""Tests for #20591 follow-up: prefer ~/.hermes/.env on credential-resolve paths.

Covers ``hermes_cli.runtime_provider._resolve_azure_foundry_runtime``: the
runtime resolver must prefer the value in ``~/.hermes/.env`` over a stale
``AZURE_FOUNDRY_API_KEY`` inherited from the parent shell, because the
returned key is what's sent upstream to Azure Foundry on every request.
"""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and clear Azure Foundry env vars."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_FOUNDRY_BASE_URL",
        "AZURE_FOUNDRY_API_MODE",
    ]:
        monkeypatch.delenv(key, raising=False)
    return home


def _write_env_file(home: Path, **kwargs) -> None:
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n")


def _resolve_azure_foundry(monkeypatch):
    """Drive the resolver with a synthetic model_cfg."""
    from hermes_cli import runtime_provider

    model_cfg = {
        "provider": "azure-foundry",
        "base_url": "https://example.azure.com",
        "api_mode": "chat_completions",
        "default": "azure-foundry/gpt-4o",
    }
    return runtime_provider._resolve_azure_foundry_runtime(
        requested_provider="azure-foundry",
        model_cfg=model_cfg,
        explicit_api_key=None,
        explicit_base_url=None,
        target_model="azure-foundry/gpt-4o",
    )


class TestAzureFoundryDotEnvPriority:
    """The Azure Foundry resolver must prefer .env over os.environ for the key.

    This is the load-bearing behaviour for the fix: a freshly rotated
    AZURE_FOUNDRY_API_KEY in ~/.hermes/.env must immediately win over a
    stale value the user exported in their shell rc file.
    """

    def test_dotenv_value_wins_over_stale_environ(
        self, isolated_hermes_home, monkeypatch
    ):
        _write_env_file(
            isolated_hermes_home, AZURE_FOUNDRY_API_KEY="sk-fresh-from-dotenv"
        )
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-stale-from-shell")

        result = _resolve_azure_foundry(monkeypatch)

        assert result["api_key"] == "sk-fresh-from-dotenv"

    def test_environ_used_when_dotenv_missing(
        self, isolated_hermes_home, monkeypatch
    ):
        # No .env entry — fall back to os.environ (matches old behaviour).
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-environ-only")

        result = _resolve_azure_foundry(monkeypatch)

        assert result["api_key"] == "sk-environ-only"

    def test_dotenv_used_when_environ_missing(
        self, isolated_hermes_home, monkeypatch
    ):
        _write_env_file(
            isolated_hermes_home, AZURE_FOUNDRY_API_KEY="sk-dotenv-only"
        )

        result = _resolve_azure_foundry(monkeypatch)

        assert result["api_key"] == "sk-dotenv-only"

    def test_blank_dotenv_value_falls_back_to_environ(
        self, isolated_hermes_home, monkeypatch
    ):
        # A blank .env entry must not mask a real shell value (a setup
        # interruption can leave KEY= behind).
        _write_env_file(isolated_hermes_home, AZURE_FOUNDRY_API_KEY="")
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-shell-fallback")

        result = _resolve_azure_foundry(monkeypatch)

        assert result["api_key"] == "sk-shell-fallback"
