"""``hermes send`` must keep load_hermes_dotenv()'s credential precedence (#151292).

``send_cmd._load_hermes_env`` reloads ``<HERMES_HOME>/.env`` with override=True after startup already
ran ``load_hermes_dotenv()``. Without the restore, a ``.env`` placeholder replaced the value an external
secret source (``secrets.command``, Bitwarden) resolved, and the user ``.env`` beat the
administrator-managed one, so ``send`` reported the platform as not configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import env_loader, send_cmd


@pytest.fixture(autouse=True)
def _fresh_state():
    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    yield
    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


def _register_fake_source(values: dict[str, str], *, override_existing: bool):
    from agent.secret_sources import registry as reg_module
    from agent.secret_sources.base import FetchResult, SecretSource

    class _Fake(SecretSource):
        name = "fakebulk"
        label = "Fake"
        shape = "bulk"
        override_existing_default = override_existing

        def fetch(self, cfg, home_path):
            result = FetchResult()
            result.secrets = dict(values)
            return result

    reg_module.register_source(_Fake(), replace=True)


def _make_home(tmp_path: Path, monkeypatch, env_text: str) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("secrets:\n  fakebulk:\n    enabled: true\n", encoding="utf-8")
    (home / ".env").write_text(env_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    return home


def test_send_keeps_the_value_an_authoritative_source_resolved(tmp_path, monkeypatch):
    home = _make_home(tmp_path, monkeypatch, "DISCORD_BOT_TOKEN=__COMMAND_MANAGED__\n")
    _register_fake_source({"DISCORD_BOT_TOKEN": "helper-value"}, override_existing=True)

    env_loader.load_hermes_dotenv(hermes_home=home)  # CLI startup
    assert os.environ["DISCORD_BOT_TOKEN"] == "helper-value"

    send_cmd._load_hermes_env()
    assert os.environ["DISCORD_BOT_TOKEN"] == "helper-value", (
        "hermes send wrote the .env placeholder back over the secret-source value"
    )


def test_send_still_lets_dotenv_beat_a_gap_filling_source(tmp_path, monkeypatch):
    """A source without ``override_existing`` only fills gaps, so a later .env value still wins."""
    home = _make_home(tmp_path, monkeypatch, "")
    _register_fake_source({"DISCORD_BOT_TOKEN": "helper-value"}, override_existing=False)

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["DISCORD_BOT_TOKEN"] == "helper-value"

    (home / ".env").write_text("DISCORD_BOT_TOKEN=local\n", encoding="utf-8")
    send_cmd._load_hermes_env()
    assert os.environ["DISCORD_BOT_TOKEN"] == "local"


def test_send_keeps_the_managed_env_over_the_user_env(tmp_path, monkeypatch):
    home = _make_home(tmp_path, monkeypatch, "DISCORD_BOT_TOKEN=user-value\n")
    (home / "config.yaml").write_text("", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("DISCORD_BOT_TOKEN=managed-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["DISCORD_BOT_TOKEN"] == "managed-value"

    send_cmd._load_hermes_env()
    assert os.environ["DISCORD_BOT_TOKEN"] == "managed-value", (
        "hermes send let the user .env beat the administrator-managed .env"
    )
