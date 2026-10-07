"""Regression harness — pins config/env load behavior BEFORE managed scope exists.

Every test here must keep passing through all later phases when NO managed scope
is present. They are the 'managed scope is invisible when absent' contract.
"""
import os
import textwrap

import pytest


@pytest.fixture
def rabbit_home(tmp_path, monkeypatch):
    home = tmp_path / "rabbit_home"
    home.mkdir()
    monkeypatch.setenv("RABBIT_HOME", str(home))
    # No managed dir: point the override at a guaranteed-absent path so a real
    # /etc/rabbit on the dev/CI box can't influence the test.
    monkeypatch.setenv("RABBIT_MANAGED_DIR", str(tmp_path / "no_such_managed_dir"))
    # Clear caches so each test re-reads from disk.
    import rabbit_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    cfg.invalidate_env_cache()
    return home


def _write_user_config(home, body: str):
    (home / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    import rabbit_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()


def test_user_config_overrides_default(rabbit_home, monkeypatch):
    from rabbit_cli.config import load_config, cfg_get

    _write_user_config(
        rabbit_home,
        """
        model:
          default: user/model-x
        """,
    )
    cfg = load_config()
    assert cfg_get(cfg, "model", "default") == "user/model-x"


def test_env_expansion_in_user_config(rabbit_home, monkeypatch):
    from rabbit_cli.config import load_config, cfg_get

    monkeypatch.setenv("MY_BASE", "https://example.test")
    _write_user_config(
        rabbit_home,
        """
        providers:
          custom:
            base_url: ${MY_BASE}/v1
        """,
    )
    cfg = load_config()
    assert cfg_get(cfg, "providers", "custom", "base_url") == "https://example.test/v1"


def test_user_env_overrides_shell(tmp_path, monkeypatch):
    from rabbit_cli.env_loader import load_rabbit_dotenv

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("FOO_TOKEN=from_user_env\n", encoding="utf-8")
    monkeypatch.setenv("FOO_TOKEN", "from_shell")
    load_rabbit_dotenv(rabbit_home=str(home))
    assert os.environ["FOO_TOKEN"] == "from_user_env"


