"""Write-guard tests — managed keys can't be set/removed by the user."""

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    (managed / "config.yaml").write_text(
        "model:\n  default: managed/model\n", encoding="utf-8"
    )
    managed_scope.invalidate_managed_cache()
    return home, managed


def test_config_set_managed_key_rejected(homes, capsys):
    from hermes_cli.config import set_config_value

    with pytest.raises(SystemExit) as exc:
        set_config_value("model.default", "user/override")
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "managed" in (captured.out + captured.err).lower()


# ── env write guards ─────────────────────────────────────────────────────────


@pytest.fixture
def env_homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / ".env").write_text(
        "OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8"
    )
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return home, managed


def test_save_env_value_managed_key_rejected(env_homes, capsys):
    from hermes_cli.config import save_env_value, get_env_path

    save_env_value("OPENAI_API_BASE", "https://user.example/v1")
    assert "managed" in capsys.readouterr().err.lower()
    env_path = get_env_path()
    body = env_path.read_text() if env_path.exists() else ""
    assert "user.example" not in body


@pytest.mark.parametrize("command", ["set", "unset"])
def test_managed_bare_config_route_refusal_is_non_mutating(env_homes, capsys, command):
    """Regression for #119928: the entire bare-setting route refuses atomically."""
    import argparse

    from hermes_cli import managed_scope
    from hermes_cli.config import config_command, invalidate_env_cache

    home, managed = env_homes
    managed_env = managed / ".env"
    user_env = home / ".env"
    config_path = home / "config.yaml"
    auth_path = home / "auth.json"
    managed_env.write_text(
        "# administrator-owned\nTELEGRAM_HOME_CHANNEL=111\n", encoding="utf-8"
    )
    user_env.write_text(
        "# user-owned\nTELEGRAM_HOME_CHANNEL=5\nC18_KEEP=unchanged\n", encoding="utf-8"
    )
    config_path.write_text(
        "model:\n  default: deepseek-chat\n"
        "TELEGRAM_HOME_CHANNEL: '5'  # stale duplicate\n",
        encoding="utf-8",
    )
    auth_path.write_text(
        '{\n  "credential_pool": {"telegram": []},\n  "suppressed_sources": {}\n}\n',
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()
    invalidate_env_cache()
    stores = {
        path: path.read_bytes()
        for path in (managed_env, user_env, config_path, auth_path)
    }
    args = argparse.Namespace(
        config_command=command,
        key="TELEGRAM_HOME_CHANNEL",
        value="999",
        force=False,
    )

    with pytest.raises(SystemExit) as exc:
        config_command(args)

    assert exc.value.code == 1
    output = capsys.readouterr()
    lock_action = "set" if command == "set" else "remove"
    success_action = "Set" if command == "set" else "Unset"
    assert f"Cannot {lock_action} TELEGRAM_HOME_CHANNEL" in output.err
    assert f"✓ {success_action} TELEGRAM_HOME_CHANNEL" not in output.out + output.err
    assert {path: path.read_bytes() for path in stores} == stores


# ── bulk save strips managed leaves ──────────────────────────────────────────
