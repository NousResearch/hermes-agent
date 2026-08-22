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




# ── bulk save strips managed leaves ──────────────────────────────────────────


def test_bulk_merge_save_does_not_restore_existing_managed_leaf(homes, capsys):
    import yaml

    from hermes_cli.config import get_config_path, save_config

    config_path = get_config_path()
    config_path.write_text(
        "model:\n  default: stale/user-model\nx_unknown:\n  keep: true\n",
        encoding="utf-8",
    )

    save_config({"timezone": "Asia/Bangkok"}, merge_existing=True)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "model" not in saved or "default" not in saved["model"]
    assert saved["x_unknown"] == {"keep": True}
    assert saved["timezone"] == "Asia/Bangkok"
    # The stale leaf came in via the merge, not the caller's dict, so the
    # notice only fires if the strip runs after _merge_partial_save.
    assert "model.default" in capsys.readouterr().err
