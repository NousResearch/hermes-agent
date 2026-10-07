"""Surfacing tests — managed scope shown in `config show` and `rabbit doctor`."""
import pytest
from rabbit_cli import doctor_config


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("RABBIT_HOME", str(home))
    monkeypatch.setenv("RABBIT_MANAGED_DIR", str(managed))
    (home / "config.yaml").write_text("model:\n  default: user/model\n", encoding="utf-8")
    (managed / "config.yaml").write_text(
        "model:\n  default: managed/model\n", encoding="utf-8"
    )
    import rabbit_cli.config as cfg
    from rabbit_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed




def test_config_show_no_managed_scope_silent(tmp_path, monkeypatch, capsys):
    """With no managed scope, the managed header must not appear."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("RABBIT_HOME", str(home))
    monkeypatch.setenv("RABBIT_MANAGED_DIR", str(tmp_path / "nope"))
    (home / "config.yaml").write_text("model:\n  default: user/model\n", encoding="utf-8")
    import rabbit_cli.config as cfg
    from rabbit_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    from rabbit_cli.config import show_config

    show_config()
    out = capsys.readouterr().out.lower()
    assert "managed by your administrator" not in out




def test_doctor_silent_with_no_managed_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RABBIT_MANAGED_DIR", str(tmp_path / "nope"))
    from rabbit_cli import managed_scope, doctor

    managed_scope.invalidate_managed_cache()
    doctor_config.managed_scope_check()
    assert capsys.readouterr().out.strip() == ""
