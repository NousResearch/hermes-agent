"""Surfacing tests — managed scope shown in `config show` and `hermes doctor`."""
import pytest
from hermes_cli import doctor_config


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (home / "config.yaml").write_text("model:\n  default: user/model\n", encoding="utf-8")
    (managed / "config.yaml").write_text(
        "model:\n  default: managed/model\n", encoding="utf-8"
    )
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed




def test_config_show_no_managed_scope_silent(tmp_path, monkeypatch, capsys):
    """With no managed scope, the managed header must not appear."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "nope"))
    (home / "config.yaml").write_text("model:\n  default: user/model\n", encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    from hermes_cli.config import show_config

    show_config()
    out = capsys.readouterr().out.lower()
    assert "managed by your administrator" not in out




def test_doctor_silent_with_no_managed_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "nope"))
    from hermes_cli import managed_scope, doctor

    managed_scope.invalidate_managed_cache()
    doctor_config.managed_scope_check()
    assert capsys.readouterr().out.strip() == ""


@pytest.mark.parametrize("content", ["model: [unclosed\n", "- just\n- a list\n", ""])
def test_doctor_flags_unusable_managed_config(homes, capsys, content):
    """A managed config.yaml that is malformed, not a mapping or empty is IGNORED fail-open by the overlay —
    doctor must say so instead of reporting a green 'Managed scope active: 0 config key(s)'."""
    from hermes_cli import managed_scope

    _, managed = homes
    (managed / "config.yaml").write_text(content, encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    doctor_config.managed_scope_check()
    out = capsys.readouterr().out
    assert "✓" not in out
    assert "config.yaml" in out and "NOT applied" in out


def test_doctor_ok_with_valid_managed_config(homes, capsys):
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    doctor_config.managed_scope_check()
    out = capsys.readouterr().out
    assert "✓" in out and "1 config key(s)" in out and "NOT applied" not in out
