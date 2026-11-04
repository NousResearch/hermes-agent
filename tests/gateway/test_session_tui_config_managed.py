"""The shared owner's ``config.set`` refuses a key the administrator's managed scope pins, like
``hermes config set``; before, it wrote the pinned key and answered success while the managed value
kept winning on every read. A managed scope that pins something else must not block other keys.
"""
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml
from hermes_state_runtime import RuntimeStoreError

from gateway.session_tui_config import _set_display, _write_profile_config


@pytest.fixture
def home(tmp_path, monkeypatch):
    from hermes_cli import managed_scope
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: corp\n")
    (home / "config.yaml").write_text("display:\n  tui_theme: dark\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    yield home
    managed_scope.invalidate_managed_cache()


def _owner_write(home, key, value):
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(home)))
    return _write_profile_config(connection, lambda cfg: _set_display(cfg, key, value))


def test_owner_config_set_refuses_a_managed_key_and_writes_nothing(home):
    before = (home / "config.yaml").read_text()
    with pytest.raises(RuntimeStoreError) as refused:
        _owner_write(home, "skin", "mono")
    assert (refused.value.reason, refused.value.field) == ("permission_denied", "display.skin")
    assert (home / "config.yaml").read_text() == before


def test_owner_config_set_still_writes_a_key_the_scope_does_not_pin(home):
    assert _owner_write(home, "theme", "light") == {"value": "light"}
    assert yaml.safe_load((home / "config.yaml").read_text())["display"]["tui_theme"] == "light"
