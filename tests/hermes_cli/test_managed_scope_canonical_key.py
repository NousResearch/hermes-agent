"""``config set`` / ``config unset`` rewrite some keys before writing (``platforms.<p>.<display key>``
to ``display.platforms.<p>.<key>``, ``gateway.platforms.*`` to ``platforms.*``, bare ``model`` to
``model.default``). The managed-key and remote-lock check must see the key that is written, not
only the key the user typed."""

import pytest
from utils import load_yaml_file_readonly

MANAGED = """\
model:
  default: managed/model
  base_url: https://admin.example/v1
display:
  platforms:
    telegram:
      streaming: false
"""


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text(MANAGED, encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\ndisplay:\n  platforms:\n    telegram:\n      streaming: true\n",
        encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home


def _user_config(home):
    return load_yaml_file_readonly(home / "config.yaml")


@pytest.mark.parametrize("typed_key", ["platforms.telegram.streaming", "gateway.platforms.telegram.streaming"])
def test_set_refuses_a_key_that_is_rewritten_onto_a_managed_key(homes, capsys, typed_key):
    from hermes_cli.config import set_config_value

    with pytest.raises(SystemExit) as exc:
        set_config_value(typed_key, "false")
    assert exc.value.code != 0
    assert "managed by your administrator" in capsys.readouterr().err
    assert _user_config(homes)["display"]["platforms"]["telegram"]["streaming"] is True


def test_set_refuses_bare_model_when_model_default_is_managed(homes, capsys):
    from hermes_cli.config import set_config_value

    with pytest.raises(SystemExit) as exc:
        set_config_value("model", "user/override")
    assert exc.value.code != 0
    assert "managed by your administrator" in capsys.readouterr().err
    assert "default" not in _user_config(homes)["model"]


def test_unset_refuses_a_key_that_is_rewritten_onto_a_managed_key(homes, capsys):
    from hermes_cli.config import unset_config_value

    with pytest.raises(SystemExit) as exc:
        unset_config_value("platforms.telegram.streaming")
    assert exc.value.code != 0
    assert "managed by your administrator" in capsys.readouterr().err
    assert _user_config(homes)["display"]["platforms"]["telegram"]["streaming"] is True


def test_set_of_an_unmanaged_rewritten_key_still_writes(homes):
    from hermes_cli.config import set_config_value

    set_config_value("platforms.discord.streaming", "false")
    assert _user_config(homes)["display"]["platforms"]["discord"]["streaming"] is False


def test_set_refuses_model_api_base_alias_to_managed_base_url(homes, capsys):
    """The api_base alias must not write a value pinned as model.base_url."""
    from hermes_cli.config import set_config_value

    config_path = homes / "config.yaml"
    before = config_path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        set_config_value("model.api_base", "https://user.example/v1")
    assert exc.value.code != 0
    assert "managed by your administrator" in capsys.readouterr().err
    assert config_path.read_bytes() == before
