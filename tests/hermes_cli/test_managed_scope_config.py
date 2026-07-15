"""Config integration tests — managed scope wins over user config at the leaf."""
import textwrap

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
    return home, managed


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def test_managed_beats_user(homes):
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")
    assert cfg_get(load_config(), "model", "default") == "managed/model"


def test_managed_leaf_does_not_freeze_siblings(homes):
    """D3/Q4: pinning model.default leaves model.fallback user-controlled."""
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n  fallback: user/fb\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")
    cfg = load_config()
    assert cfg_get(cfg, "model", "default") == "managed/model"
    assert cfg_get(cfg, "model", "fallback") == "user/fb"  # sibling preserved


def test_no_managed_config_is_unchanged(homes):
    from hermes_cli.config import load_config, cfg_get

    home, _ = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    assert cfg_get(load_config(), "model", "default") == "user/model"


def test_managed_list_wins_wholesale(homes):
    """D3: a managed list value replaces the user's wholesale."""
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "toolsets:\n  enabled: [a, b, c]\n")
    _write(managed / "config.yaml", "toolsets:\n  enabled: [x]\n")
    assert cfg_get(load_config(), "toolsets", "enabled") == ["x"]


def test_editing_managed_file_invalidates_cache(homes):
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    _write(managed / "config.yaml", "model:\n  default: managed/v1\n")
    assert cfg_get(load_config(), "model", "default") == "managed/v1"
    _write(managed / "config.yaml", "model:\n  default: managed/v2\n")
    assert cfg_get(load_config(), "model", "default") == "managed/v2"


def test_user_cannot_shadow_managed_literal_via_envref(homes, monkeypatch):
    """A managed literal must NOT be expandable via a ${VAR} the user controls.

    The managed value is a plain literal 'managed/locked' with no ${...}, so a
    user-defined env var has nothing to substitute. This asserts the managed
    literal survives verbatim regardless of user env, and that managed wins.
    """
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    monkeypatch.setenv("EVIL", "user/override")
    _write(home / "config.yaml", "model:\n  default: ${EVIL}\n")
    _write(managed / "config.yaml", "model:\n  default: managed/locked\n")
    assert cfg_get(load_config(), "model", "default") == "managed/locked"
