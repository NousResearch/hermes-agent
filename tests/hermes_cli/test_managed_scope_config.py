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


def test_managed_list_wins_wholesale(homes):
    """D3: a managed list value replaces the user's wholesale."""
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "toolsets:\n  enabled: [a, b, c]\n")
    _write(managed / "config.yaml", "toolsets:\n  enabled: [x]\n")
    assert cfg_get(load_config(), "toolsets", "enabled") == ["x"]


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


def test_managed_nested_dict_default_flattens_on_load(homes):
    """A dict-valued managed ``model.default`` must flatten on load.

    ``load_config()`` merges the managed overlay after its single
    normalization pass, so a managed ``model.default: {provider: ...,
    model: ...}`` used to reach runtime readers as a raw dict. The overlay
    is now normalized before merging (parity with
    ``managed_scope.apply_managed_overlay``), so the merged config exposes a
    string ``default`` paired with the nested ``provider``.
    """
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    _write(managed / "config.yaml", "model:\n  default:\n    provider: nous\n    model: managed/nested\n")
    cfg = load_config()
    assert cfg_get(cfg, "model", "default") == "managed/nested"
    assert cfg_get(cfg, "model", "provider") == "nous"


def test_managed_bare_string_model_flattens_to_default_on_load(homes):
    """A bare ``model: <string>`` in the managed file stays a dict shape.

    Mirrors the existing managed-overlay contract: a bare string model must
    merge as ``model.default`` so readers that do
    ``cfg["model"]["default"]`` keep working (never a bare string at
    ``cfg["model"]``).
    """
    from hermes_cli.config import load_config, cfg_get

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    _write(managed / "config.yaml", "model: managed/bare\n")
    cfg = load_config()
    assert cfg_get(cfg, "model", "default") == "managed/bare"
