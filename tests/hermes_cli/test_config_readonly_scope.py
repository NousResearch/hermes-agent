"""Context-scoped read overlay for load_config_readonly() (#125624).

Behavior contracts against the real load path with a temp HERMES_HOME:
- inside readonly_config_scope, load_config_readonly() serves the projection;
- the shared cache is never mutated and the overlay vanishes on scope exit;
- load_config() (the writable path) keeps the persisted view inside the scope.

The scope is imported from its defining module hermes_cli.config_readonly; the reads below go
through the hermes_cli.config re-export, which is the seam production callers use.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: base-model\nagent:\n  max_turns: 50\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import config as cfgmod

    cfgmod._LOAD_CONFIG_CACHE.clear()
    cfgmod._RAW_CONFIG_CACHE.clear()
    yield home
    cfgmod._LOAD_CONFIG_CACHE.clear()
    cfgmod._RAW_CONFIG_CACHE.clear()
    from hermes_cli.config_readonly import _readonly_projection

    _readonly_projection.set(None)


def _project(cfg):
    cfg = dict(cfg)
    model = dict(cfg.get("model") or {})
    model["default"] = "overlay-model"
    cfg["model"] = model
    return cfg


def test_overlay_visible_inside_scope_and_cache_untouched(config_home):
    from hermes_cli import config as cfgmod
    from hermes_cli.config_readonly import readonly_config_scope

    before = cfgmod.load_config_readonly()
    assert before.get("model", {}).get("default") == "base-model"

    with readonly_config_scope(_project):
        overlaid = cfgmod.load_config_readonly()
        assert overlaid.get("model", {}).get("default") == "overlay-model"

    after = cfgmod.load_config_readonly()
    assert after.get("model", {}).get("default") == "base-model"
    # Unset path keeps the zero-copy identity invariant (one ContextVar.get on hit).
    assert after is before


def test_writable_path_keeps_persisted_view_inside_scope(config_home):
    from hermes_cli import config as cfgmod
    from hermes_cli.config_readonly import readonly_config_scope

    with readonly_config_scope(_project):
        writable = cfgmod.load_config()
        assert writable.get("model", {}).get("default") == "base-model"
