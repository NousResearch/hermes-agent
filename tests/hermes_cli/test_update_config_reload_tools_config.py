"""``hermes update`` runs config migrations in the PRE-pull updater process. A migration step
that imports a helper at call time (``_migrate_to_45`` → ``hermes_cli.tools_config``) resolves
against whatever OLD module object is still cached, and dies with ``cannot import name`` when the
pull added that symbol — config silently stays behind while the code moves on (#111271).

The class fix: the migration step evicts every cached Hermes module (``_purge_stale_hermes_modules``,
the same primitive the fleet-restart phase uses) before importing migration code, so ANY symbol the
pull added to ANY module a migration imports is found. The pinned-list reload of ``tools_config`` is
the per-symptom half; the purge covers the class.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """The purge under test evicts real Hermes modules; put the originals back for later tests."""
    snapshot = dict(sys.modules)
    yield
    for name, mod in snapshot.items():
        sys.modules[name] = mod


def test_reload_config_modules_restores_missing_tools_config_symbol():
    """A pre-pull ``tools_config`` cache must be reloaded before migrations run."""
    tools_config = sys.modules.get("hermes_cli.tools_config")
    if tools_config is None:
        import hermes_cli.tools_config as tools_config  # noqa: F811
    assert hasattr(tools_config, "_configurable_keys")

    del tools_config._configurable_keys
    try:
        from hermes_cli.update_cmd_config import _reload_config_modules
        _reload_config_modules()
        assert hasattr(sys.modules["hermes_cli.tools_config"], "_configurable_keys")
    finally:
        reloaded = sys.modules["hermes_cli.tools_config"]
        if not hasattr(reloaded, "_configurable_keys"):
            importlib.reload(reloaded)  # keep other tests on a clean module


def test_update_migration_survives_stale_module_missing_call_time_symbol(tmp_path, monkeypatch):
    """The reporter's exact shape: the cached ``tools_config`` predates ``_configurable_keys`` while
    the on-disk v45 step imports it at call time. The updater's migration entry must still carry the
    v44 store to the latest version instead of printing "Config format update failed"."""
    home = tmp_path / "flat-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "_config_version: 44\nplatform_toolsets:\n  telegram:\n    - web\n    - file\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        import hermes_cli.tools_config as tools_config
        monkeypatch.delattr(tools_config, "_configurable_keys")

        from hermes_cli.update_cmd import _check_and_apply_config_migration
        _check_and_apply_config_migration(assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None)
    finally:
        from hermes_constants import reset_hermes_home_override
        reset_hermes_home_override(token)

    import yaml
    from hermes_cli.config import DEFAULT_CONFIG

    on_disk = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert on_disk["_config_version"] == DEFAULT_CONFIG["_config_version"], on_disk


def test_update_migration_import_failure_after_purge_prints_fallback(tmp_path, monkeypatch, capsys):
    """The purge protects ``hermes_constants`` (identity-bearing ContextVar state), so the
    post-purge ``from hermes_cli.config import ...`` re-executes NEW config.py against that OLD
    root module and can raise ImportError. That must print the 'run hermes config migrate' fallback and
    return — not escape and abort the rest of post-update maintenance (fleet restart)."""
    home = tmp_path / "flat-home"
    home.mkdir()
    (home / "config.yaml").write_text("_config_version: 44\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config  # noqa: F401  (pre-pull world has config cached)
    import hermes_constants
    # A pre-pull root module lacking a symbol the post-pull hermes_cli/config.py imports at module level.
    monkeypatch.delattr(hermes_constants, "get_process_hermes_home")

    from hermes_cli.update_cmd import _check_and_apply_config_migration
    _check_and_apply_config_migration(assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None)

    out = capsys.readouterr().out
    assert "Could not check config version" in out, out
    assert "hermes config migrate" in out, out
