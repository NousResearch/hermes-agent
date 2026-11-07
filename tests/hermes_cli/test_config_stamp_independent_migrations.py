"""A current ``_config_version`` does not prove the stamp-independent steps ran.

``run_migrations`` applies :data:`STAMP_INDEPENDENT_STEPS` (46: the retired ``disabled: true`` MCP
key, 50: the retired ``security.tirith_*`` keys) on every pass "whatever the stamp" — because
unreleased builds of the gateway-runtime branch stamped past their own numbers and displaced those
steps. Every caller that gates the pass on ``current_ver < latest_ver`` then skipped it entirely, so
such a home kept ``disabled: true`` forever: the runtime (``tools.mcp_tool_common.mcp_server_enabled``
reads only ``enabled``) kept the server ON while the profile editor showed it off.

These tests drive the four entry points that made that decision — boot maintenance, ``hermes
update``, ``hermes config migrate``, and the sibling-profile sweep — plus the probe they now ask.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

try:  # the probe this fix adds; absent on the base commit, where the tests must still run
    from hermes_cli.config_migrations import STAMP_INDEPENDENT_TRIGGERS
except ImportError:  # pragma: no cover - base-commit control run
    STAMP_INDEPENDENT_TRIGGERS = None


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # sibling-profile scan stays in tmp
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _latest() -> int:
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return int(DEFAULT_CONFIG["_config_version"])


def _stamped_config(home: Path, extra: str = "") -> None:
    """A config.yaml stamped at the CURRENT schema version, i.e. nothing left to migrate by stamp."""
    (home / "config.yaml").write_text(f"_config_version: {_latest()}\n{extra}", encoding="utf-8")


DISABLED_SERVER = "mcp_servers:\n  demo:\n    command: /bin/true\n    disabled: true\n"
TIRITH_KEYS = "security:\n  tirith_enabled: true\n"


def _raw(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _demo_entry(home: Path) -> dict:
    return dict((_raw(home).get("mcp_servers") or {}).get("demo") or {})


def _assert_demo_turned_off(home: Path) -> None:
    from tools.mcp_tool_common import mcp_server_enabled

    entry = _demo_entry(home)
    assert "disabled" not in entry, entry
    assert entry.get("enabled") is False, entry
    # The point of the repair: the runtime agrees with the profile editor.
    assert mcp_server_enabled(entry) is False


# ---------------------------------------------------------------------------
# Boot maintenance (`hermes update` tail and every gateway start)
# ---------------------------------------------------------------------------


def test_boot_maintenance_repairs_a_current_stamped_config(hermes_home):
    from hermes_cli.post_update import step_migrate_config

    _stamped_config(hermes_home, DISABLED_SERVER)

    result = step_migrate_config()

    assert result.get("skipped") != "up-to-date", result
    _assert_demo_turned_off(hermes_home)


def test_boot_maintenance_still_skips_a_clean_current_config(hermes_home):
    """The 99% case must stay a no-op — no rewrite, no backup, same result dict."""
    from hermes_cli.post_update import step_migrate_config

    _stamped_config(hermes_home)
    before = (hermes_home / "config.yaml").read_text(encoding="utf-8")

    assert step_migrate_config() == {"ok": True, "skipped": "up-to-date"}
    assert (hermes_home / "config.yaml").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# `hermes update` (active profile + siblings)
# ---------------------------------------------------------------------------


def test_update_repairs_a_current_stamped_config(hermes_home, monkeypatch):
    import hermes_cli.config as cfg
    from hermes_cli.update_cmd import _check_and_apply_config_migration

    _stamped_config(hermes_home, DISABLED_SERVER)
    # Keep the run on the "nothing new to configure" branch this test is about.
    monkeypatch.setattr(cfg, "get_missing_env_vars", lambda **kw: [])
    monkeypatch.setattr(cfg, "get_missing_config_fields", list)

    _check_and_apply_config_migration()

    _assert_demo_turned_off(hermes_home)


def test_update_repairs_a_sibling_profile_at_a_current_stamp(hermes_home):
    from hermes_cli.update_cmd import _migrate_sibling_profile_configs

    _stamped_config(hermes_home)
    sibling = hermes_home / "profiles" / "work"
    sibling.mkdir(parents=True)
    _stamped_config(sibling, DISABLED_SERVER)

    migrated = _migrate_sibling_profile_configs()

    assert "work" in [name for name, _from, _to in migrated], migrated
    _assert_demo_turned_off(sibling)


# ---------------------------------------------------------------------------
# `hermes config migrate` — the user's explicit repair command
# ---------------------------------------------------------------------------


def test_config_migrate_command_repairs_a_current_stamped_config(hermes_home, monkeypatch):
    import hermes_cli.config as cfg

    _stamped_config(hermes_home, DISABLED_SERVER + TIRITH_KEYS)
    monkeypatch.setattr(cfg, "get_missing_env_vars", lambda **kw: [])
    monkeypatch.setattr(cfg, "get_missing_config_fields", list)

    cfg._cmd_config_migrate(None)

    _assert_demo_turned_off(hermes_home)
    assert "tirith_enabled" not in (_raw(hermes_home).get("security") or {})


# ---------------------------------------------------------------------------
# Docker boot (`scripts/docker_config_migrate.py`, the container's every start)
# ---------------------------------------------------------------------------


def _docker_module():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "docker_config_migrate.py"
    spec = importlib.util.spec_from_file_location("docker_config_migrate_stamp_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_boot_repairs_a_current_stamped_config(hermes_home):
    _stamped_config(hermes_home, DISABLED_SERVER)

    assert _docker_module().main() == 0

    _assert_demo_turned_off(hermes_home)


def test_docker_boot_still_skips_a_clean_current_config(hermes_home):
    _stamped_config(hermes_home)
    before = (hermes_home / "config.yaml").read_text(encoding="utf-8")

    assert _docker_module().main() == 0

    assert (hermes_home / "config.yaml").read_text(encoding="utf-8") == before
    assert not (hermes_home / "backups").exists(), "a no-op boot must not write a backup"


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_pending_probe_covers_every_stamp_independent_step(hermes_home):
    triggers = STAMP_INDEPENDENT_TRIGGERS
    if triggers is None:
        pytest.skip("stamp_independent_pending() not present on this commit")
    assert triggers is not None
    from hermes_cli.config_migrations import STAMP_INDEPENDENT_STEPS, stamp_independent_pending

    # A step that is re-run regardless of the stamp needs a trigger here, or the callers below
    # keep short-circuiting it.
    assert set(triggers) == set(STAMP_INDEPENDENT_STEPS)

    _stamped_config(hermes_home)
    assert stamp_independent_pending() is False

    _stamped_config(hermes_home, DISABLED_SERVER)
    assert stamp_independent_pending() is True

    _stamped_config(hermes_home, TIRITH_KEYS)
    assert stamp_independent_pending() is True


def test_pending_probe_ignores_an_inert_falsy_disabled(hermes_home):
    if STAMP_INDEPENDENT_TRIGGERS is None:
        pytest.skip("stamp_independent_pending() not present on this commit")
    from hermes_cli.config_migrations import stamp_independent_pending

    _stamped_config(hermes_home, "mcp_servers:\n  demo:\n    command: /bin/true\n    disabled: false\n")

    # v46 leaves a falsy `disabled` alone (the runtime never read it), so there is nothing to do.
    assert stamp_independent_pending() is False
