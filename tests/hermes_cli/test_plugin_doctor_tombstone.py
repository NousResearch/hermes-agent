"""`plugins doctor` must not break on a deleted named profile's tombstone.

Doctor's probe home is created under the ambient temp root — normally a profile's
``cache/scratch`` (Hermes points ``TMPDIR`` there at boot). ``named_profile_home()``
attributes that whole subtree to the named profile, and ``ensure_hermes_home()`` —
called by every ``load_config()`` during plugin registration — refuses the probe home
via ``assert_named_profile_home_live`` as soon as that profile carries a tombstone,
killing the unrelated doctor command with ``FileNotFoundError: Named profile home
does not exist`` and ``registrations: 0 tool(s), 0 hook(s)``. A tombstone's only job
is refusing name REUSE; the probe home is not a use of the deleted profile, so Doctor
must skip (exclude) such scratch roots and still register tools normally. The
tombstone's own protection stays: the deleted profile HOME itself is still refused.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def _tombstoned_scratch(tmp_path: Path) -> Path:
    """A real ``<root>/.hermes/profiles/<name>`` tree whose profile was deleted: the
    tombstone marker sits under ``profiles/.deleted/`` and the stale ``cache/scratch``
    is where a scoped session's TMPDIR (and thus Doctor's probe home) would land."""
    scratch = tmp_path / ".hermes" / "profiles" / "gone" / "cache" / "scratch"
    scratch.mkdir(parents=True)
    marker = tmp_path / ".hermes" / "profiles" / ".deleted" / "gone"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("deleted\n", encoding="utf-8")
    return scratch


def _write_plugin(root: Path) -> Path:
    plugin = root / "tombstone-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "name: tombstone-plugin\nprovides_tools: [tombstone_ping]\n", encoding="utf-8"
    )
    (plugin / "__init__.py").write_text(
        "import json\n\n"
        "def ping(args, **kwargs):\n    return json.dumps({'ok': True})\n\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='tombstone_ping', toolset='tombstone', "
        "schema={'name': 'tombstone_ping', 'description': 'test', "
        "'parameters': {'type': 'object'}}, handler=ping)\n",
        encoding="utf-8",
    )
    return plugin


def test_doctor_probe_home_skips_tombstoned_profile_scratch(tmp_path, monkeypatch):
    """The regression: probe scratch inside a tombstoned profile's tree no longer kills
    the doctor run — it skips that root and registers the plugin's tools normally."""
    from hermes_cli.plugin_dev import doctor_plugin

    scratch = _tombstoned_scratch(tmp_path)
    plugin = _write_plugin(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    report = doctor_plugin(plugin)

    assert report.ok, report.format_text()
    assert report.registered_tools == ("tombstone_ping",)
    # The probe home never materialized inside the deleted profile's tree.
    assert list(scratch.iterdir()) == []


def test_plugin_doctor_command_succeeds_with_tombstone_present(tmp_path, monkeypatch):
    """`hermes plugins doctor <pkg> --ci` (the real command shape) exits 0 and registers."""
    from hermes_cli.plugins_cmd import cmd_plugin_doctor

    scratch = _tombstoned_scratch(tmp_path)
    plugin = _write_plugin(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    cmd_plugin_doctor(str(plugin), ci=True)  # raises SystemExit(1) on any finding

    assert list(scratch.iterdir()) == []


def test_probe_home_root_excludes_tombstoned_but_keeps_live_profile_scratch(
    tmp_path, monkeypatch
):
    from hermes_cli.plugin_dev import _probe_home_root

    scratch = _tombstoned_scratch(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    chosen = _probe_home_root()
    assert chosen != str(scratch)
    assert not Path(chosen).is_relative_to(scratch)

    # A LIVE named profile's scratch stays the probe root (unchanged behavior).
    live = tmp_path / ".hermes" / "profiles" / "charlie" / "cache" / "scratch"
    live.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(live))
    assert _probe_home_root() == str(live)


def test_tombstone_still_refuses_direct_use_of_the_deleted_home(tmp_path, monkeypatch):
    """The fix only moves Doctor's probe home; the tombstone's refusal of the deleted
    profile HOME itself (name-reuse / resurrection protection) is untouched."""
    from hermes_cli.config import ensure_hermes_home

    scratch = _tombstoned_scratch(tmp_path)
    dead_home = scratch.parent.parent  # profiles/gone
    monkeypatch.setenv("HERMES_HOME", str(dead_home))

    with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
        ensure_hermes_home()
