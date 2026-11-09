"""A plugin directory discovery cannot read is reported, not silently dropped (#152247).

Discovery skips one unreadable directory so every other plugin stays visible (#111804). What was
missing is the report: ``hermes plugins list`` and the Desktop Plugins page rendered a row list that
simply lacked the folder, so the plugin read as uninstalled. These tests pin the report on both
surfaces plus the loader's own scan, and the repair command each one names.
"""

import json
import os
from argparse import Namespace
from unittest.mock import patch

import pytest

from hermes_cli import plugins_cmd
from hermes_cli import plugins_discovery
from hermes_dir_acl import unusable_dir_remedy

pytestmark = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs a mode-000 directory a non-root user can create and cannot read",
)


def _list_args(**kwargs) -> Namespace:
    defaults = {"enabled": False, "user": False, "no_bundled": False, "plain": False, "json": False}
    defaults.update(kwargs)
    return Namespace(**defaults)


@pytest.fixture()
def plugin_home(tmp_path, monkeypatch):
    """A user plugins dir with one readable and one mode-000 plugin, wired as the only scan root."""
    root = tmp_path / "plugins"
    good = root / "good-plugin"
    good.mkdir(parents=True)
    (good / "plugin.yaml").write_text("name: good-plugin\nversion: 1.0.0\ndescription: Readable\n")
    locked = root / "locked-plugin"
    locked.mkdir()
    (locked / "plugin.yaml").write_text("name: locked-plugin\nversion: 2.0.0\n")
    os.chmod(locked, 0o000)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: root)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    monkeypatch.setattr("hermes_cli.plugins.discover_entrypoint_manifests", list)
    yield root, good, locked
    os.chmod(locked, 0o700)  # let pytest clean the tmp tree up


def test_discovery_reports_the_directory_it_could_not_read(plugin_home):
    """The scan still lists the readable plugin AND hands the skipped directory back to its caller."""
    _root, good, locked = plugin_home
    skips: list = []

    entries = plugins_cmd._discover_all_plugins(skips)

    assert [entry[0] for entry in entries] == ["good-plugin"]
    assert [str(path) for path, _error in skips] == [str(locked)]
    # The reason travels with it: the row has to say why the folder is unusable.
    assert "Permission denied" in skips[0][1] or "permission" in skips[0][1].lower()
    assert str(good) not in [str(path) for path, _error in skips]


def test_cmd_list_plain_names_the_unreadable_plugin(plugin_home):
    """``--plain`` is what scripts and bug reports read: the row must be there, marked unreadable."""
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        plugins_cmd.cmd_list(_list_args(plain=True))

    printed = {line.split()[-1]: line for line in out.getvalue().splitlines() if line.strip()}
    assert "locked-plugin" in printed, out.getvalue()
    assert printed["locked-plugin"].startswith("unreadable")
    assert not printed["good-plugin"].startswith("unreadable")


def test_cmd_list_json_carries_the_status_and_the_repair(plugin_home):
    """The JSON listing feeds tooling: an unreadable directory is a row with status + the repair."""
    import io
    from contextlib import redirect_stdout

    _root, _good, locked = plugin_home
    out = io.StringIO()
    with redirect_stdout(out):
        plugins_cmd.cmd_list(_list_args(json=True))
    payload = {row["name"]: row for row in json.loads(out.getvalue())}

    assert payload["locked-plugin"]["status"] == "unreadable"
    assert unusable_dir_remedy(locked) in payload["locked-plugin"]["description"]
    assert payload["good-plugin"]["status"] != "unreadable"


def test_plugin_rows_carry_it_to_the_plugins_page(plugin_home):
    """Desktop's Plugins page renders ``plugins.manage list``: the skipped folder is one of its rows."""
    from tui_gateway import server

    _root, _good, locked = plugin_home
    with patch("hermes_cli.plugins_cmd_catalog.catalog_rows_maps", return_value=({}, {}, {})), \
         patch("hermes_cli.plugins_cmd._read_install_metadata", return_value={}):
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    rows = {row["name"]: row for row in resp["result"]["plugins"]}
    assert rows["locked-plugin"]["status"] == "unreadable"
    assert unusable_dir_remedy(locked) in rows["locked-plugin"]["description"]
    # An empty key means no toggle is offered for a plugin that cannot load.
    assert rows["locked-plugin"]["key"] == ""
    assert rows["good-plugin"]["status"] != "unreadable"


def test_the_loaders_own_scan_names_the_repair(plugin_home, caplog):
    """What the loader logs is the only trace a headless run leaves, so it must carry the repair."""
    _root, _good, locked = plugin_home
    caplog.set_level("WARNING", logger="hermes_cli.plugins")

    assert plugins_discovery.scan_directory(locked, "user") == []

    assert unusable_dir_remedy(locked) in caplog.text
    assert str(locked) in caplog.text
