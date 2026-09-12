"""Publication failures through the command and real worker, not a fake sync."""
import shutil
import sys

import pytest

from tests.hermes_cli.plugin_worker_support import (
    plugin_world as plugin_world,
    worker_command,
    isolated_python as isolated_python,
)


@pytest.mark.parametrize("failure", ["config", "facts"])
def test_failed_publication_preserves_selection_and_imports(plugin_world, monkeypatch, failure):
    from hermes_cli import runtime_state
    from hermes_cli.plugins_admission import AdmissionRefused
    from pm import client, paths, receipt

    world = plugin_world
    origin, sha = world.origin()
    world.command("install", identifier=origin.as_uri(), ref=sha, no_enable=True, allow_removed=True)
    world.command("enable", name="plugin-worker-proof", no_allow_tool_override=True)
    world.imports()
    selected = world.selected()
    origin, sha = world.origin(name="publication-candidate")
    world.command("install", identifier=origin.as_uri(), ref=sha, no_enable=True, allow_removed=True)
    config = world.home / "config.yaml"
    watched = [config, paths.runtime_facts_path()]
    before = {path: path.read_bytes() for path in watched}
    if failure == "config":
        atomic_bytes = runtime_state._atomic_bytes
        def fail(path, data):
            if path == config and b"publication-candidate" in data:
                raise OSError("fixture config disk full")
            return atomic_bytes(path, data)
        monkeypatch.setattr(runtime_state, "_atomic_bytes", fail)
    else:
        # Fail on the worker's real disk-publication boundary, after proving
        # the client callback has already changed config. Undo must cross the
        # wire before the command can return failure.
        prelude = (
            "from pm.lock import Facts\n"
            "def fail(self, *args, **kwargs):\n"
            f"    assert 'publication-candidate' in Path({str(config)!r}).read_text()\n"
            "    raise OSError('fixture facts disk full')\n"
            "Facts.record_state = fail\n"
        )
        uv = shutil.which("uv")
        assert uv
        monkeypatch.setattr(client, "runtime_command", lambda path, **kw:
                            worker_command(path, uv, sys.executable, prelude=prelude,
                                           runtime_python=world.runtime_python))
    with pytest.raises(AdmissionRefused, match=f"fixture {failure} disk full"):
        world.command("enable", name="publication-candidate", no_allow_tool_override=True)
    assert {path: path.read_bytes() for path in watched} == before
    assert world.selected() == selected
    world.imports()
    assert receipt.latest()["outcome"] == "failed"


def test_historical_writer_hands_off_without_mutating_config(plugin_world, monkeypatch):
    from hermes_cli.plugins_cmd import _save_enabled_set

    config = plugin_world.home / "config.yaml"
    before = config.read_bytes()
    def takeover():
        raise SystemExit(73)
    monkeypatch.setattr("hermes_cli._old_updater.stop_for_relaunch", takeover)
    with pytest.raises(SystemExit) as exc:
        _save_enabled_set({"must-not-be-saved"})
    assert exc.value.code == 73
    assert config.read_bytes() == before


@pytest.mark.parametrize("surface", ["dashboard", "composite"])
def test_ui_conflict_is_reported_without_changing_selection(plugin_world, surface):
    from hermes_cli import plugins_cmd
    from hermes_cli.plugins_admission import AdmissionRefused
    from pm import paths

    world = plugin_world
    origin, sha = world.origin()
    world.command("install", identifier=origin.as_uri(), ref=sha, no_enable=True, allow_removed=True)
    world.command("enable", name="plugin-worker-proof", no_allow_tool_override=True)
    origin, sha = world.origin(name="conflicting-ui-plugin", pin="2.0")
    world.command("install", identifier=origin.as_uri(), ref=sha, no_enable=True, allow_removed=True)
    watched = [world.home / "config.yaml", paths.runtime_facts_path()]
    before = {path: path.read_bytes() for path in watched}
    if surface == "dashboard":
        result = plugins_cmd.dashboard_set_agent_plugin_enabled("conflicting-ui-plugin", enabled=True)
        assert not result["ok"] and "plugin-proof-dep" in result["error"]
    else:
        with pytest.raises(AdmissionRefused, match="plugin-proof-dep"):
            plugins_cmd._persist_plugin_selection(["plugin-worker-proof", "conflicting-ui-plugin"], {0, 1}, set())
    assert {path: path.read_bytes() for path in watched} == before
    world.imports()