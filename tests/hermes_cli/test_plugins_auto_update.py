"""Unattended plugin updates: catalog installs ride ``plugins.auto_apply``, one updater per plugin, the
Desktop/TUI RPC switch + update-all, and the catalog notice after ``hermes update``."""

from __future__ import annotations

import json
import threading

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class _Row:
    def __init__(self, name, klass, update_available=True):
        self.name, self.klass, self.update_available = name, klass, update_available
        self.needs_fixing = self.reason = self.min_hermes = None
        self.current, self.latest = "a" * 40, "b" * 40

    def to_json(self):
        return {"name": self.name, "class": self.klass, "update_available": self.update_available}


def test_auto_apply_moves_catalog_installs_too(home):
    """A catalog install is re-pinned by auto-apply like a git install; pip rows still never move."""
    from hermes_cli import plugins_cadence as cad

    applied = []
    rows = [_Row("git-plug", "git"), _Row("catalog-plug", "catalog"), _Row("pip-plug", "pip")]
    cad.run_scheduled_check(run_checks_fn=lambda _dir: rows, plugins_dir=home / "plugins",
                            apply_updates_fn=applied.append,
                            config_get=lambda _s, key: True if key == "auto_apply" else 24)
    assert applied == ["git-plug", "catalog-plug"]


def test_second_updater_of_the_same_plugin_is_refused_without_touching_it(home):
    """The gateway's auto-apply tick holds the plugin; a Desktop click on Update changes nothing."""
    from hermes_cli.plugins_auto_update import plugin_update_lock
    from hermes_cli.plugins_cmd_update import dashboard_update_user_plugin

    target = home / "plugins" / "busy-plug"
    target.mkdir()
    (target / "plugin.yaml").write_text("name: busy-plug\nversion: 1.0.0\n", encoding="utf-8")
    held, release = threading.Event(), threading.Event()

    def holder():
        with plugin_update_lock(target):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(5)
        result = dashboard_update_user_plugin("busy-plug")
    finally:
        release.set()
        thread.join(5)
    assert result["ok"] is False and "already being updated" in result["error"]
    assert (target / "plugin.yaml").read_text(encoding="utf-8").startswith("name: busy-plug")


def test_rpc_switch_and_update_all_round_trip(home, monkeypatch):
    """Desktop's switch writes the profile's plugins.auto_apply; Update all reports each plugin and
    never accepts a widened pin on the user's behalf."""
    from hermes_cli import plugins_cmd_update, plugins_updates
    from tui_gateway import server

    def rpc(**params):
        return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})["result"]

    assert rpc(action="auto_update")["auto_apply"] is False
    assert rpc(action="auto_update", enable=True)["auto_apply"] is True
    assert "auto_apply: true" in (home / "config.yaml").read_text(encoding="utf-8")

    rows = [_Row("fresh", "git"), _Row("widened", "catalog"), _Row("pinned", "git", update_available=False)]
    rows[2].reason = "pinned to an exact revision"
    monkeypatch.setattr(plugins_updates, "run_checks", lambda _dir: rows)
    replies = {"fresh": {"ok": True, "name": "fresh", "unchanged": False},
               "widened": {"ok": False, "consent_required": True, "error": "adds tools"}}
    calls = []
    monkeypatch.setattr(plugins_cmd_update, "dashboard_update_user_plugin",
                        lambda name, **kw: calls.append((name, kw)) or replies[name])
    out = rpc(action="update_all")
    assert [(r["name"], r["status"]) for r in out["results"]] == [("fresh", "updated"), ("widened", "needs_review")]
    assert out["skipped"] == [{"name": "pinned", "reason": "pinned to an exact revision"}]
    assert calls == [("fresh", {}), ("widened", {})]  # no accept_capabilities / force


def test_hermes_update_names_catalog_pins_it_moved_and_rearms_the_check(home, monkeypatch, capsys):
    from hermes_cli import plugin_catalog
    from hermes_cli.plugins_auto_update import report_catalog_updates_after_hermes_update

    new_pin = "c" * 40
    entry = plugin_catalog.PluginCatalogEntry(name="cat-plug", repo="https://example.invalid/r", sha=new_pin,
                                              description="d", maintainer="m")
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda catalog_dir=None: [entry])
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({
        "cat-plug": {"catalog": {"name": "cat-plug", "sha": "a" * 40, "pin": "a" * 40}},
        "current": {"catalog": {"name": "cat-plug", "sha": new_pin, "pin": new_pin}}}), encoding="utf-8")
    marker = home / "plugin-update-checks" / "last-run"
    marker.parent.mkdir()
    marker.write_text("1", encoding="utf-8")

    report_catalog_updates_after_hermes_update(home, [])
    assert "1 plugin update(s) from the new catalog in this profile: cat-plug" in capsys.readouterr().out
    assert not marker.exists()


def _held_install_metadata(home):
    """`hermes plugins install cat-plug --ref <older sha>`: pinned, off the catalog pin (empty `pin`), next
    to a catalog install that is simply behind its pin."""
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({
        "held": {"pinned": True, "revision": "a" * 40,
                 "catalog": {"name": "held", "sha": "a" * 40, "pin": ""}},
        "behind": {"pinned": True, "revision": "a" * 40,
                   "catalog": {"name": "behind", "sha": "a" * 40, "pin": "a" * 40}}}), encoding="utf-8")


def test_auto_apply_leaves_a_catalog_install_held_with_ref_alone(home, caplog):
    """A user who rolled a catalog plugin back with --ref keeps it there: the tick moves only the
    install that is merely behind its pin, and names the held one."""
    from hermes_cli import plugins_cadence as cad

    _held_install_metadata(home)
    applied = []
    rows = [_Row("held", "catalog"), _Row("behind", "catalog")]
    with caplog.at_level("INFO"):
        cad.run_scheduled_check(run_checks_fn=lambda _dir: rows, plugins_dir=home / "plugins",
                                apply_updates_fn=applied.append,
                                config_get=lambda _s, key: True if key == "auto_apply" else 24)
    assert applied == ["behind"]
    assert "held with --ref" in caplog.text


def test_update_all_and_hermes_update_notice_skip_a_held_catalog_install(home, monkeypatch, capsys):
    from hermes_cli import plugin_catalog, plugins_cmd_update, plugins_updates
    from hermes_cli.plugins_auto_update import catalog_pins_behind, update_all_unattended

    _held_install_metadata(home)
    monkeypatch.setattr(plugins_updates, "run_checks", lambda _dir: [_Row("held", "catalog"), _Row("behind", "catalog")])
    calls = []
    monkeypatch.setattr(plugins_cmd_update, "dashboard_update_user_plugin",
                        lambda name, **kw: calls.append(name) or {"ok": True, "name": name})
    out = update_all_unattended()
    assert calls == ["behind"]
    assert [s["name"] for s in out["skipped"]] == ["held"] and "--ref" in out["skipped"][0]["reason"]

    entries = [plugin_catalog.PluginCatalogEntry(name=n, repo="https://example.invalid/r", sha="c" * 40,
                                                 description="d", maintainer="m") for n in ("held", "behind")]
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda catalog_dir=None: entries)
    assert catalog_pins_behind(home) == ["behind"]


def test_widened_pin_is_one_review_line_not_a_traceback(home, caplog):
    """The real tick seam lets a widened catalog pin surface as a pending review naming the command."""
    from hermes_cli import plugins_auto_update, plugins_cadence as cad
    from hermes_cli.plugins_cmd_catalog import RepinConsentRequired

    def widened(name):
        raise RepinConsentRequired(name, "c" * 40, {"tools": ["new_tool"]})

    with caplog.at_level("INFO"):
        cad.run_scheduled_check(run_checks_fn=lambda _dir: [_Row("wide", "catalog")], plugins_dir=home / "plugins",
                                apply_updates_fn=widened,
                                config_get=lambda _s, key: True if key == "auto_apply" else 24)
    lines = [r for r in caplog.records if "wide" in r.getMessage()]
    assert [r.getMessage() for r in lines if "review" in r.getMessage()] == [
        "plugin wide has an update that needs your review: run `hermes plugins update wide`"]
    assert not any(r.exc_info for r in lines)


def test_gateway_tick_seam_raises_the_review_instead_of_exiting(home, monkeypatch):
    """`maybe_run_gateway_check`'s default apply hands the refusal back (cmd_update turned it into
    SystemExit, which the tick logged as a failure with a traceback)."""
    from hermes_cli import plugins_auto_update, plugins_cmd_update
    from hermes_cli.plugins_cmd_catalog import RepinConsentRequired

    (home / "plugins" / "wide").mkdir()
    calls = []

    def update_one(name, target, console, *, interactive):
        calls.append((name, target.name, interactive))
        raise RepinConsentRequired(name, "c" * 40, {"tools": ["new_tool"]})

    monkeypatch.setattr(plugins_cmd_update, "update_one", update_one)
    with pytest.raises(RepinConsentRequired):
        plugins_auto_update.apply_unattended("wide")
    assert calls == [("wide", "wide", False)]
