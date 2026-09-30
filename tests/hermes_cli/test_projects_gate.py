"""The ``projects.enabled`` config toggle — one gate, three surfaces (#58588).

The projects feature reaches users through the ``hermes project`` CLI verb,
the ``projects.*`` JSON-RPCs and the ``project`` model toolset folded into GUI
sessions. None of those surfaces may expose the feature when the profile's
config turns it off; the predicate lives in one module (``projects_gate``).
"""

from __future__ import annotations

import pytest

import tui_gateway.server as server
from hermes_cli import projects_gate


@pytest.fixture(autouse=True)
def _home_with_config(tmp_path, monkeypatch):
    """A hermes home whose config.yaml the test rewrites per assertion."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_HOME", raising=False)
    token = set_hermes_home_override(tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    yield tmp_path
    reset_hermes_home_override(token)


def _write_config(home, text: str) -> None:
    (home / "config.yaml").write_text(text, encoding="utf-8")


# ── the predicate ────────────────────────────────────────────────────────────


def test_enabled_by_default_and_through_a_badly_shaped_section(tmp_path):
    _write_config(tmp_path, "")
    assert projects_gate.projects_enabled() is True

    _write_config(tmp_path, "projects:\n  enabled: true\n")
    assert projects_gate.projects_enabled() is True

    # A mis-shaped section coerces to enabled rather than crashing a surface.
    _write_config(tmp_path, 'projects: "off"\n')
    assert projects_gate.projects_enabled() is True


def test_disabled_by_config_file(tmp_path):
    _write_config(tmp_path, "projects:\n  enabled: false\n")
    assert projects_gate.projects_enabled() is False

    # Truthy string spellings a user might write still mean on.
    _write_config(tmp_path, 'projects:\n  enabled: "yes"\n')
    assert projects_gate.projects_enabled() is True
    _write_config(tmp_path, 'projects:\n  enabled: "off"\n')
    assert projects_gate.projects_enabled() is False


def test_disabled_message_names_the_toggle():
    msg = projects_gate.projects_disabled_message()
    assert "projects.enabled" in msg


# ── the CLI verb ─────────────────────────────────────────────────────────────


def test_cli_verb_answers_disabled_and_skips_the_db(tmp_path, monkeypatch, capsys):
    from hermes_cli import projects_cmd

    _write_config(tmp_path, "projects:\n  enabled: false\n")

    def _fail_open_db(*_a, **_kw):
        raise AssertionError("projects.db must not be opened while disabled")

    monkeypatch.setattr("hermes_cli.projects_db.connect_closing", _fail_open_db)

    assert projects_cmd.projects_command(_args("list")) == 1
    assert "projects.enabled" in capsys.readouterr().err


def test_cli_verb_still_dispatches_when_enabled(tmp_path, capsys):
    from hermes_cli import projects_cmd

    args = _args("list")
    assert projects_cmd.projects_command(args) == 0
    assert "projects.enabled" not in capsys.readouterr().err


def _args(action: str):
    """Parsed argv for one ``hermes project <action>`` dispatch."""
    import argparse

    from hermes_cli import projects_cmd

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    p = projects_cmd.build_parser(sub)
    p.set_defaults(func=projects_cmd.projects_command)
    return parser.parse_args(["project", action])


# ── the projects.* JSON-RPCs ─────────────────────────────────────────────────


def _rpc(method: str, params: dict | None = None) -> dict:
    handler = server._methods[method]
    return handler(1, params or {})


def test_rpc_surfaces_refused_before_touching_the_db(tmp_path, monkeypatch):
    _write_config(tmp_path, "projects:\n  enabled: false\n")

    def _fail_open_db(*_a, **_kw):
        raise AssertionError("projects.db must not be opened while disabled")

    monkeypatch.setattr("hermes_cli.projects_db.connect_closing", _fail_open_db)

    resp = _rpc("projects.list")
    assert resp["error"]["code"] == 5061
    assert "disabled by config" in resp["error"]["message"]

    # The read-only tree handlers (methods_config._projects_handler) refuse too.
    resp = _rpc("projects.tree")
    assert resp["error"]["code"] == 5061


def test_rpc_surfaces_work_when_enabled(tmp_path):
    listing = _rpc("projects.list")
    assert "error" not in listing, listing.get("error")
    tree = _rpc("projects.tree")
    assert "error" not in tree, tree.get("error")


def test_config_get_exposes_the_gate(tmp_path):
    # The desktop probes `config.get {key: projects_enabled}` on connect; the
    # getter must answer for both settings without touching projects.db.
    probe = _rpc("config.get", {"key": "projects_enabled"})
    assert probe["result"]["value"] == "on"

    _write_config(tmp_path, "projects:\n  enabled: false\n")
    probe = _rpc("config.get", {"key": "projects_enabled"})
    assert probe["result"]["value"] == "off"


# ── the project model toolset ─────────────────────────────────────────────────


def test_gui_surface_toolsets_drop_project_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    assert "project" in server._gui_surface_toolsets("desktop")
    assert "project" in server._gui_surface_toolsets("tui")

    _write_config(tmp_path, "projects:\n  enabled: false\n")
    assert "project" not in server._gui_surface_toolsets("desktop")
    assert "project" not in server._gui_surface_toolsets("tui")
    # The desktop-only toolsets are untouched.
    assert "desktop_ui" in server._gui_surface_toolsets("desktop")


def test_status_resolver_skips_the_db_when_disabled(tmp_path, monkeypatch):
    def _fail_open_db(*_a, **_kw):
        raise AssertionError("projects.db must not be opened while disabled")

    monkeypatch.setattr("hermes_cli.projects_db.connect_closing", _fail_open_db)
    _write_config(tmp_path, "projects:\n  enabled: false\n")

    assert server._project_info_for_cwd(str(tmp_path)) is None
