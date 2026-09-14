"""CLI snapshot import must not look successful when every skill is cancelled."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import hermes_cli.skills_hub as cli_hub


def _console(sink: StringIO) -> Console:
    return Console(file=sink, force_terminal=False, color_system=None)


def _import_args(path, *, force: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        skills_action="snapshot",
        snapshot_action="import",
        input=str(path),
        force=force,
    )


def _write_snapshot(tmp_path, skills):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"skills": skills}), encoding="utf-8")
    return path


def _patch_cli_console(monkeypatch, sink: StringIO) -> None:
    monkeypatch.setattr(cli_hub, "_console", _console(sink))


def _cancelled_install(identifier, category="", force=False, console=None, **kwargs):
    c = console or cli_hub._console
    c.print("[dim]Installation cancelled.[/]\n")
    return False


def test_snapshot_import_all_cancelled_exits_nonzero(tmp_path, monkeypatch):
    """Identified skills that all cancel must SystemExit != 0 and not print complete."""
    snap = _write_snapshot(tmp_path, [
        {"name": "alpha", "identifier": "official/alpha"},
        {"name": "beta", "identifier": "official/beta"},
    ])
    monkeypatch.setattr(cli_hub, "do_install", _cancelled_install)
    sink = StringIO()
    _patch_cli_console(monkeypatch, sink)

    with pytest.raises(SystemExit) as excinfo:
        cli_hub.skills_command(_import_args(snap))

    assert excinfo.value.code not in (0, None)
    out = sink.getvalue()
    assert "Snapshot import complete." not in out
    lowered = out.lower()
    assert any(word in lowered for word in ("cancelled", "blocked", "unrestored"))


def test_snapshot_import_partial_success_keeps_complete(tmp_path, monkeypatch):
    """One install + one cancel is CONTROL: complete message, no SystemExit."""
    snap = _write_snapshot(tmp_path, [
        {"name": "alpha", "identifier": "official/alpha"},
        {"name": "beta", "identifier": "official/beta"},
    ])

    def fake_install(identifier, category="", force=False, console=None, **kwargs):
        c = console or cli_hub._console
        if identifier.endswith("alpha"):
            c.print("[bold green]Installed:[/] alpha\n")
            return True
        c.print("[dim]Installation cancelled.[/]\n")
        return False

    monkeypatch.setattr(cli_hub, "do_install", fake_install)
    sink = StringIO()
    _patch_cli_console(monkeypatch, sink)

    cli_hub.skills_command(_import_args(snap))
    assert "Snapshot import complete." in sink.getvalue()


def test_snapshot_import_empty_skills_is_fail_open(tmp_path, monkeypatch):
    """Empty skills list keeps the existing dim message and exits 0."""
    snap = _write_snapshot(tmp_path, [])
    sink = StringIO()
    _patch_cli_console(monkeypatch, sink)

    cli_hub.skills_command(_import_args(snap))
    out = sink.getvalue()
    assert "No skills in snapshot to install." in out
    assert "Snapshot import complete." not in out


def test_snapshot_import_already_installed_skip_is_success(tmp_path, monkeypatch):
    """Re-import of already-installed skills (do_install returns True, no new work)."""
    snap = _write_snapshot(tmp_path, [
        {"name": "alpha", "identifier": "official/alpha"},
    ])

    def fake_install(identifier, category="", force=False, console=None, **kwargs):
        c = console or cli_hub._console
        c.print("[yellow]Warning:[/] 'alpha' is already installed\n")
        return True

    monkeypatch.setattr(cli_hub, "do_install", fake_install)
    sink = StringIO()
    _patch_cli_console(monkeypatch, sink)

    cli_hub.skills_command(_import_args(snap))
    assert "Snapshot import complete." in sink.getvalue()


def test_slash_snapshot_import_all_cancelled_does_not_exit(tmp_path, monkeypatch):
    """In-chat /skills snapshot import prints failure but must not SystemExit."""
    snap = _write_snapshot(tmp_path, [
        {"name": "alpha", "identifier": "official/alpha"},
    ])
    monkeypatch.setattr(cli_hub, "do_install", _cancelled_install)
    sink = StringIO()
    console = _console(sink)

    cli_hub.handle_skills_slash(f"/skills snapshot import {snap}", console=console)
    out = sink.getvalue()
    assert "Snapshot import complete." not in out
    lowered = out.lower()
    assert any(word in lowered for word in ("cancelled", "blocked", "unrestored"))
