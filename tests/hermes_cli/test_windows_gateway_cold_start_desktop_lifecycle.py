"""#76129: post-update Windows cold-start must not steal Desktop-owned lifecycle.

A vestigial Startup/Scheduled-Task autostart is not proof the user wants a
standalone ``gateway run``. When Desktop currently supervises this install's
control plane, the updater must not spawn a competing messaging daemon.

Serve/dashboard are the control plane, not the messaging gateway (#92091).
``looks_like_gateway_command_line`` stays strict; ownership is a separate
predicate.
"""

from __future__ import annotations

from hermes_cli import gateway as hermes_gateway
from hermes_cli import gateway_windows
from hermes_cli import main as cli_main
from hermes_cli import process_identity
from hermes_cli import update_cmd


def _live_serve_ledger_entry() -> dict:
    return {
        "pid": 111,
        "create_time": 1.0,
        "purpose": "serve",
        "install": "abc",
        "spawner_pid": 99,
        "spawner_create": 0.5,
    }


def test_control_plane_argv_is_not_a_gateway():
    from gateway.status import looks_like_gateway_command_line

    serve = "C:\\Hermes\\.venv\\Scripts\\python.exe -m hermes_cli.main serve --host 127.0.0.1"
    run = "C:\\Hermes\\.venv\\Scripts\\python.exe -m hermes_cli.main gateway run"

    assert update_cmd._looks_like_desktop_control_plane(serve) is True
    assert looks_like_gateway_command_line(serve) is False
    assert update_cmd._looks_like_desktop_control_plane(run) is False
    assert looks_like_gateway_command_line(run) is True


def test_control_plane_classifier_is_token_based_not_substring():
    """#90778/#91869 class: flag values and lookalike tokens must not read
    as a control plane. The salvage swapped the original substring check
    for the parser-derived subcommand classifier."""
    py = "C:\\Hermes\\.venv\\Scripts\\python.exe -m hermes_cli.main"
    # "dashboard" as a FLAG VALUE, real subcommand is chat
    assert update_cmd._looks_like_desktop_control_plane(f"{py} -m dashboard chat") is False
    # "--preserve-cache" contains "serve"; real subcommand is kanban
    assert (
        update_cmd._looks_like_desktop_control_plane(f"{py} kanban --preserve-cache")
        is False
    )
    # profile selector before the real subcommand still classifies correctly
    assert (
        update_cmd._looks_like_desktop_control_plane(f"{py} --profile serve dashboard")
        is True
    )
    # dashboard as the real subcommand
    assert update_cmd._looks_like_desktop_control_plane(f"{py} dashboard") is True
    # undeterminable subcommand → NOT a control plane (never guess ownership)
    assert update_cmd._looks_like_desktop_control_plane("python.exe -c import time") is False


def test_ledger_live_serve_with_live_spawner_owns_lifecycle(monkeypatch):
    monkeypatch.setattr(
        process_identity, "ledger_entries", lambda **_k: [_live_serve_ledger_entry()]
    )
    monkeypatch.setattr(process_identity, "spawner_is_dead", lambda _e: False)
    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: [])

    assert update_cmd._desktop_owns_gateway_lifecycle() is True


def test_orphaned_control_plane_does_not_own_lifecycle(monkeypatch):
    monkeypatch.setattr(
        process_identity, "ledger_entries", lambda **_k: [_live_serve_ledger_entry()]
    )
    monkeypatch.setattr(process_identity, "spawner_is_dead", lambda _e: True)
    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: [])

    assert update_cmd._desktop_owns_gateway_lifecycle() is False


def test_pause_skips_cold_start_plan_when_desktop_owns_lifecycle(monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        hermes_gateway, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)

    assert update_cmd._pause_windows_gateways_for_update() is None


def test_pause_still_cold_starts_when_autostart_and_no_desktop_owner(monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        hermes_gateway, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)

    token = update_cmd._pause_windows_gateways_for_update()

    assert token == {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "cold_start_if_installed": True,
    }


def test_cold_start_aborts_when_desktop_owns_lifecycle(monkeypatch):
    spawned = []
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(
        gateway_windows, "_spawn_detached", lambda: spawned.append(1) or 4242
    )

    update_cmd._cold_start_windows_gateway_after_update()

    assert spawned == []
