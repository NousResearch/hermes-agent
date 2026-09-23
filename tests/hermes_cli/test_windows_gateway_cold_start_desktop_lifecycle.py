"""#76129: post-update Windows cold-start must not steal Desktop-owned lifecycle.

A vestigial Startup/Scheduled-Task autostart is not proof the user wants a
standalone ``gateway run``. When Desktop currently supervises this install's
control plane, the updater must not spawn a competing messaging daemon.

Serve/dashboard are the control plane, not the messaging gateway (#92091).
``looks_like_gateway_command_line`` stays strict; ownership is a separate
predicate.

#109538: ownership alone must not hide a gateway that *died*. The Desktop
hand-off exits the app before the updater starts and can kill the running
gateway in those same seconds, so discovery finds no live PID while a start
attestation still vouches for the dead one. In that case the cold-start
survives both the plan-time and the spawn-time ownership check — the Desktop
does not restart the messaging gateway itself.
"""

from __future__ import annotations


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


