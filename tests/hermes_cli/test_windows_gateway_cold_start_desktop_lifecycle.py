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

import json

from hermes_cli import gateway as hermes_gateway
from hermes_cli import gateway_windows
from hermes_cli import main as cli_main
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import process_identity
from hermes_cli import update_cmd
import hermes_cli.update_cmd_windows as update_cmd_windows


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
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        hermes_gateway, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "attested_death_generation", lambda **_k: None)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: True)

    assert update_cmd._pause_windows_gateways_for_update() is None


def test_pause_still_cold_starts_when_autostart_and_no_desktop_owner(monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        hermes_gateway, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: False)

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
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "attested_death_generation", lambda **_k: None)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(
        gateway_windows, "_spawn_detached", lambda: spawned.append(1) or 4242
    )

    update_cmd._cold_start_windows_gateway_after_update()

    assert spawned == []


def test_attested_dead_gateway_survives_desktop_ownership_and_marker_is_consumed_on_spawn(
    monkeypatch, tmp_path, capsys
):
    """#109538: the Desktop hand-off can kill the running gateway moments before update
    discovery runs, so a dead start attestation is the surviving "a gateway was up"
    evidence. It must keep the plan AND survive the spawn-time ownership re-check.
    Once the spawn happens the marker is consumed, so a stale crash marker cannot
    re-authorize a cold start against Desktop ownership on a later update."""
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(hermes_gateway, "find_windows_gateway_services", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: True)
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")
    marker = tmp_path / "state" / "gateway.start-attestation.json"

    token = update_cmd._pause_windows_gateways_for_update()

    generation = token.pop("attested_generation")
    assert generation == json.loads(marker.read_text(encoding="utf-8"))["generation"]
    assert token == {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "cold_start_if_installed": True,
    }
    token["attested_generation"] = generation
    assert marker.exists()  # plan-time probe is read-only

    spawned = []
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: spawned.append(1) or 4242)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: [4242])
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)

    # A token written by pre-generation code and resumed across this very update carries no
    # ``attested_generation`` key: the marker is probed again rather than the spawn skipped.
    legacy_token = {k: v for k, v in token.items() if k != "attested_generation"}
    assert update_cmd._cold_start_windows_gateway_after_update(legacy_token) is True
    assert spawned == [1]
    assert "Gateway started via cold-start after update (PID: 4242)" in capsys.readouterr().out
    assert not marker.exists()  # consumed by the spawn
    assert gateway_windows.attested_death_generation(current_pids=[]) is None


def test_cold_start_is_authorized_by_the_token_generation_not_the_mutable_marker(
    monkeypatch, tmp_path, capsys
):
    """#110020 review (a): the marker is a one-shot that a concurrent ``hermes gateway status``
    consumes between plan and execution. The spawn must still happen (authority lives on the
    token), and a *newer* marker written by a concurrent ``hermes gateway start`` must not be
    consumed as if it were ours."""
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(hermes_gateway, "find_windows_gateway_services", lambda **_k: [])
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: True)
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")
    marker = tmp_path / "state" / "gateway.start-attestation.json"
    token = update_cmd._pause_windows_gateways_for_update()
    assert token["attested_generation"]

    # Concurrent ``hermes gateway status`` consumed the marker...
    assert gateway_windows.check_start_attestation(current_pids=[]) is not None
    assert not marker.exists()
    # ...and a concurrent ``hermes gateway start`` wrote a fresh one for its own PID.
    gateway_windows._write_start_attestation([777], "direct spawn (PID 777)")
    newer = json.loads(marker.read_text(encoding="utf-8"))["generation"]
    assert newer != token["attested_generation"]

    spawned = []
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: spawned.append(1) or 4242)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: [4242])
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)

    assert update_cmd._cold_start_windows_gateway_after_update(token) is True
    assert spawned == [1]  # authorized by the token, not by the (consumed) marker
    assert json.loads(marker.read_text(encoding="utf-8"))["generation"] == newer  # not ours to consume


def _running_beta_pause_fixture(monkeypatch, tmp_path):
    """Windows update with ``beta`` (PID 777) running and the default profile home at ``tmp_path``."""
    from types import SimpleNamespace
    import hermes_cli.profiles as profiles_mod

    homes = {"default": tmp_path, "beta": tmp_path / "profiles" / "beta"}
    homes["beta"].mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_desktop_owns_gateway_lifecycle", lambda: True)
    beta = SimpleNamespace(pid=777, profile="beta")
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({777: beta}, [], set(), [777]))
    monkeypatch.setattr(update_cmd_windows, "_request_socket_pauses", lambda *a: ({"beta": 777}, [777], []))
    monkeypatch.setattr(cli_main, "_venv_launcher_ancestors", lambda pids: [])
    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", lambda pids, timeout: set())
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(profiles_mod, "profiles_to_serve", lambda multiplex: [(n, h) for n, h in homes.items() if n in ("default", "beta")])
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: homes[name])
    # Resume side.
    monkeypatch.setattr(cli_main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(hermes_gateway, "launch_detached_profile_gateway_restart", lambda p, o: True)
    ready_probes: list = []
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: ready_probes.append(k) or [4242])
    homes["_ready_probes"] = ready_probes
    return homes


def test_dead_attested_default_is_cold_started_beside_running_beta(monkeypatch, tmp_path, capsys):
    """#110959: the all-or-nothing plan never ran while ``beta`` was alive, so a default gateway
    that died after a ✓ stayed down after the update. Its dead attestation must become a per-profile
    cold-start obligation on the token; resume spawns it under the default home and consumes only
    that generation, while ``beta`` still goes through the ordinary relaunch."""
    homes = _running_beta_pause_fixture(monkeypatch, tmp_path)
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")
    marker = tmp_path / "state" / "gateway.start-attestation.json"
    generation = json.loads(marker.read_text(encoding="utf-8"))["generation"]

    token = update_cmd._pause_windows_gateways_for_update()

    assert token["profiles"] == {"beta": 777}
    assert token["cold_start_profiles"] == {"default": generation}
    assert marker.exists()  # plan-time probe is read-only

    spawned = []
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda **k: spawned.append(k) or 4242)
    update_cmd._resume_windows_gateways_after_update(token)

    assert spawned == [{"home": homes["default"]}]
    # Readiness is probed in the default profile's own home: live ``beta`` must not vouch for it.
    assert {"home": homes["default"]} in homes["_ready_probes"]
    # The authorizing generation is consumed and the NEW PID is attested in the same profile home,
    # so a death after this CLI exits stays visible to the next update.
    reattested = json.loads(marker.read_text(encoding="utf-8"))
    assert (reattested["pids"], reattested["generation"] != generation) == ([4242], True)
    assert "cold_start_profiles" not in token
    assert token["relaunched_profiles"] == ["beta"]
    assert token["resume_needed"] is False
    out = capsys.readouterr().out
    assert "Gateway profile default started via cold-start after update (PID: 4242)" in out


def test_no_attested_profile_leaves_the_pause_token_unchanged(monkeypatch, tmp_path):
    _running_beta_pause_fixture(monkeypatch, tmp_path)

    token = update_cmd._pause_windows_gateways_for_update()

    assert "cold_start_profiles" not in token
    assert token["profiles"] == {"beta": 777}
    spawned = []
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda **k: spawned.append(k) or 4242)
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)
    update_cmd._resume_windows_gateways_after_update(token)
    assert spawned == []
    assert token["relaunched_profiles"] == ["beta"]
