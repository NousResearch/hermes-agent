"""``hermes update`` rewrites the launch files it owns before it starts or restarts any gateway.

A service manager starts a gateway from the definition it holds, so the rewrite has to land
after the update steps that change what the generators write (the Node dependency preparation
recreates ``node_modules/.bin``, which the unit's ``PATH`` names) and before the first start.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_windows, source_build, update_cmd, update_completion

STALE_UNIT = "[Service]\nEnvironment=\"PATH=/usr/bin:/bin\"\nRestartSec=30\n"


def _generator_reads_node_bin(root: Path):
    """Stand-in for ``generate_systemd_unit``: its PATH names ``node_modules/.bin`` when it exists.

    The emitted path is fixed: refresh refuses to persist a unit naming a pytest directory."""
    def generate(system=False, run_as_user=None):
        node_bin = "/opt/hermes/node_modules/.bin:" if (root / "node_modules" / ".bin").is_dir() else ""
        return f"[Service]\nEnvironment=\"PATH={node_bin}/usr/bin:/bin\"\nRestartSec=5\n"
    return generate


@pytest.fixture
def systemd_update(tmp_path, monkeypatch):
    """One user unit on disk, a paused gateway, and the real completion ordering around them."""
    root = tmp_path / "checkout"
    root.mkdir()
    unit = tmp_path / "systemd" / "hermes-gateway.service"
    unit.parent.mkdir()
    unit.write_text(STALE_UNIT, encoding="utf-8")
    events: list[tuple] = []

    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gateway_cli, "generate_systemd_unit", _generator_reads_node_bin(root))
    monkeypatch.setattr(gateway_cli, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(
        gateway_cli, "_run_systemctl",
        lambda args, **kw: events.append(("systemctl", *args, unit.read_text(encoding="utf-8"))))

    # The real build ordering; only the external commands are replaced.
    monkeypatch.setattr("hermes_cli.main_install_repair._install_configured_features_missing_deps", lambda r: None)
    monkeypatch.setattr("hermes_cli.memory_provider_migration.migrate_all_homes", lambda: None)
    monkeypatch.setattr("hermes_cli.left_core_migration.migrate_all_homes", lambda: None)
    monkeypatch.setattr(source_build, "source_frontends", lambda r: ("ui-tui",))
    monkeypatch.setattr(source_build, "source_build_env", lambda **kw: {"PATH": ""})
    monkeypatch.setattr(source_build, "source_product_current", lambda *a: False)

    def prepare_node(project_root, workspaces, **kw):
        (project_root / "node_modules" / ".bin").mkdir(parents=True)
        events.append(("node dependencies",))

    monkeypatch.setattr(source_build, "prepare_source_dependencies", prepare_node)
    monkeypatch.setattr(source_build, "build_source_tui", lambda r, env: events.append(("tui build",)))
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda r: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._run_post_update_maintenance", lambda **kw: True)
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda branch: None)

    def start_paused(token):
        events.append(("start gateway", unit.read_text(encoding="utf-8")))
        token["resume_needed"] = False

    monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", start_paused)
    monkeypatch.setattr(update_cmd, "_fleet_restart_skip_reason", lambda plan: None)
    monkeypatch.setattr(update_cmd, "_pending_fleet_restart_needed", lambda **kw: False)
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update",
                        lambda plan, mode: events.append(("fleet restart", unit.read_text(encoding="utf-8"))))
    monkeypatch.setattr(update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", lambda *a, **kw: None)

    request = {"source": str(root), "branch": "main", "desktop": False, "assume_yes": True,
               "gateway_mode": False, "snapshot_id": None, "pre_update_version": None,
               "sibling_snapshots": {}, "plan": None,
               "windows_resume": {"platform": "posix", "resume_needed": True}}
    return root, unit, events, request


@pytest.mark.platforms("linux")
def test_the_paused_gateway_starts_under_the_unit_generated_after_the_node_dependencies(systemd_update):
    """A stale unit is rewritten (and reloaded) once ``node_modules/.bin`` exists, before the gateway
    starts; the start sees exactly what the generator writes for the finished checkout."""
    _root, unit, events, request = systemd_update

    update_completion._complete_selected(request)

    final = gateway_cli.generate_systemd_unit()
    assert "/opt/hermes/node_modules/.bin" in final
    started_under = next(event[1] for event in events if event[0] == "start gateway")
    assert started_under == final == unit.read_text(encoding="utf-8"), events
    names = [event[0] for event in events]
    assert names.index("node dependencies") < names.index("systemctl") < names.index("start gateway")
    assert names.index("start gateway") < names.index("tui build")
    assert next(event for event in events if event[0] == "systemctl")[1] == "daemon-reload"


@pytest.mark.platforms("linux")
def test_one_failed_launch_file_is_a_followup_and_the_other_profiles_are_still_rewritten(
        tmp_path, monkeypatch, capsys):
    from hermes_cli import update_launch_files
    from hermes_cli.config_backend import ConfigBackendUnavailable

    homes = [("default", tmp_path / "default"), ("work", tmp_path / "work"), ("lab", tmp_path / "lab")]
    seen: list[str] = []

    def refresh():
        home = Path(gateway_cli.get_hermes_home())
        seen.append(home.name)
        if home.name == "work":
            raise ConfigBackendUnavailable("the remote config backend is unreachable")

    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: homes)
    monkeypatch.setattr(update_launch_files, "_refreshers", lambda: [("systemd user unit", refresh)])
    monkeypatch.setattr(update_launch_files.os, "geteuid", lambda: 1000)

    update_launch_files.refresh_launch_files()

    assert seen == ["default", "work", "lab"]
    out = capsys.readouterr().out
    assert "Update follow-up 'launch_files' did not finish: systemd user unit for profile work" in out


def test_windows_launcher_scripts_are_rewritten_when_their_contents_drift(tmp_path, monkeypatch):
    """The task XML can match the template while the .cmd/.vbs it runs still name an older command."""
    from hermes_cli import update_launch_files

    script = tmp_path / "gateway-service" / "Hermes_Gateway.cmd"
    script.parent.mkdir()
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "_launcher_settings",
                        lambda: ("C:\\hermes\\venv\\Scripts\\python.exe", "C:\\hermes", "C:\\hermes", ""))
    monkeypatch.setattr(gateway_windows, "_resolve_detached_python",
                        lambda python: (python, Path("C:\\hermes\\venv"), []))

    gateway_windows._write_task_script()
    generated = {path.name: path.read_bytes() for path in script.parent.iterdir()}
    assert gateway_windows.launcher_script_drift() == []

    stale = script.with_suffix(".vbs")
    stale.write_text("sh.Run \"python -m hermes_cli.main gateway run\", 0, False\r\n", encoding="utf-16")
    assert gateway_windows.launcher_script_drift() == [stale.name]

    update_launch_files._refresh_windows_launcher_scripts()

    assert {path.name: path.read_bytes() for path in script.parent.iterdir()} == generated
