"""Recovery of a ``state == "failed"`` migration manifest after manual rollback, and closing guidance in
``_manual_rollback_steps``.
"""

from __future__ import annotations

import hashlib
import plistlib
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.gateway_migrate_backends_apply.test_engine import _StuckGatewayHost
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args, _config_files
from tests.hermes_cli.test_gateway_migrate_backends_launchd import (
    _SYSTEM_LABEL,
    _SYSTEM_PLIST,
    LaunchdHost,
    SystemLaunchDaemonHost,
)


def _make_plist(tmp_path: Path, name: str = "local.hermes-dashboard.plist") -> Path:
    path = tmp_path / "LaunchAgents" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps({
        "Label": "local.hermes-dashboard",
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProgramArguments": ["/Users/u/.local/bin/hermes", "dashboard", "--no-open", "--host", "0.0.0.0"],
    }))
    return path


def _failed_system_launchdaemon_manifest(home: Path, plist_path: str = _SYSTEM_PLIST, *,
                                         was_active: bool = True) -> dict:
    """A realistic ``state == 'failed'`` manifest for a system LaunchDaemon after incomplete rollback."""
    folder = home / engine.BACKUP_DIR
    folder.mkdir(mode=0o700, exist_ok=True)
    cfg_bak = folder / "config.yaml.bak"
    env_bak = folder / "env.bak"
    cfg_bytes = (home / "config.yaml").read_bytes()
    env_bytes = (home / ".env").read_bytes()
    cfg_bak.write_bytes(cfg_bytes)
    env_bak.write_bytes(env_bytes)
    cfg_after_digest = hashlib.sha256(cfg_bytes + b"\nwebapp:\n  enabled: true\n").hexdigest()
    env_after_digest = hashlib.sha256(env_bytes + b"\nHERMES_DASHBOARD_BASIC_AUTH_USERNAME=spike\n").hexdigest()
    return {
        "version": 1,
        "kind": "apply",
        "state": "failed",
        "step": "verify",
        "message": "verify failed; rollback incomplete: re-enabled the old backend: Operation not permitted",
        "home": str(home),
        "old_address": "0.0.0.0:9119",
        "new_address": "0.0.0.0:9119",
        "dial": {
            "host": "127.0.0.1",
            "old_host": "127.0.0.1",
            "verify_hosts": ["127.0.0.1"],
            "old_port": 9119,
            "new_port": 9119,
        },
        "started_at": 100.0,
        "finished_at": 105.0,
        "pid": 999999,
        "pid_start_time": 1,
        "entered": ["manifest", "write_config", "gateway_service", "stop_old_backend", "start_gateway", "verify"],
        "steps": [
            {"name": "preflight", "state": "done"},
            {"name": "manifest", "state": "done"},
            {"name": "webapp_bundle", "state": "done"},
            {"name": "write_config", "state": "done"},
            {"name": "gateway_service", "state": "done"},
            {"name": "stop_old_backend", "state": "done"},
            {"name": "start_gateway", "state": "done"},
            {"name": "verify", "state": "failed"},
        ],
        "backups": {
            "config": str(cfg_bak),
            "env": str(env_bak),
            "launcher_env": None,
            "config_mode": 0o600,
            "env_mode": 0o600,
        },
        "files_after_apply": {"config.yaml": cfg_after_digest, ".env": env_after_digest},
        "gateway": {"installed": False, "enabled": False, "active": False, "kind": "launchd",
                    "unit": "ai.hermes.gateway"},
        "linger": None,
        "old_backend": {
            "supervisor": "launchd",
            "unit": _SYSTEM_LABEL,
            "scope": "system",
            "unit_path": plist_path,
            "was_active": was_active,
            "was_loaded": True,
        },
        "old_backend_enabled": True,
        "old_backend_disabled": True,
        "old_backend_stopped": True,
    }


def _invoke(host, mode: str) -> int:
    if mode == "restore":
        return engine.cmd_restore(host, _args(apply=False, restore=True))
    return engine.cmd_apply(host, _args())


# --------------------------------------------------------------------------- closing step guidance


@pytest.mark.parametrize("supervisor", ["systemd", "launchd", "manual"])
def test_manual_rollback_steps_close_guidance_names_restore_address_and_manifest(home, tmp_path, supervisor):
    if supervisor == "launchd":
        host = LaunchdHost(home, _make_plist(tmp_path))
    else:
        host = FakeHost(home, supervisor=supervisor)
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("cannot start old backend")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    closing = steps[-1]
    assert closing.startswith("# ")
    assert "0.0.0.0:9119" in closing
    assert "hermes -p default gateway migrate --backends --restore" in closing
    assert str(engine.manifest_path(home)) in closing


def test_manual_rollback_steps_close_guidance_pins_non_standard_root(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda _root: False)
    host = FakeHost(home)
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("cannot start")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    closing = engine.status_payload(host)["manual_steps"][-1]
    assert closing.startswith("# ")
    assert "HERMES_HOME=" in closing
    assert "gateway migrate --backends --restore" in closing
    assert str(engine.manifest_path(home)) in closing


def test_manual_rollback_steps_for_pre_stop_failure_still_include_close_guidance(home):
    host = FakeHost(home)

    def partial_install():
        host._change("install_gateway")
        host.gw["installed"] = True
        raise RuntimeError("install failed partway")

    host.install_gateway = partial_install
    host.fail_at["uninstall_gateway"] = RuntimeError("uninstall blocked")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert not any("systemctl" in s for s in steps)
    closing = steps[-1]
    assert closing.startswith("# ")
    assert "hermes -p default gateway migrate --backends --restore" in closing
    assert str(engine.manifest_path(home)) in closing


def test_manual_rollback_steps_for_done_manifest_tell_operator_to_move_record_aside(home, capsys):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    (home / "config.yaml").write_text(
        (home / "config.yaml").read_text(encoding="utf-8-sig") + "# edited after move\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    assert engine.restore(host, yes=True) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    closing = engine._manual_rollback_steps({**engine.read_manifest(home), "home": str(home)})[-1]
    assert closing.startswith("# ")
    assert str(engine.manifest_path(home)) in closing
    assert "--restore" not in closing
    assert closing in out


# --------------------------------------------------------------------------- reconciling manually-recovered failed manifests


@pytest.mark.parametrize("mode", ["restore", "apply"])
def test_reconciles_manually_recovered_system_launchdaemon_manifest(home, mode):
    engine.write_manifest(home, _failed_system_launchdaemon_manifest(home))
    host = SystemLaunchDaemonHost(home)
    host.old = {"enabled": True, "active": True}  # recovered with sudo mv/enable/bootstrap
    host.fail_at["old_enable"] = RuntimeError("Operation not permitted")
    host.fail_at["old_start"] = RuntimeError("Operation not permitted")

    assert _invoke(host, mode) == engine.EXIT_ROLLED_BACK

    manifest = engine.read_manifest(home)
    assert manifest["state"] == "rolled_back"
    assert manifest["old_backend_disabled"] is True  # unprivileged enable was skipped, not falsely cleared
    assert manifest["old_backend_stopped"] is False
    assert "old_start" not in host.calls
    status = engine.status_payload(host)
    assert status["state"] == "rolled_back" and status["manual_steps"] == []

    # Subsequent --apply is no longer trapped in failed-manifest recovery.
    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_UNSUPPORTED


@pytest.mark.parametrize("mode", ["restore", "apply"])
@pytest.mark.parametrize("lose_snapshot", [False, True])
def test_reconciles_manually_recovered_manual_backend_without_respawning(home, mode, lose_snapshot):
    host = FakeHost(home, supervisor="manual")
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("cannot respawn")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"

    if lose_snapshot:
        Path(engine.read_manifest(home)["backups"]["launcher_env"]).unlink(missing_ok=True)
    host.same_process = lambda pid, start_time: False  # operator started a brand-new manual process
    host.live_manual_backend = lambda port: None
    host.old["active"] = True  # operator started the manual backend by hand
    host.calls.clear()

    assert _invoke(host, mode) == engine.EXIT_ROLLED_BACK

    manifest = engine.read_manifest(home)
    assert manifest["state"] == "rolled_back"
    assert "old_start" not in host.calls
    assert manifest["backups"]["launcher_argv"] is None
    assert manifest["backups"]["launcher_env"] is None
    assert not list((home / engine.BACKUP_DIR).glob("launcher-*.json"))


@pytest.mark.parametrize("supervisor", ["systemd", "launchd"])
def test_still_re_enables_user_service_when_operator_only_started_it(home, tmp_path, supervisor):
    if supervisor == "launchd":
        plist = _make_plist(tmp_path)
        host = LaunchdHost(home, plist)
    else:
        plist = None
        host = FakeHost(home)
    host.gateway_lies = True
    host.fail_at["old_enable"] = RuntimeError("transient enable error")
    host.fail_at["old_start"] = RuntimeError("transient start error")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"

    # Operator moved the plist back (for launchd) and started the service, but forgot to re-enable it.
    host.fail_at.clear()
    if plist is not None:
        Path(f"{plist}.disabled").rename(plist)
    host.old["active"] = True
    host.old["enabled"] = False
    host.calls.clear()

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLED_BACK

    assert "old_enable" in host.calls
    assert "old_start" not in host.calls
    assert host.old["enabled"] is True
    assert engine.read_manifest(home)["state"] == "rolled_back"


@pytest.mark.parametrize("supervisor", ["systemd", "launchd"])
@pytest.mark.parametrize("mode", ["restore", "apply"])
def test_user_service_re_enable_failure_keeps_manifest_failed_without_restarting(
    home, tmp_path, capsys, supervisor, mode
):
    if supervisor == "launchd":
        plist = _make_plist(tmp_path)
        host = LaunchdHost(home, plist)
        expected_enable = "launchctl enable gui/501/"
    else:
        plist = None
        host = FakeHost(home)
        expected_enable = "systemctl --user enable "
    host.gateway_lies = True
    host.fail_at["old_enable"] = RuntimeError("cannot enable")
    host.fail_at["old_start"] = RuntimeError("cannot start")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"

    # Operator brought the user service back up by hand, but re-enabling still fails.
    if plist is not None:
        Path(f"{plist}.disabled").rename(plist)
    host.old.update(active=True, enabled=False)
    host.calls.clear()
    capsys.readouterr()

    assert _invoke(host, mode) == engine.EXIT_ROLLBACK_INCOMPLETE

    manifest = engine.read_manifest(home)
    assert manifest["state"] == "failed"
    assert manifest["old_backend_disabled"] is True
    assert host.old["enabled"] is False
    assert "old_enable" in host.calls
    assert "old_start" not in host.calls
    out = capsys.readouterr().out
    assert "Rollback incomplete" in out
    status = engine.status_payload(host)
    assert status["state"] == "failed"
    assert any(expected_enable in step for step in status["manual_steps"])
    assert expected_enable in out
    assert str(engine.manifest_path(home)) in status["manual_steps"][-1]
    assert status["manual_steps"][-1] in out

    # Once enable recovers, retry re-enables and closes the manifest without restarting the running backend.
    host.fail_at.pop("old_enable")
    host.calls.clear()
    assert _invoke(host, mode) == engine.EXIT_ROLLED_BACK
    assert "old_enable" in host.calls
    assert "old_start" not in host.calls
    assert host.old["enabled"] is True
    closed = engine.read_manifest(home)
    assert closed["state"] == "rolled_back"
    assert closed["old_backend_disabled"] is False


# --------------------------------------------------------------------------- negative controls (never falsely succeed)


def test_never_reconciles_while_disabled_plist_remains_on_disk(home, tmp_path):
    plist = _make_plist(tmp_path)
    disabled = Path(f"{plist}.disabled")
    plist.rename(disabled)
    engine.write_manifest(home, _failed_system_launchdaemon_manifest(home, str(plist)))
    host = SystemLaunchDaemonHost(home, plist)
    host.old = {"enabled": True, "active": True}
    host.fail_at["old_enable"] = RuntimeError("Operation not permitted")
    host.fail_at["old_start"] = RuntimeError("Operation not permitted")

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert disabled.exists() and not plist.exists()
    assert engine.read_manifest(home)["state"] == "failed"


def test_never_reconciles_while_gateway_still_holds_port(home):
    host = _StuckGatewayHost(home)
    host.gateway_lies = True
    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"

    host.old["active"] = True  # even if something answers, gateway still holds port 9119
    host.calls.clear()

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert "old_start" not in host.calls
    assert engine.read_manifest(home)["state"] == "failed"


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_never_reconciles_when_file_restore_fails(home, name):
    engine.write_manifest(home, _failed_system_launchdaemon_manifest(home))
    path = home / name
    marker = "# operator edit that must not be clobbered\n"
    path.write_text(path.read_text(encoding="utf-8-sig") + marker, encoding="utf-8")
    before = _config_files(home)

    host = SystemLaunchDaemonHost(home)
    host.old = {"enabled": True, "active": True}
    host.fail_at["old_enable"] = RuntimeError("Operation not permitted")
    host.fail_at["old_start"] = RuntimeError("Operation not permitted")

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"
    assert _config_files(home) == before


@pytest.mark.parametrize("probe_mode", ["down", "health_missing"])
def test_never_reconciles_when_old_backend_is_not_answering(home, probe_mode):
    engine.write_manifest(home, _failed_system_launchdaemon_manifest(home))
    host = SystemLaunchDaemonHost(home)
    host.old = {"enabled": True, "active": probe_mode == "health_missing"}
    if probe_mode == "health_missing":
        host.http_status = lambda h, p: {"health": "missing"}
    host.fail_at["old_enable"] = RuntimeError("Operation not permitted")
    host.fail_at["old_start"] = RuntimeError("Operation not permitted")

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"


def test_never_reconciles_idle_backend_when_enable_fails(home):
    engine.write_manifest(home, _failed_system_launchdaemon_manifest(home, was_active=False))
    host = SystemLaunchDaemonHost(home, running=False)
    host.old = {"enabled": False, "active": False}
    host.fail_at["old_enable"] = RuntimeError("Operation not permitted")
    host.fail_at["old_start"] = RuntimeError("Operation not permitted")

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert engine.read_manifest(home)["state"] == "failed"
