import subprocess

from hermes_cli.main_platform_setup import _whatsapp_install_bridge
from hermes_cli.web_routers.messaging import _ensure_whatsapp_bridge_dependencies


def test_explicit_maintenance_paths_refresh_and_stamp_whatsapp_dependencies(
    tmp_path, monkeypatch
):
    import hermes_cli.main as hm
    import hermes_constants

    checkout = tmp_path / "checkout"
    bridge_dir = checkout / "scripts" / "whatsapp-bridge"
    checkout.mkdir()
    (checkout / "package.json").write_text("{}", encoding="utf-8")
    (bridge_dir / "node_modules").mkdir(parents=True)
    (bridge_dir / "package.json").write_text(
        '{"dependencies": {}}', encoding="utf-8"
    )
    (bridge_dir / "package-lock.json").write_text(
        '{"lockfileVersion": 3}', encoding="utf-8"
    )

    monkeypatch.setattr(hm, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(
        hermes_constants,
        "find_node_executable",
        lambda _name: "/usr/bin/npm",
    )
    monkeypatch.setattr(
        hermes_constants,
        "with_hermes_node_path",
        lambda _env=None: {},
    )

    installs = []
    phase = ["cli"]

    def fake_run(*_args, **_kwargs):
        installs.append(phase[0])
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert _whatsapp_install_bridge(bridge_dir) is True
    stamp = bridge_dir / "node_modules" / ".hermes-pkg-hash"
    cli_stamp = stamp.read_text(encoding="utf-8").strip()
    assert cli_stamp

    phase[0] = "dashboard"
    (bridge_dir / "package.json").write_text(
        '{"dependencies": {"a": "1"}}', encoding="utf-8"
    )
    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    dashboard_stamp = stamp.read_text(encoding="utf-8").strip()
    assert dashboard_stamp and dashboard_stamp != cli_stamp

    phase[0] = "update"
    (bridge_dir / "package-lock.json").write_text(
        '{"lockfileVersion": 3, "packages": {"a": {}}}', encoding="utf-8"
    )

    def fake_deterministic_install(*_args, **_kwargs):
        installs.append(phase[0])
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(hm, "_run_npm_install_deterministic", fake_deterministic_install)
    monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: "/usr/bin/npm")
    monkeypatch.setattr(hm, "_nixos_build_env", lambda: {})
    monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda _root: False)
    monkeypatch.setattr(
        hermes_constants,
        "get_default_hermes_root",
        lambda: tmp_path / "hermes-home",
    )
    monkeypatch.setattr(
        "tools.browser_tool_install.warm_agent_browser_npx_cache",
        lambda: True,
    )
    from hermes_cli.update_cmd_deps import _update_node_dependencies

    assert _update_node_dependencies() == []
    update_stamp = stamp.read_text(encoding="utf-8").strip()
    assert update_stamp and update_stamp != dashboard_stamp
    assert installs == ["cli", "dashboard", "update"]
