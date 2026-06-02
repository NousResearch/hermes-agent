from __future__ import annotations

from pathlib import Path


def test_normalize_allowed_hosts_accepts_urls_ports_and_lists():
    from hermes_cli.dashboard_service import normalize_allowed_hosts

    assert normalize_allowed_hosts(
        ["https://node.tailnet.ts.net/hermes", "dashboard.example.com:443", "NODE.tailnet.ts.net"]
    ) == ("node.tailnet.ts.net", "dashboard.example.com")


def test_generate_systemd_unit_uses_dashboard_service_command(monkeypatch, tmp_path, _isolate_hermes_home):
    import hermes_cli.dashboard_service as svc

    monkeypatch.setattr(svc, "get_python_path", lambda: "/usr/bin/python3")
    monkeypatch.setattr(svc, "_stable_service_working_dir", lambda: str(tmp_path))
    monkeypatch.setattr(svc, "_detect_venv_dir", lambda: tmp_path / "venv")
    monkeypatch.setattr(svc, "_build_service_path_dirs", lambda: ["/opt/hermes/bin"])

    unit = svc.generate_systemd_unit(
        svc.DashboardServiceOptions(
            host="127.0.0.1",
            port=9120,
            tui=True,
            allowed_hosts=("node.tailnet.ts.net",),
            public_url="https://node.tailnet.ts.net",
        )
    )

    assert "Description=Hermes Agent Dashboard" in unit
    assert "dashboard --host 127.0.0.1 --port 9120 --no-open --skip-build --tui" in unit
    assert "--allowed-hosts node.tailnet.ts.net" in unit
    assert 'Environment="HERMES_DASHBOARD_PUBLIC_URL=https://node.tailnet.ts.net"' in unit
    assert 'Environment="HERMES_DASHBOARD_ALLOWED_HOSTS=node.tailnet.ts.net"' in unit
    assert "NoNewPrivileges=true" in unit
    assert "UMask=0077" in unit


def test_service_options_roundtrip(_isolate_hermes_home):
    import hermes_cli.dashboard_service as svc

    options = svc.DashboardServiceOptions(
        host="127.0.0.1",
        port=9120,
        tui=True,
        allowed_hosts=("node.tailnet.ts.net",),
        public_url="https://node.tailnet.ts.net",
    )
    svc.save_service_options(options)

    assert svc.load_service_options() == options


def test_generate_launchd_plist_has_profiled_dashboard_args(monkeypatch, tmp_path, _isolate_hermes_home):
    import hermes_cli.dashboard_service as svc

    monkeypatch.setattr(svc, "get_python_path", lambda: "/usr/bin/python3")
    monkeypatch.setattr(svc, "_stable_service_working_dir", lambda: str(tmp_path))
    monkeypatch.setattr(svc, "_detect_venv_dir", lambda: tmp_path / "venv")
    monkeypatch.setattr(svc, "_build_service_path_dirs", lambda: ["/opt/hermes/bin"])

    plist = svc.generate_launchd_plist(
        svc.DashboardServiceOptions(port=9121, allowed_hosts=("dash.example.com",))
    )

    assert "<string>dashboard</string>" in plist
    assert "<string>--port</string>" in plist
    assert "<string>9121</string>" in plist
    assert "<key>HERMES_DASHBOARD_ALLOWED_HOSTS</key>" in plist
    assert "<integer>4096</integer>" in plist
    assert "dashboard.log" in plist


def test_windows_cmd_script_runs_dashboard(monkeypatch, tmp_path, _isolate_hermes_home):
    import hermes_cli.dashboard_service as svc

    monkeypatch.setattr(svc, "get_python_path", lambda: r"C:\Python\python.exe")
    monkeypatch.setattr(svc, "PROJECT_ROOT", Path(r"C:\Hermes"))

    script = svc.generate_windows_cmd_script(
        svc.DashboardServiceOptions(port=9122, tui=True)
    )

    assert "hermes_cli.main" in script
    assert "dashboard" in script
    assert "--skip-build" in script
    assert "--tui" in script


def test_access_helper_command_generation():
    from hermes_cli.dashboard_service import (
        build_cloudflare_config,
        build_tailscale_serve_command,
    )

    assert build_tailscale_serve_command(target="127.0.0.1:9119") == [
        "tailscale",
        "serve",
        "--bg",
        "--yes",
        "--https=443",
        "127.0.0.1:9119",
    ]

    config = build_cloudflare_config(
        tunnel="abc",
        credentials_file="/secure/abc.json",
        hostname="dash.example.com",
        service="http://127.0.0.1:9119",
    )
    assert "tunnel: abc" in config
    assert "credentials-file: /secure/abc.json" in config
    assert "hostname: dash.example.com" in config
    assert "service: http://127.0.0.1:9119" in config
