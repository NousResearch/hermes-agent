from types import SimpleNamespace


def test_default_multiplexer_probe_requires_live_runtime(monkeypatch, tmp_path):
    from gateway import status as runtime_status
    from hermes_cli import gateway

    runtime = {
        "pid": 4321,
        "gateway_state": "running",
        "served_profiles": ["default", "coder", "coder", ""],
    }
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: tmp_path
    )
    monkeypatch.setattr(runtime_status, "read_runtime_status", lambda path: runtime)
    monkeypatch.setattr(
        runtime_status,
        "get_runtime_status_running_pid",
        lambda record, expected_home=None: 4321,
    )

    coverage = gateway.get_default_multiplex_gateway()

    assert coverage == gateway.DefaultMultiplexGateway(
        pid=4321, served_profiles=("default", "coder")
    )

    monkeypatch.setattr(
        runtime_status,
        "get_runtime_status_running_pid",
        lambda record, expected_home=None: None,
    )
    assert gateway.get_default_multiplex_gateway() is None


def test_gateway_list_marks_profiles_served_by_default_multiplexer(
    monkeypatch, capsys, tmp_path
):
    from gateway import status as runtime_status
    from hermes_cli import gateway
    from hermes_cli import profiles as profiles_mod

    profiles = [
        SimpleNamespace(name="default", path=tmp_path, gateway_running=True),
        SimpleNamespace(
            name="coder", path=tmp_path / "profiles" / "coder", gateway_running=False
        ),
        SimpleNamespace(
            name="idle", path=tmp_path / "profiles" / "idle", gateway_running=False
        ),
    ]
    monkeypatch.setattr(profiles_mod, "list_profiles", lambda: profiles)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "coder")
    monkeypatch.setattr(runtime_status, "get_running_pid", lambda *a, **k: 4321)
    monkeypatch.setattr(
        gateway,
        "get_default_multiplex_gateway",
        lambda: gateway.DefaultMultiplexGateway(
            pid=4321, served_profiles=("default", "coder")
        ),
    )

    gateway._gateway_list()

    output = capsys.readouterr().out
    assert "default" in output
    assert "PID 4321" in output
    assert "serves 2 profiles" in output
    assert "coder (current)" in output
    assert "served by default multiplexer (PID 4321)" in output
    assert "✗ idle" in output and "not running" in output


def test_gateway_status_reports_secondary_profile_coverage(monkeypatch, capsys):
    from hermes_cli import gateway
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "coder")
    monkeypatch.setattr(
        gateway,
        "get_gateway_runtime_snapshot",
        lambda system=False: gateway.GatewayRuntimeSnapshot(manager="manual process"),
    )
    monkeypatch.setattr(
        gateway,
        "get_default_multiplex_gateway",
        lambda: gateway.DefaultMultiplexGateway(
            pid=4321, served_profiles=("default", "coder")
        ),
    )
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    monkeypatch.setattr(gateway, "_print_other_profiles_gateway_status", lambda: None)

    gateway.gateway_command(
        SimpleNamespace(gateway_command="status", deep=False, full=False, system=False)
    )

    output = capsys.readouterr().out
    assert "running via the default-profile multiplexer (PID: 4321)" in output
    assert "Profiles served: default, coder" in output
    assert "Gateway is not running" not in output


def test_hermes_status_reports_secondary_profile_coverage(
    monkeypatch, capsys, tmp_path
):
    from hermes_cli import gateway
    from hermes_cli import profiles as profiles_mod
    from hermes_cli import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "test-model"})
    monkeypatch.setattr(
        status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex"
    )
    monkeypatch.setattr(
        status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex"
    )
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex")
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "coder")
    monkeypatch.setattr(
        gateway,
        "get_gateway_runtime_snapshot",
        lambda: gateway.GatewayRuntimeSnapshot(manager="manual process"),
    )
    monkeypatch.setattr(
        gateway,
        "get_default_multiplex_gateway",
        lambda: gateway.DefaultMultiplexGateway(
            pid=4321, served_profiles=("default", "coder")
        ),
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    gateway_section = capsys.readouterr().out.split("◆ Gateway Service", 1)[1].split(
        "◆ Scheduled Jobs", 1
    )[0]
    assert "Status:       ✓ running" in gateway_section
    assert "Manager:      default-profile multiplexer" in gateway_section
    assert "PID(s):       4321" in gateway_section
    assert "Profiles:     default, coder" in gateway_section
    assert "stopped" not in gateway_section
