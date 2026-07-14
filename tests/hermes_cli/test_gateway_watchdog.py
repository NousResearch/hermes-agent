"""Behavior tests for the Windows gateway watchdog (issue #41662)."""

from pathlib import Path

from hermes_cli import gateway_watchdog, gateway_windows


def test_watchdog_noops_when_gateway_is_running(monkeypatch):
    spawned = []
    monkeypatch.setattr(gateway_watchdog, "_service_is_installed", lambda: True)
    monkeypatch.setattr(gateway_watchdog, "_gateway_is_alive", lambda: (True, 42))
    monkeypatch.setattr(gateway_watchdog, "_respawn", lambda: spawned.append(True))

    assert gateway_watchdog.main([]) == 0
    assert spawned == []


def test_watchdog_respawns_when_gateway_crashed(monkeypatch):
    probes = iter([(False, None), (False, None)])
    spawned = []
    monkeypatch.setattr(gateway_watchdog, "_service_is_installed", lambda: True)
    monkeypatch.setattr(gateway_watchdog, "_gateway_is_alive", lambda: next(probes))
    monkeypatch.setattr(gateway_watchdog, "_was_intentionally_stopped", lambda: False)
    monkeypatch.setattr(
        gateway_watchdog,
        "_respawn",
        lambda: spawned.append(True) or 12345,
    )
    monkeypatch.setattr(gateway_watchdog, "_log", lambda _message: None)

    assert gateway_watchdog.main([]) == 0
    assert spawned == [True]


def test_watchdog_honors_intentional_stop(monkeypatch):
    spawned = []
    monkeypatch.setattr(gateway_watchdog, "_service_is_installed", lambda: True)
    monkeypatch.setattr(gateway_watchdog, "_gateway_is_alive", lambda: (False, None))
    monkeypatch.setattr(gateway_watchdog, "_was_intentionally_stopped", lambda: True)
    monkeypatch.setattr(gateway_watchdog, "_respawn", lambda: spawned.append(True))

    assert gateway_watchdog.main([]) == 0
    assert spawned == []


def test_watchdog_loop_exits_after_uninstall(monkeypatch):
    slept = []
    monkeypatch.setattr(gateway_watchdog, "_service_is_installed", lambda: False)
    monkeypatch.setattr(gateway_watchdog, "_log", lambda _message: None)
    monkeypatch.setattr(gateway_watchdog.time, "sleep", lambda delay: slept.append(delay))

    assert gateway_watchdog.main(["--loop", "--interval", "1"]) == 0
    assert slept == []


def test_watchdog_task_xml_is_periodic_windowless_and_single_flight():
    xml = gateway_windows._build_watchdog_scheduled_task_xml(
        "Hermes_Gateway_Watchdog",
        Path(r"C:\Hermes\watchdog.vbs"),
        r"DOMAIN\alice",
    )

    assert "<Interval>PT2M</Interval>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<Command>wscript.exe</Command>" in xml
    assert "watchdog.vbs" in xml
    assert "cmd.exe" not in xml


def test_watchdog_vbs_waits_for_ticks_but_detaches_startup_loop(monkeypatch):
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda _exe: (r"C:\Python\pythonw.exe", Path(r"C:\venv"), []),
    )
    tick = gateway_windows._build_watchdog_vbs_script(
        r"C:\Python\python.exe",
        r"C:\Hermes",
        r"C:\HermesHome",
    )
    loop = gateway_windows._build_watchdog_vbs_script(
        r"C:\Python\python.exe",
        r"C:\Hermes",
        r"C:\HermesHome",
        loop=True,
    )

    assert "cmd.exe" not in tick
    assert ", 0, True" in tick
    assert "--loop" not in tick
    assert ", 0, False" in loop
    assert "--loop" in loop


def test_startup_launcher_starts_gateway_then_hidden_watchdog_loop():
    launcher = gateway_windows._build_startup_launcher(
        Path(r"C:\Hermes\gateway.cmd"),
        Path(r"C:\Hermes\watchdog.loop.vbs"),
    )

    gateway_pos = launcher.index("gateway.vbs")
    watchdog_pos = launcher.index("watchdog.loop.vbs")
    assert gateway_pos < watchdog_pos
    assert launcher.count(", 0, False") == 2


def test_startup_fallback_writes_and_installs_watchdog(monkeypatch, tmp_path):
    gateway_script = tmp_path / "gateway.cmd"
    startup_entry = tmp_path / "startup.vbs"
    calls = []
    monkeypatch.setattr(
        gateway_windows,
        "_write_watchdog_script",
        lambda: calls.append("write_watchdog") or (tmp_path / "watchdog.cmd"),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_install_startup_entry",
        lambda path: calls.append(("install_startup", path)) or startup_entry,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_get_watchdog_loop_vbs_path",
        lambda: tmp_path / "watchdog.loop.vbs",
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda: [99],
    )
    monkeypatch.setattr(
        gateway_windows,
        "_print_next_steps",
        lambda: calls.append("next_steps"),
    )

    gateway_windows._install_startup_fallback(
        gateway_script,
        start_now=False,
        detail="access denied",
    )

    assert calls == [
        "write_watchdog",
        ("install_startup", gateway_script),
        "next_steps",
    ]


def test_uninstall_removes_watchdog_task_and_artifacts(monkeypatch, tmp_path):
    gateway_script = tmp_path / "gateway.cmd"
    watchdog_script = tmp_path / "watchdog.cmd"
    startup_entry = tmp_path / "startup.vbs"
    legacy_entry = tmp_path / "startup.cmd"
    watchdog_loop = tmp_path / "watchdog.loop.vbs"
    for path in (
        gateway_script,
        gateway_script.with_suffix(".vbs"),
        watchdog_script,
        watchdog_script.with_suffix(".vbs"),
        watchdog_loop,
        startup_entry,
        legacy_entry,
    ):
        path.write_text("x", encoding="utf-8")

    calls = []
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Gateway")
    monkeypatch.setattr(
        gateway_windows,
        "_get_watchdog_task_name",
        lambda: "Gateway_Watchdog",
    )
    monkeypatch.setattr(
        gateway_windows,
        "get_task_script_path",
        lambda: gateway_script,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_get_watchdog_script_path",
        lambda: watchdog_script,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_get_watchdog_loop_vbs_path",
        lambda: watchdog_loop,
    )
    monkeypatch.setattr(
        gateway_windows,
        "get_startup_entry_path",
        lambda: startup_entry,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_legacy_startup_entry_path",
        lambda: legacy_entry,
    )
    registered = {"gateway": True, "watchdog": True}
    monkeypatch.setattr(
        gateway_windows,
        "is_task_registered",
        lambda: registered["gateway"],
    )
    monkeypatch.setattr(
        gateway_windows,
        "is_watchdog_task_registered",
        lambda: registered["watchdog"],
    )

    def fake_schtasks(argv):
        calls.append(argv)
        if argv[-1] == "Gateway":
            registered["gateway"] = False
        if argv[-1] == "Gateway_Watchdog":
            registered["watchdog"] = False
        return (0, "", "")

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

    gateway_windows.uninstall()

    assert ["/Delete", "/F", "/TN", "Gateway"] in calls
    assert ["/Delete", "/F", "/TN", "Gateway_Watchdog"] in calls
    assert not gateway_script.exists()
    assert not gateway_script.with_suffix(".vbs").exists()
    assert not watchdog_script.exists()
    assert not watchdog_script.with_suffix(".vbs").exists()
    assert not watchdog_loop.exists()
    assert not startup_entry.exists()
    assert not legacy_entry.exists()


def test_status_reports_periodic_watchdog(monkeypatch, capsys):
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Gateway")
    monkeypatch.setattr(
        gateway_windows,
        "_get_watchdog_task_name",
        lambda: "Gateway_Watchdog",
    )
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "is_watchdog_task_registered",
        lambda: True,
    )
    monkeypatch.setattr(
        gateway_windows,
        "is_startup_entry_installed",
        lambda: False,
    )
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [321])
    monkeypatch.setattr(
        gateway_windows,
        "query_task_status",
        lambda task_name=None: {"status": "Ready"},
    )

    gateway_windows.status()

    output = capsys.readouterr().out
    assert "Scheduled Task registered: Gateway" in output
    assert "Watchdog Scheduled Task registered: Gateway_Watchdog" in output
    assert "Gateway process running (PID: 321)" in output
