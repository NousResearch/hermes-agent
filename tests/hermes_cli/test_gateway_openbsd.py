from __future__ import annotations

import pytest

from hermes_cli.gateway_openbsd import (
    CrontabReadError,
    cron_marker,
    install_boot_line,
    merge_crontab,
    missing_crontab,
    remove_marked_lines,
    render_cron_line,
)

# Same synthetic target user as tests/hermes_cli/test_gateway_service.py.
# These strings are not opened; tests that need a real directory use tmp_path / "alice".
_USER = "alice"
_HOME = "/home/alice/.hermes"
_BINARY = "/home/alice/.local/bin/hermes"


def test_render_cron_line_does_not_use_bare_hermes():
    line = render_cron_line([_BINARY, "gateway", "run"], _HOME, "default")
    assert line.startswith("@reboot ")
    assert "PATH=/usr/local/bin:/usr/bin:/bin:" in line
    assert _BINARY in line
    assert "@reboot hermes " not in line
    assert line.endswith(f">> {_HOME}/logs/gateway.log 2>&1 # hermes-gateway home={_HOME} profile=default")


def test_merge_replaces_existing_marker():
    marker = cron_marker(_HOME, "default")
    existing = f"0 * * * * echo keep\n@reboot old {marker}\n"
    merged = merge_crontab(existing, "@reboot new " + marker, marker)
    assert merged.count("@reboot") == 1
    assert "echo keep" in merged
    assert "old" not in merged


def test_remove_marked_line_keeps_unrelated():
    marker = cron_marker(_HOME, "default")
    existing = f"0 * * * * echo keep\n@reboot gone {marker}\n"
    assert remove_marked_lines(existing, marker).strip() == "0 * * * * echo keep"


def test_missing_crontab_is_only_that_stderr():
    assert missing_crontab(1, f"crontab: no crontab for {_USER}\n")
    assert not missing_crontab(1, f"crontab: you ({_USER}) are not allowed to use this program\n")
    assert not missing_crontab(0, "")


def test_read_failure_does_not_write():
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)

        class R:
            returncode = 1
            stdout = ""
            stderr = f"crontab: you ({_USER}) are not allowed to use this program\n"
        return R()

    with pytest.raises(CrontabReadError):
        install_boot_line([_BINARY, "gateway", "run"], _HOME, "default", run=run)
    assert calls == [["crontab", "-l"]]


def test_write_failure_raises():
    def run(argv, **kwargs):
        class R:
            returncode = 0 if argv == ["crontab", "-l"] else 1
            stdout = ""
            stderr = "crontab: permission denied\n"
        return R()

    with pytest.raises(CrontabReadError):
        install_boot_line([_BINARY, "gateway", "run"], _HOME, "default", run=run)


def test_system_boot_command_runs_as_named_user():
    from hermes_cli.gateway_openbsd import system_boot_command
    cmd = system_boot_command(_USER, _HOME, [_BINARY, "gateway", "run"])
    assert cmd[:4] == ["/usr/bin/su", "-s", "/bin/sh", _USER]
    assert cmd[4] == "-c"
    assert f"HERMES_HOME={_HOME}" in cmd[5]
    assert _BINARY in cmd[5]


@pytest.mark.platforms("openbsd")
def test_start_does_not_spawn_when_pid_is_alive(monkeypatch):
    import hermes_cli.gateway as gw
    spawned = []
    monkeypatch.setattr(gw.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *args, **kwargs: 4242)
    gw._openbsd_start_gateway()
    assert spawned == []


@pytest.mark.platforms("openbsd")
def test_service_backend_is_openbsd(monkeypatch):
    from hermes_cli.gateway import _service_backend, is_openbsd
    assert is_openbsd()
    assert _service_backend() == "openbsd"
