import re

import pytest

from hermes_cli import gateway


def test_launchd_plist_declares_fd_resource_limits():
    """macOS launchd defaults to a 256 fd soft cap; the dashboard's slash/PTY
    children exhaust it. The plist must raise it (#24775)."""
    plist = gateway.generate_launchd_plist()
    soft = re.search(r"<key>SoftResourceLimits</key>\s*<dict>(.*?)</dict>", plist, re.S)
    hard = re.search(r"<key>HardResourceLimits</key>\s*<dict>(.*?)</dict>", plist, re.S)
    assert soft and re.search(r"<key>NumberOfFiles</key>\s*<integer>8192</integer>", soft.group(1))
    assert hard and re.search(r"<key>NumberOfFiles</key>\s*<integer>16384</integer>", hard.group(1))


def test_launchd_plist_without_fd_limits_reads_as_stale(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist_path)

    current = gateway.generate_launchd_plist()
    plist_path.write_text(current, encoding="utf-8")
    assert gateway.launchd_plist_is_current() is True

    # A plist predating the fd-limit bump must be detected as stale so the
    # boot-time self-heal rewrites it.
    old = re.sub(
        r"\s*<key>(Soft|Hard)ResourceLimits</key>\s*<dict>.*?</dict>", "", current, flags=re.S
    )
    assert old != current
    plist_path.write_text(old, encoding="utf-8")
    assert gateway.launchd_plist_is_current() is False


def test_run_gateway_self_heals_launchd_plist_on_macos(monkeypatch):
    """A code update that ships new plist settings (e.g. the fd-limit bump) must
    reach macOS on the next boot, not only on an explicit `hermes update`."""
    monkeypatch.setattr(gateway, "is_macos", lambda: True)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    healed = []
    monkeypatch.setattr(
        gateway, "refresh_launchd_plist_if_needed", lambda restart=True: healed.append(restart)
    )

    class _Halt(Exception):
        pass

    monkeypatch.setattr("gateway.run.start_gateway", lambda *a, **k: (_ for _ in ()).throw(_Halt()))

    with pytest.raises(_Halt):
        gateway.run_gateway()
    # restart=False: in-process self-heal must refresh the plist on disk but
    # must NOT bootout/bootstrap the gateway it is currently running inside.
    assert healed == [False]


def _stale_plist(tmp_path, monkeypatch):
    """Install a current plist, then downgrade it to a pre-fd-limit (stale) one."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist_path)
    current = gateway.generate_launchd_plist()
    old = re.sub(
        r"\s*<key>(Soft|Hard)ResourceLimits</key>\s*<dict>.*?</dict>", "", current, flags=re.S
    )
    assert old != current
    plist_path.write_text(old, encoding="utf-8")
    return plist_path, current


def test_refresh_launchd_no_restart_rewrites_without_bootout(tmp_path, monkeypatch):
    """In-process refresh (restart=False) must rewrite the stale plist on disk but
    never shell out to launchctl bootout/bootstrap — that would SIGTERM the running
    gateway mid-startup (and orphan the job if it died before bootstrap ran)."""
    plist_path, current = _stale_plist(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(gateway.subprocess, "run", lambda *a, **k: calls.append(a[0]))

    assert gateway.refresh_launchd_plist_if_needed(restart=False) is True
    assert plist_path.read_text(encoding="utf-8") == current  # rewritten to current
    assert gateway.launchd_plist_is_current() is True
    assert calls == []  # no launchctl invocation


def test_refresh_launchd_with_restart_does_bootout(tmp_path, monkeypatch):
    """The explicit-CLI path (restart=True, default) still bootout/bootstraps so an
    operator running `hermes gateway install/start` applies the change immediately."""
    _stale_plist(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(gateway.subprocess, "run", lambda *a, **k: calls.append(a[0][:2]))

    assert gateway.refresh_launchd_plist_if_needed() is True
    assert ["launchctl", "bootout"] in calls
    assert ["launchctl", "bootstrap"] in calls


def test_refresh_launchd_noop_when_current(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist_path)
    plist_path.write_text(gateway.generate_launchd_plist(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(gateway.subprocess, "run", lambda *a, **k: calls.append(a[0]))

    assert gateway.refresh_launchd_plist_if_needed(restart=False) is False
    assert calls == []
