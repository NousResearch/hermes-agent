"""Unit tests for the Desktop setup flow's mDNS publishing (tmp conf files).

Avahi registers one name per local interface address, so the flow sets the
daemon's own host-name rather than aliasing via /etc/avahi/hosts (which
collides for the host's own IP — verified live on avahi 0.8).
"""
from types import SimpleNamespace

from hermes_cli.webapp_publish import mdns

CONF = "[server]\n#host-name=old\nuse-ipv4=yes\n\n[publish]\ndisable-publishing=no\n"


def test_label_of_strips_local_suffix():
    assert mdns.label_of("hermes-desktop.local") == "hermes-desktop"
    assert mdns.label_of("Desk.Local") == "desk"


def test_current_hostname_reads_server_section(tmp_path):
    f = tmp_path / "avahi-daemon.conf"
    f.write_text(CONF)
    assert mdns.current_hostname(f) is None  # commented line doesn't count
    f.write_text("[server]\nhost-name=box\n[publish]\nhost-name=decoy\n")
    assert mdns.current_hostname(f) == "box"


def test_current_hostname_missing_file(tmp_path):
    assert mdns.current_hostname(tmp_path / "nope.conf") is None


def test_set_hostname_inserts_into_server_section(tmp_path):
    f = tmp_path / "avahi-daemon.conf"
    f.write_text(CONF)
    changed, backup = mdns.set_hostname(f, "hermes-desktop.local")
    assert changed is True and backup is not None and backup.exists()
    text = f.read_text()
    assert "host-name=hermes-desktop" in text
    assert mdns.current_hostname(f) == "hermes-desktop"
    assert "[publish]" in text and "disable-publishing=no" in text  # other sections kept
    assert text.count("host-name=hermes-desktop") == 1


def test_set_hostname_replaces_existing(tmp_path):
    f = tmp_path / "avahi-daemon.conf"
    f.write_text("[server]\nhost-name=old-box\nuse-ipv4=yes\n")
    mdns.set_hostname(f, "hermes-desktop.local")
    text = f.read_text()
    assert "host-name=old-box" not in text
    assert mdns.current_hostname(f) == "hermes-desktop"


def test_set_hostname_idempotent(tmp_path):
    f = tmp_path / "avahi-daemon.conf"
    f.write_text(CONF)
    mdns.set_hostname(f, "hermes-desktop.local")
    changed, backup = mdns.set_hostname(f, "hermes-desktop.local")
    assert changed is False and backup is None


def test_set_hostname_no_server_section(tmp_path):
    f = tmp_path / "avahi-daemon.conf"
    f.write_text("[publish]\ndisable-publishing=no\n")
    mdns.set_hostname(f, "desk.local")
    assert mdns.current_hostname(f) == "desk"


def test_publish_without_avahi_gives_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(mdns, "avahi_installed", lambda: False)
    result = mdns.publish_local_name("hermes-desktop.local", "192.168.1.5", conf_file=tmp_path / "x.conf")
    assert result["published"] is False
    assert any("apt-get install" in line for line in result["instructions"])
    assert not (tmp_path / "x.conf").exists()


def test_publish_permission_error_degrades_to_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(mdns, "avahi_installed", lambda: True)

    def boom(conf_file, name):
        raise PermissionError("read-only")

    monkeypatch.setattr(mdns, "set_hostname", boom)
    result = mdns.publish_local_name("hermes-desktop.local", "192.168.1.5", conf_file=tmp_path / "x.conf")
    assert result["published"] is False
    assert any("host-name=" in line for line in result["instructions"])


def test_publish_success_restarts_daemon(tmp_path, monkeypatch):
    monkeypatch.setattr(mdns, "avahi_installed", lambda: True)
    f = tmp_path / "avahi-daemon.conf"
    f.write_text(CONF)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="active")

    result = mdns.publish_local_name("hermes-desktop.local", "192.168.1.5", conf_file=f, run=run)
    assert result["published"] is True and result["changed"] is True
    assert ["systemctl", "restart", "avahi-daemon"] in calls
    assert mdns.current_hostname(f) == "hermes-desktop"


def test_publish_reports_previous_hostname(tmp_path, monkeypatch):
    monkeypatch.setattr(mdns, "avahi_installed", lambda: True)
    f = tmp_path / "avahi-daemon.conf"
    f.write_text("[server]\nhost-name=old-box\n")
    result = mdns.publish_local_name("hermes-desktop.local", "192.168.1.5", conf_file=f,
                                     run=lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=""))
    assert result["previous"] == "old-box"


def test_publish_restart_failure_reports_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(mdns, "avahi_installed", lambda: True)
    f = tmp_path / "avahi-daemon.conf"
    f.write_text(CONF)
    result = mdns.publish_local_name("hermes-desktop.local", "192.168.1.5", conf_file=f,
                                     run=lambda cmd, **kw: SimpleNamespace(returncode=1, stdout=""))
    assert result["published"] is False
    assert result["instructions"] == ["sudo systemctl restart avahi-daemon"]


def test_resolves_matches_expected_ip(monkeypatch):
    monkeypatch.setattr(mdns.shutil, "which", lambda name: "/usr/bin/avahi-resolve-host-name")
    ok = mdns.resolves("hermes-desktop.local", "192.168.1.5",
                       run=lambda cmd, **kw: SimpleNamespace(stdout="hermes-desktop.local\t192.168.1.5\n"))
    assert ok is True
    bad = mdns.resolves("hermes-desktop.local", "192.168.1.5",
                        run=lambda cmd, **kw: SimpleNamespace(stdout=""))
    assert bad is False