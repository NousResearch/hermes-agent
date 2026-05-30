import re

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
