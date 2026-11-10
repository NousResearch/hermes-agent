from __future__ import annotations

import subprocess

from hermes_cli import _subprocess_compat as compat


def test_user_transport_config_replays_allowlisted_system_and_global_values(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        key = command[-1]
        values = {
            "http.proxy": b"http://system-proxy\0",
            "https.proxy": b"http://global-proxy\0",
            "http.sslBackend": b"openssl\0",
            "http.sslCAInfo": b"/etc/custom-ca.pem\0",
        }
        return subprocess.CompletedProcess(command, 0, stdout=values.get(key, b""), stderr=b"")

    monkeypatch.setattr(compat.subprocess, "run", fake_run)

    result = compat._user_transport_config({"HOME": "/tmp/home"})

    assert result == [
        ("http.proxy", "http://system-proxy"),
        ("https.proxy", "http://global-proxy"),
        ("http.sslBackend", "openssl"),
        ("http.sslCAInfo", "/etc/custom-ca.pem"),
        ("http.proxy", "http://system-proxy"),
        ("https.proxy", "http://global-proxy"),
        ("http.sslBackend", "openssl"),
        ("http.sslCAInfo", "/etc/custom-ca.pem"),
    ]
    assert len(calls) == 8
    assert all(call[0:3] == ["git", "config", "--system"] or call[0:3] == ["git", "config", "--global"] for call in calls)
