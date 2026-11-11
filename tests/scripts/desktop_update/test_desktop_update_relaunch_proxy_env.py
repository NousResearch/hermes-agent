"""posix.sh relaunches the Desktop with the user's proxy env, not the one the update was given.

The Desktop hands its resolved system proxy to the update's children only
(apps/desktop/electron/updater/system-proxy-env.ts) and marks it:
HERMES_UPDATE_SYSTEM_PROXY names the variables it added, HERMES_UPDATE_USER_<KEY>
holds a NO_PROXY of the user's that it extended. The relaunched app inheriting
that env would proxy every LAN host the PAC sends direct, and the next update
would read the stale proxy as the user's own setting.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts.desktop_update.test_desktop_update_posix_marker import _install, _run
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux")

_DESKTOP_PROXY_ENV = {
    "HTTPS_PROXY": "http://10.0.0.7:7890",
    "https_proxy": "http://10.0.0.7:7890",
    "HTTP_PROXY": "http://10.0.0.7:7890",
    "http_proxy": "http://10.0.0.7:7890",
    "NO_PROXY": "corp.internal,localhost,127.0.0.1,::1,10.0.0.0/8",
    "no_proxy": "corp.internal,localhost,127.0.0.1,::1,10.0.0.0/8",
    "HERMES_UPDATE_SYSTEM_PROXY": "HTTPS_PROXY,https_proxy,HTTP_PROXY,http_proxy,no_proxy",
    "HERMES_UPDATE_USER_NO_PROXY": "corp.internal",
}


def test_relaunched_desktop_gets_the_users_proxy_env_back(tmp_path, monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
    home, install = _install(tmp_path)
    unpacked = install / "apps/desktop/release/linux-unpacked"
    unpacked.mkdir(parents=True)
    app = unpacked / "hermes"
    dump = tmp_path / "relaunched-env.txt"
    app.write_text(f"#!/bin/sh\necho $$ > {dump}.pid\n"
                   f"env | grep -i -E '_proxy=|^HERMES_UPDATE_(SYSTEM_PROXY|USER_)' > {dump}.part; mv {dump}.part {dump}\n"
                   # Outlive posix.sh's 1.5 s launch-acceptance check, then exit before teardown.
                   "sleep 2\n",
                   encoding="utf-8")
    app.chmod(0o755)

    result = _run(tmp_path, home, install, "--relaunch-target", str(app), **_DESKTOP_PROXY_ENV)

    assert result.returncode == 0, result.stdout + result.stderr
    pid = dump.with_name(dump.name + ".pid").read_text(encoding="utf-8").strip()
    wait_until(lambda: not Path(f"/proc/{pid}").exists(), timeout=10, what="the relaunched app to exit")
    relaunched = dict(line.split("=", 1) for line in dump.read_text(encoding="utf-8").splitlines())
    assert relaunched == {"NO_PROXY": "corp.internal"}
