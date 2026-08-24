"""The real Windows update server retains terminal events until acknowledged."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pytestmark = pytest.mark.windows_only
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/desktop-update/windows.ps1"


@contextmanager
def _server(tmp_path: Path, *, failed: bool = False):
    powershell = shutil.which("powershell.exe")
    assert powershell, "Windows updater tests require Windows PowerShell."
    env = os.environ.copy()
    env.update(TEMP=str(tmp_path), TMP=str(tmp_path), HERMES_SELFTEST_HOLD_SECONDS="0")
    env.pop("HERMES_SELFTEST_FAIL", None)
    if failed:
        env["HERMES_SELFTEST_FAIL"] = "1"
    output_path = tmp_path / "ui-delivery.log"
    with output_path.open("wb") as output:
        process = subprocess.Popen(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-SelfTestUi", "-NoUi"],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            text = output_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"SELF-TEST: shim at (http://127\.0\.0\.1:\d+/)", text)
            if match:
                yield process, match.group(1)
                return
            if process.poll() is not None:
                break
            time.sleep(0.05)
        pytest.fail(f"Update server did not publish a serving URL: {text}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _request(url: str, *, post: bool = False):
    request = Request(url, data=b"" if post else None, method="POST" if post else "GET")
    return urlopen(request, timeout=3)


@pytest.mark.parametrize("failed", [False, True], ids=["complete", "failed"])
def test_delayed_client_receives_terminal_state_before_acknowledging(tmp_path: Path, failed: bool) -> None:
    with _server(tmp_path, failed=failed) as (process, url):
        # A background browser can miss the former 900ms terminal-state window.
        time.sleep(2)
        assert process.poll() is None, "the updater discarded its final state before the delayed client received it"
        with _request(url + "progress") as response:
            state = json.load(response)
        assert state["status"] == ("error" if failed else "done")
        receipt = state["receipt"]
        assert isinstance(receipt, str) and receipt

        with pytest.raises(HTTPError) as wrong:
            _request(url + "ack/not-the-published-receipt", post=True)
        assert wrong.value.code == 409
        assert process.poll() is None, "an unrelated acknowledgement must not dispose the window's state"

        for route, post in [("progress-other", False), ("ack/" + receipt, False), ("ack/" + receipt + "/extra", True)]:
            with pytest.raises(HTTPError) as invalid:
                _request(url + route, post=post)
            assert invalid.value.code == 404

        with _request(url + "ack/" + receipt, post=True) as response:
            assert response.status == 204
        assert process.wait(timeout=15) == 0


def test_no_client_does_not_keep_the_update_process_alive_forever(tmp_path: Path) -> None:
    with _server(tmp_path) as (process, _url):
        assert process.wait(timeout=25) == 0
