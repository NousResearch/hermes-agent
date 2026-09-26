"""One-shot Desktop update handoff, not a second updater (#118098)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from hermes_cli._subprocess_compat import windows_detach_flags

UPDATE_ALL_REQUEST_SWITCH = "--hermes-update-all-request="


def _write_request(path: Path, data: dict) -> None:
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(data), encoding="utf-8")
    staged.replace(path)


def _launcher_pids() -> list[int]:
    pids = [os.getpid()]
    if sys.platform == "win32":
        # The distlib console shim waits for this interpreter. Both must release
        # the venv before Desktop's native updater inspects it.
        import psutil
        parent = psutil.Process().parent()
        if parent and Path(parent.exe()).resolve() == Path(sys.executable).with_name("hermes.exe").resolve():
            pids.append(parent.pid)
    return pids


def launch_desktop_update_all(command: list[str], *, cwd: Path, env: dict,
                              timeout: float = 30.0) -> int:
    """Return dispatch status only. Desktop owns progress, blockers and results.

    A short-lived request/ack/commit exchange catches older apps that ignore
    the switch. Desktop does not start updating until this launcher has exited;
    a timeout or crash before commit cannot become a delayed surprise update.
    """
    launcher_pids = _launcher_pids()
    directory = Path(tempfile.mkdtemp(prefix="hermes-update-all-"))
    request = directory / "request.json"
    committed = False
    data = {
        "version": 1, "id": uuid4().hex, "state": "pending",
        "launcher_pids": launcher_pids,
        "expires_at": int((time.time() + timeout + 10) * 1000),
    }
    try:
        _write_request(request, data)
        # Fire-and-forget is intentional: Desktop waits for launcher_pids to
        # disappear before updating, so waiting here would keep the venv locked
        # and deadlock the handoff. Explicitly close inherited descriptors and
        # opportunistically poll below so a short-lived secondary instance is
        # still reaped when it exits during the acknowledgement window.
        desktop = subprocess.Popen(
            [*command, f"{UPDATE_ALL_REQUEST_SWITCH}{request}"],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
            creationflags=windows_detach_flags(),
            close_fds=True,
        )
        deadline = time.monotonic() + timeout
        poll_delay = 0.05
        while time.monotonic() < deadline:
            response = json.loads(request.read_text(encoding="utf-8"))
            if response.get("id") != data["id"]:
                raise ValueError("Desktop returned an unrelated update acknowledgement")
            if response.get("state") == "interrupted":
                desktop.poll()
                print("The previous update lost its Desktop window. Check its outcome in Desktop, "
                      "then restart Desktop before retrying; no new update was started.", file=sys.stderr)
                return 1
            if response.get("state") == "coalesced":
                desktop.poll()
                print("An update-all request is already pending or running in Hermes Desktop.")
                return 0
            if response.get("state") == "accepted":
                _write_request(request, {**data, "state": "committed"})
                committed = True
                desktop.poll()
                print("Hermes Desktop accepted the update request. See Desktop for progress and safety prompts.")
                return 0
            time.sleep(poll_delay)
            poll_delay = min(poll_delay * 2, 0.25)
        print("Desktop did not acknowledge the update request; no update was committed. "
              "Quit any older Desktop instance and launch a current build with `hermes desktop`, "
              "then retry.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Could not hand the update request to Desktop: {exc}", file=sys.stderr)
        return 1
    finally:
        # After commit, Electron removes this directory when it consumes the
        # request. Do not wait for Electron here: that would hold the venv open.
        if not committed:
            shutil.rmtree(directory)
