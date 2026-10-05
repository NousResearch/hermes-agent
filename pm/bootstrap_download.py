"""Windows TLS for pinned tool acquisition before PM's runtime exists."""

from __future__ import annotations

from email.message import Message
import json
from pathlib import Path
import ssl
import subprocess
import tempfile
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request


def fetch_windows(url: str, destination: Path, progress: Callable[[int], None]) -> int:
    """Stream through SChannel; the downloader still owns hashes/publication.

    PowerShell is available before a managed toolchain, including in a -I -S
    caller. Request data is a file, never executable shell interpolation.
    """
    from pm.downloader import _UA

    proxy = ""
    parsed = urllib.parse.urlsplit(url)
    if not urllib.request.proxy_bypass(parsed.netloc):
        proxy = urllib.request.getproxies().get(parsed.scheme, "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tls-", dir=destination.parent) as scratch:
        request = Path(scratch) / "request.json"
        result = Path(scratch) / "result.json"
        request.write_text(
            json.dumps({
                "url": url,
                "destination": str(destination),
                "proxy": proxy,
                "user_agent": _UA["User-Agent"],
            }),
            encoding="utf-8",
        )
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path(__file__).with_suffix(".ps1")),
            str(request),
            str(result),
        ]
        with subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ) as child:
            deadline = time.monotonic() + 600
            try:
                while child.poll() is None:
                    progress(destination.stat().st_size if destination.exists() else 0)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Windows bootstrap download timed out")
                    time.sleep(0.1)
                _, stderr = child.communicate(timeout=5)
            except BaseException:
                child.kill()
                child.wait()
                raise
        if not result.is_file():
            raise urllib.error.URLError(
                f"Windows TLS helper exited {child.returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        outcome = json.loads(result.read_text(encoding="utf-8-sig"))
        if not outcome["ok"]:
            if outcome.get("status"):
                headers = Message()
                for key, value in outcome.get("headers", {}).items():
                    headers[key] = value
                raise urllib.error.HTTPError(
                    url, outcome["status"], outcome["error"], headers, None
                )
            if outcome.get("tls"):
                reason = ssl.SSLError(outcome["error"])
            elif outcome.get("timeout"):
                reason = TimeoutError(outcome["error"])
            elif outcome.get("connection"):
                reason = ConnectionError(outcome["error"])
            else:
                reason = RuntimeError(outcome["error"])
            raise urllib.error.URLError(reason)
        if child.returncode:
            raise urllib.error.URLError(f"Windows TLS helper exited {child.returncode}")
        size = destination.stat().st_size
        progress(size)
        return size
