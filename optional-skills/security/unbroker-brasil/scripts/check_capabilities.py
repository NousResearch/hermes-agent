#!/usr/bin/env python3
"""Capability audit for unbroker-brasil.

Prints booleans and paths only. Never prints secret values.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def present(name: str, envfile: dict[str, str]) -> bool:
    return bool(os.environ.get(name) or envfile.get(name))


def mode(path: Path) -> str | None:
    try:
        return oct(stat.S_IMODE(path.stat().st_mode))
    except FileNotFoundError:
        return None


def tcp_open(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "unbroker-brasil-capability-audit"})
        with urlopen(req, timeout=timeout) as r:  # noqa: S310 - local diagnostic only
            return 200 <= getattr(r, "status", 0) < 500
    except (URLError, HTTPError, TimeoutError, OSError):
        return False


def main() -> None:
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    pdd_data_dir = Path(os.environ.get("PDD_DATA_DIR", str(hermes_home / "unbroker")))
    env_path = hermes_home / ".env"
    envfile = load_dotenv(env_path)

    browser_bins = {
        name: shutil.which(name)
        for name in [
            "chrome",
            "google-chrome",
            "chromium",
            "chromium-browser",
            "brave-browser",
            "microsoft-edge",
            "chrome-for-testing",
        ]
    }
    browser_bins = {k: v for k, v in browser_bins.items() if v}

    ledger = pdd_data_dir / "ledger.json"
    dossiers = pdd_data_dir / "dossiers.json"

    result = {
        "hermes_home": str(hermes_home),
        "pdd_data_dir": str(pdd_data_dir),
        "env_file_present": env_path.exists(),
        "secrets_present": {
            "BROWSERBASE_API_KEY": present("BROWSERBASE_API_KEY", envfile),
            "EMAIL_ADDRESS": present("EMAIL_ADDRESS", envfile),
            "EMAIL_PASSWORD": present("EMAIL_PASSWORD", envfile),
            "EMAIL_SMTP_HOST": present("EMAIL_SMTP_HOST", envfile),
            "EMAIL_IMAP_HOST": present("EMAIL_IMAP_HOST", envfile),
            "GOOGLE_APPLICATION_CREDENTIALS": present("GOOGLE_APPLICATION_CREDENTIALS", envfile),
        },
        "browser": {
            "binaries": browser_bins,
            "cdp_127_0_0_1_9222_tcp_open": tcp_open("127.0.0.1", 9222),
            "cdp_json_version_reachable": http_ok("http://127.0.0.1:9222/json/version"),
        },
        "commands": {
            "python3": shutil.which("python3"),
            "age": shutil.which("age"),
            "gws": shutil.which("gws"),
            "himalaya": shutil.which("himalaya"),
            "gh": shutil.which("gh"),
        },
        "storage": {
            "pdd_data_dir_exists": pdd_data_dir.exists(),
            "ledger_exists": ledger.exists(),
            "ledger_mode": mode(ledger),
            "dossiers_exists": dossiers.exists(),
            "dossiers_mode": mode(dossiers),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
