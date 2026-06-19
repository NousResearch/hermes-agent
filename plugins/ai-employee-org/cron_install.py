"""Install per-role cron jobs from bundled plugin scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import scripts_dir

CRON_INSTALLERS: tuple[str, ...] = (
    "install-secretary-heartbeat-cron.py",
    "install-job-seeker-cron.py",
    "install-job-recruiter-cron.py",
    "install-delivery-worker-cron.py",
    "install-self-improver-cron.py",
)


def list_installers() -> list[str]:
    root = scripts_dir()
    return [name for name in CRON_INSTALLERS if (root / name).is_file()]


def _default_telegram_chat_id() -> str | None:
    raw = os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    return raw or None


def install_all_crons(
    *,
    telegram_chat_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = scripts_dir()
    results: dict[str, Any] = {"ok": True, "jobs": {}, "deployed_scripts": []}
    chat_id = telegram_chat_id or _default_telegram_chat_id()

    if not dry_run:
        for profile in ("job-seeker", "job-recruiter", "delivery-worker", "self-improver", "secretary"):
            dest_dir = Path.home() / ".hermes" / "profiles" / profile / "scripts"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for name in CRON_INSTALLERS:
                src = root / name
                if not src.is_file():
                    continue
                dest = dest_dir / name
                dest.write_bytes(src.read_bytes())
                results["deployed_scripts"].append(str(dest))

    for name in CRON_INSTALLERS:
        script = root / name
        if not script.is_file():
            results["jobs"][name] = {"ok": False, "error": "missing"}
            results["ok"] = False
            continue
        cmd = [sys.executable, str(script)]
        if name == "install-secretary-heartbeat-cron.py":
            cmd += ["--profile", "secretary"]
        elif name == "install-job-seeker-cron.py":
            cmd += ["--profile", "job-seeker"]
        elif name == "install-job-recruiter-cron.py":
            cmd += ["--profile", "job-recruiter"]
        elif name == "install-delivery-worker-cron.py":
            cmd += ["--profile", "delivery-worker"]
        elif name == "install-self-improver-cron.py":
            cmd += ["--profile", "self-improver"]
        if chat_id:
            cmd += ["--telegram-chat-id", chat_id]
        if dry_run:
            cmd.append("--dry-run")
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(root),
            env=env,
        )
        results["jobs"][name] = {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip() if dry_run else proc.stdout.strip()[:300],
            "stderr": proc.stderr.strip()[:300],
        }
        if proc.returncode != 0:
            results["ok"] = False
    return results
