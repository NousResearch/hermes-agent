#!/usr/bin/env python3.11
"""Run one isolated Hermes Lab self-improvement smoke pass."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.lab_environment import lab_paths_from_env, validate_lab_or_raise  # noqa: E402
from gateway.dev_control.production_signals import DevProductionSignalStore  # noqa: E402
from seed_dev_lab_data import seed_lab_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke one lab-only self-improvement loop.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8662")
    parser.add_argument("--skip-gateway-health", action="store_true")
    parser.add_argument("--stable-db-path", default=str(Path("~/.hermes/profiles/dev/state.db").expanduser()))
    args = parser.parse_args()

    paths = lab_paths_from_env()
    db_path = Path(args.db_path or paths["db_path"]).expanduser()
    repos_dir = Path(paths["repos_dir"])
    validate_lab_or_raise(
        hermes_home=db_path.parent,
        gateway_port="8662",
        repo_roots=[
            repos_dir / "hermes-agent",
            repos_dir / "Oryn",
        ],
    )
    stable_before = _mtime(args.stable_db_path)
    gateway = _gateway_health(args.gateway_url) if not args.skip_gateway_health else {"skipped": True}
    seed = seed_lab_data(db_path)
    digest = _run_digest(db_path)
    proposals = DevProductionSignalStore(db_path).list_proposals(limit=20)
    stable_after = _mtime(args.stable_db_path)
    result = {
        "ok": bool(seed.get("ok")) and bool(digest.get("ok")) and len(proposals) > 0 and stable_before == stable_after,
        "object": "hermes.dev_lab_loop_smoke",
        "db_path": str(db_path),
        "gateway": gateway,
        "seed": seed,
        "digest": digest,
        "proposal_count": len(proposals),
        "stable_db_unchanged": stable_before == stable_after,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


def _run_digest(db_path: Path) -> dict:
    env = dict(os.environ)
    env.setdefault("HERMES_DEV_SIGNAL_DIGEST_SOURCES", "deterministic,product,reliability")
    lock_path = Path(lab_paths_from_env()["run_dir"]) / "hermes_dev_signal_digest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_dev_signal_digest.py"),
        "--db-path",
        str(db_path),
        "--lock-path",
        str(lock_path),
        "--window-days",
        "7",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        return {"ok": False, "returncode": completed.returncode, "stderr": completed.stderr.strip(), "stdout": completed.stdout.strip()}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"ok": False, "warning": f"Digest output was not JSON: {exc}", "stdout": completed.stdout.strip()}


def _gateway_health(base_url: str) -> dict:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=3) as response:
            return {"ok": 200 <= response.status < 300, "status": response.status}
    except Exception as exc:  # noqa: BLE001 - smoke reports health failures without hiding loop evidence.
        return {"ok": False, "warning": str(exc)}


def _mtime(path: str) -> float | None:
    candidate = Path(path).expanduser()
    return candidate.stat().st_mtime if candidate.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
