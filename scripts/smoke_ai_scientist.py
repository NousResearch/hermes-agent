#!/usr/bin/env python3
"""Smoke runner: nc_kan template + ai_scientist_research + harness /scientist/run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_AI = REPO_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
HARNESS_SCRIPTS = REPO_ROOT / "vendor" / "openclaw-mirror" / "extensions" / "hypura-harness" / "scripts"
NC_KAN = VENDOR_AI / "templates" / "nc_kan"


def smoke_nc_kan_template() -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "experiment.py", "--out_dir=run_smoke_cli"],
        cwd=str(NC_KAN),
        capture_output=True,
        text=True,
        timeout=60,
    )
    ok = proc.returncode == 0 and (NC_KAN / "run_smoke_cli" / "final_info.json").is_file()
    return {"step": "nc_kan_template", "ok": ok, "stderr_tail": proc.stderr[-400:]}


def smoke_ai_scientist_research(live: bool) -> dict[str, object]:
    if not live:
        return {"step": "ai_scientist_research", "ok": True, "skipped": "use --live-api for launch_scientist.py"}

    os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-smoke-"))
    sys.path.insert(0, str(REPO_ROOT))
    from tools.ai_scientist_env import describe_credential_resolution

    model = os.getenv("AI_SCIENTIST_SMOKE_MODEL", "auto")
    creds = describe_credential_resolution(model)
    if not creds.get("has_credentials"):
        return {
            "step": "ai_scientist_research",
            "ok": False,
            "error": "no Hermes-bridged credentials for model",
            "credential_status": creds,
        }

    from tools.ai_scientist_tool import ai_scientist_research

    payload = json.loads(
        ai_scientist_research(
            experiment="nc_kan",
            num_ideas=1,
            model=model,
            task_id="smoke_nc_kan",
            skip_novelty_check=True,
            use_gpu=False,
        )
    )
    return {
        "step": "ai_scientist_research",
        "ok": bool(payload.get("success")),
        "payload": payload,
        "credential_status": creds,
    }


def smoke_harness_scientist_run() -> dict[str, object]:
    sys.path.insert(0, str(HARNESS_SCRIPTS))
    sys.path.insert(0, str(REPO_ROOT / "tests" / "vendor"))
    from unittest.mock import MagicMock, patch

    from fake_redis import FakeRedis
    from fastapi.testclient import TestClient

    import harness_daemon as hd

    fake = FakeRedis()
    runner = MagicMock()
    runner.run_ideas.return_value = [{"Name": "smoke", "Title": "Smoke", "Experiment": "noop"}]

    with patch.object(hd, "_get_scientist", return_value=runner), patch.object(
        hd.redis_loop, "_get_redis", return_value=fake
    ):
        client = TestClient(hd.app)
        resp = client.post(
            "/scientist/run",
            json={"topic": "smoke", "template": "nc_kan", "num_ideas": 1, "run_experiment": False},
        )

    ok = resp.status_code == 200 and resp.json().get("success") and fake.llen("ai_scientist:findings") == 1
    return {"step": "harness_scientist_run", "ok": ok, "status_code": resp.status_code, "body": resp.json()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-Scientist smoke tests")
    parser.add_argument("--live-api", action="store_true", help="Run real launch_scientist via Hermes launcher (OAuth/free-tier).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        smoke_nc_kan_template(),
        smoke_harness_scientist_run(),
        smoke_ai_scientist_research(live=args.live_api),
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
