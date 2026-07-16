from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conductor.engine import Conductor, TickResult
from conductor.launcher import TmuxLauncher
from conductor.models import CampaignPlan, Step, StepKind
from conductor.store import ConductorStore


WORKER = r"""
import hashlib, json, os, pathlib, sys
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
repo = pathlib.Path(spec["cwd"])
if spec["role"] == "writer":
    (repo / "known.txt").write_text("created by governed worker\n")
usage = {"input_tokens": 2, "output_tokens": 3, "reasoning_tokens": 1,
         "cache_read_tokens": 4, "cache_write_tokens": 5}
body = {"schema": 1, "worker_id": spec["worker_id"], "campaign_id": spec["campaign_id"],
        "step_index": spec["step_index"], "role": spec["role"], "cwd": spec["cwd"],
        "tmux_session": spec["tmux_session"], "provider": spec["provider"],
        "model": spec["model"], "prompt_hash": spec["prompt_hash"],
        "mutable_manifest": spec["mutable_manifest"], "nonce": spec["nonce"],
        "status": "COMPLETE", "usage": usage, "worker_turns": 1, "model_fallback": False}
raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
body["receipt_hash"] = hashlib.sha256(raw).hexdigest()
path = pathlib.Path(spec["receipt_path"]); path.write_text(json.dumps(body, sort_keys=True))
pathlib.Path(spec["output_path"]).write_text(json.dumps({"model_calls": 1, "role": spec["role"]}))
"""

FAKE_TMUX = r"""#!/usr/bin/env python3
import subprocess, sys
args = sys.argv[1:]
if args[:1] == ["-S"]:
    args = args[2:]
if args and args[0] == "new-session":
    cwd = args[args.index("-c") + 1]
    command = args[-1]
    raise SystemExit(subprocess.run(command, cwd=cwd, shell=True).returncode)
if args and args[0] == "display-message":
    raise SystemExit(1)
if args and args[0] == "has-session":
    raise SystemExit(1)
raise SystemExit(0)
"""


def wait_tick(engine, campaign_id, terminal, timeout=10):
    deadline = time.time() + timeout
    results = []
    while time.time() < deadline:
        result = engine.tick(campaign_id)
        results.append(result)
        if result is terminal:
            return results
        time.sleep(0.05)
    pytest.fail(f"campaign did not reach {terminal}: {results}")


def test_fully_isolated_tmux_campaign(tmp_path, monkeypatch):
    if sys.platform == "darwin":
        probe = subprocess.run(
            [
                "/usr/bin/sandbox-exec",
                "-p",
                "(version 1) (allow default)",
                "/usr/bin/true",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if (
            probe.returncode == 71
            and "sandbox_apply: Operation not permitted" in probe.stderr
        ):
            pytest.skip(
                "the enclosing test runner sandbox forbids nested Seatbelt profiles"
            )
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(WORKER)
    steps = [
        Step("implement", StepKind.IMPLEMENTATION, "create known.txt"),
        Step(
            "gate", StepKind.DETERMINISTIC_GATE, "", command=["test", "-f", "known.txt"]
        ),
        Step("review", StepKind.JUDGMENT_REVIEW, "review known.txt"),
    ]
    command = [sys.executable, str(worker_script)]
    plan = CampaignPlan(
        "e2e",
        str(repo),
        ["known.txt"],
        steps,
        {"command": command, "provider": "fake", "model": "fake-writer"},
        {"command": command, "provider": "fake", "model": "fake-reviewer"},
        {"max_processed_tokens_per_run": 20, "max_processed_tokens_per_day": 30},
    )
    store = ConductorStore(hermes_home / "conductor" / "state.sqlite")
    store.create_campaign(plan)
    fake_tmux = tmp_path / "fake_tmux.py"
    fake_tmux.write_text(FAKE_TMUX)
    fake_tmux.chmod(0o700)
    launcher = TmuxLauncher(tmux=str(fake_tmux), socket_path=tmp_path / "tmux.sock")
    engine = Conductor(store, launcher)
    assert engine.tick("e2e") is TickResult.LAUNCHED_WRITER
    writer = store.active_worker("e2e")
    results = wait_tick(engine, "e2e", TickResult.LAUNCHED_REVIEWER)
    reviewer = store.active_worker("e2e")
    assert reviewer.worker_id != writer.worker_id
    assert reviewer.tmux_session != writer.tmux_session
    assert reviewer.read_only
    wait_tick(engine, "e2e", TickResult.COMPLETE)
    assert (repo / "known.txt").read_text() == "created by governed worker\n"
    usage = store.daily_usage("e2e")
    assert usage["processed_tokens"] == 20
    assert usage["runs"] == 2
    outputs = [
        json.loads(Path(path).read_text())
        for path in (writer.output_path, reviewer.output_path)
    ]
    assert sum(item["model_calls"] for item in outputs) == 2
    assert all(
        result
        in {TickResult.WAITING_STALE, TickResult.ADVANCED, TickResult.LAUNCHED_REVIEWER}
        for result in results
    )
    assert not launcher.is_running(writer.tmux_session)
    assert not launcher.is_running(reviewer.tmux_session)
