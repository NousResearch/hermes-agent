"""Auto-decomposition routing against a real gateway and isolated Hermes home.

The fake auxiliary model deliberately names an excluded profile. The gateway must keep that name
out of the prompt and route the returned choice through the same eligible-profile check before it
can write child cards. Explicitly assigned cards still use the dispatcher's independent allowlist.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from tests.e2e.core.kanban._helpers import PY, Board, wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups for the gateway child"),
]

GATEWAY_CONFIG = (
    "kanban:\n"
    "  dispatch_in_gateway: true\n"
    "  dispatch_interval_seconds: 1\n"
    "  auto_decompose: true\n"
    "  max_spawn: 0\n"
    "  auto_promote_children: false\n"
    "  dispatch_profiles: [default, orchestrator, homelabadmin, reviewer]\n"
    "  auto_decompose_excluded_profiles: [reviewer]\n"
    "  orchestrator_profile: orchestrator\n"
    "  default_assignee: orchestrator\n"
)


class ExcludedProfileModel:
    """Return a graph that tests both the valid route and the excluded-profile fallback."""

    def __call__(self, _record: dict) -> Text:
        return Text(json.dumps({
            "fanout": True,
            "rationale": "Split the remote administration work by deliverable.",
            "tasks": [
                {"title": "review the handoff", "body": "Review the task handoff.",
                 "assignee": "reviewer", "parents": []},
                {"title": "inventory the homelab", "body": "Inventory the remote services.",
                 "assignee": "homelabadmin", "parents": []},
            ],
        }))


def seed_profile(board: Board, name: str, description: str) -> None:
    profile = board.hermes_home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {board.base_url}\n"
        "  default: fake-model\n",
        encoding="utf-8",
    )
    (profile / "profile.yaml").write_text(f"description: {description}\n", encoding="utf-8")


@contextmanager
def gateway(board: Board) -> Iterator[subprocess.Popen]:
    env = board.env()
    env.update({"HERMES_GATEWAY_LOCK_DIR": str(board.root / "gw-locks"), "PYTHONUNBUFFERED": "1"})
    log_path = board.root / "gateway.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [PY, "-m", "hermes_cli.main", "gateway", "run"], cwd=str(board.root), env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
    try:
        yield proc
    finally:
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            if proc.poll() is not None:
                break
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                continue


def gateway_tail(board: Board) -> str:
    path = board.root / "gateway.log"
    return path.read_text(encoding="utf-8", errors="replace")[-3000:] if path.exists() else "(no gateway log)"


def test_gateway_routes_only_to_eligible_profiles(tmp_path: Path) -> None:
    model = ExcludedProfileModel()
    with FakeLLMServer(aux=model) as server:
        board = Board(tmp_path, server.base_url, extra_config=GATEWAY_CONFIG)
        seed_profile(board, "orchestrator", "Owns decomposition and coordinates the work.")
        seed_profile(board, "homelabadmin", "Administers the remote home lab over SSH.")
        seed_profile(board, "reviewer", "Reviews contract-bound pull requests.")
        root = board.create("inventory and review a homelab handoff", "--triage")

        with gateway(board) as proc:
            wait_until(
                lambda: proc.poll() is not None or board.events(root, "decomposed"), 120,
                f"gateway decomposes the triage task\n{gateway_tail(board)}",
            )
            assert proc.poll() is None, f"gateway exited rc={proc.returncode}\n{gateway_tail(board)}"

        decomposed = board.events(root, "decomposed")
        assert len(decomposed) == 1, board.diag(root)
        child_ids = decomposed[0]["payload"]["child_ids"]
        children = [board.task(child_id) for child_id in child_ids]
        assert [child["assignee"] for child in children] == ["orchestrator", "homelabadmin"]
        assert all(child["worker_pid"] is None for child in children), board.diag(root)
        assert all(board.runs(child_id) == [] for child_id in child_ids), board.diag(root)
        assert len(server.aux_requests()) == 1, "the decomposed root must not be billed again"
        prompt = json.dumps(server.aux_requests()[0].get("messages", []))
        assert "reviewer" not in prompt
