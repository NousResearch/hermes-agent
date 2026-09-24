"""Real-process Kanban board harness: a scratch HOME/HERMES_HOME, the recording fake provider, and
the real ``hermes kanban`` CLI (dispatcher ticks spawn real ``hermes chat -q`` workers).

Nothing below imports the dispatcher: every board mutation goes through a child process and every
verdict is read back from ``kanban.db`` (tasks / task_runs / task_events rows) or the files on disk.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[4]
PY = sys.executable

# Clocks shrunk through the documented env knobs, never by patching code.
FAST_ENV = {
    "HERMES_KANBAN_CRASH_GRACE_SECONDS": "0",
    "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS": "0",
}


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = pred()
        if val:
            return val
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError):
        return False
    # A zombie child of ours still answers kill(0); /proc tells the truth.
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


@dataclass
class Board:
    """One isolated Kanban install driven only through real child processes."""

    root: Path
    base_url: str
    extra_config: str = ""
    env_extra: dict[str, str] = field(default_factory=dict)
    spawned_pids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.home = self.root / "home"
        self.hermes_home = self.home / ".hermes"
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        (self.hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: custom\n"
            f"  base_url: {self.base_url}\n"
            "  default: fake-model\n"
            "  context_length: 128000\n"
            "agent:\n"
            "  api_max_retries: 1\n"
            "updates:\n"
            "  check: false\n"
            + self.extra_config,
            encoding="utf-8",
        )
        (self.hermes_home / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e\n", encoding="utf-8")
        # Workers run with cwd=<task workspace>; the wrapper pins THIS checkout on the path.
        self.hermes_bin = self.root / "hermes-bin"
        self.hermes_bin.write_text(
            "#!/bin/sh\n"
            f"PYTHONPATH={REPO} exec {PY} -m hermes_cli.main \"$@\"\n", encoding="utf-8")
        self.hermes_bin.chmod(0o755)

    # env / processes -------------------------------------------------------
    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("HERMES_") and not k.endswith(("_API_KEY", "_TOKEN"))}
        env.pop("PYTEST_CURRENT_TEST", None)
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home),
            "HERMES_BIN": str(self.hermes_bin), "PYTHONPATH": str(REPO),
            "NO_COLOR": "1", "TERM": "dumb",
            # Children keep pytest's PYTEST_VERSION, which arms the live-DB guard against the scratch
            # HOME's own state.db; the whole tree is under ``root`` (asserted below), so let workers
            # open their real session store.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            **FAST_ENV, **self.env_extra,
        })
        assert env["HERMES_HOME"].startswith(str(self.root))
        return env

    def cli(self, *args: str, timeout: float = 90.0, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [PY, "-m", "hermes_cli.main", "kanban", *args], cwd=str(self.root), env=self.env(),
            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
        )
        if check and proc.returncode != 0:
            raise AssertionError(f"kanban {' '.join(args)} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        return proc

    def cli_json(self, *args: str, **kw: Any) -> Any:
        out = self.cli(*args, "--json", **kw).stdout
        return json.loads(out[out.index("{"):] if "{" in out else out)

    def create(self, title: str, *extra: str) -> str:
        return self.cli_json("create", title, "--assignee", "default", *extra)["id"]

    def dispatch(self, *extra: str) -> dict:
        res = self.cli_json("dispatch", *extra)
        for tid in [s["task_id"] for s in res.get("spawned", [])]:
            pid = self.task(tid)["worker_pid"]
            if pid:
                self.spawned_pids.add(int(pid))
        return res

    def kill_workers(self) -> None:
        for pid in list(self.spawned_pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # state readback --------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.hermes_home / "kanban.db"

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def task(self, tid: str) -> dict:
        rows = self._q("SELECT * FROM tasks WHERE id = ?", (tid,))
        assert rows, f"task {tid} row is gone"
        return rows[0]

    def tasks(self) -> list[dict]:
        return self._q("SELECT * FROM tasks ORDER BY created_at, id")

    def runs(self, tid: str) -> list[dict]:
        return self._q("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,))

    def events(self, tid: str, kind: Optional[str] = None) -> list[dict]:
        rows = self._q("SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,))
        for r in rows:
            try:
                r["payload"] = json.loads(r.get("payload") or "null")
            except (TypeError, ValueError):
                pass
        return [r for r in rows if kind is None or r["kind"] == kind]

    def worker_log(self, tid: str) -> str:
        hits = list(self.hermes_home.rglob(f"{tid}.log"))
        return hits[0].read_text(encoding="utf-8", errors="replace") if hits else ""

    def wait_worker_exit(self, tid: str, pid: int, timeout: float = 90.0) -> None:
        wait_until(lambda: not pid_alive(pid), timeout,
                   f"worker {pid} of {tid} to exit; log tail:\n{self.worker_log(tid)[-2000:]}")

    def diag(self, tid: str) -> str:
        return (f"task={json.dumps(self.task(tid), default=str)[:1500]}\n"
                f"runs={json.dumps(self.runs(tid), default=str)[:2500]}\n"
                f"events={[(e['kind'], e['payload']) for e in self.events(tid)][-15:]}\n"
                f"log tail:\n{self.worker_log(tid)[-2500:]}")
