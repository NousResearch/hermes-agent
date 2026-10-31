"""Cross-process falsification of the per-session fence.

The unit tests share one interpreter, so they cannot see the failure this whole
change exists to prevent: two SEPARATE gateway processes, each with its own
snapshot of a conversation, both writing to it. That is how the defect was found
and it is the only way to prove it is closed.

Run against the fork's own HERMES_HOME so nothing here touches a real profile:

    python scripts/probe_active_session_exclusivity.py

It drives real ``python -m tui_gateway.entry`` processes over stdio JSON-RPC.
The standalone entry now reserves the home's ``gateway.lock`` (the gateway's own
owner reservation) before it opens ``state.db``, so the second writer is refused
before it exists rather than per turn:

    A  create S, submit          -> claims the session
    B  start on the same home    -> refused at startup (exit 1), never ready
    A  exits                     -> its reservation dies with it
    B  start, resume S, submit   -> succeeds

No provider is required. The fence is checked BEFORE the agent is built, so a
submit that later fails for want of a model still proves who owns the session --
which is the property under test, and keeps the probe free of credentials and of
inference cost.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # posix layout
    PYTHON = REPO / "venv" / "bin" / "python"


class Gateway:
    """One gateway process, spoken to the way the TUI speaks to it."""

    def __init__(self, name: str, home: Path):
        env = dict(os.environ)
        env["HERMES_HOME"] = str(home)
        env["PYTHONUNBUFFERED"] = "1"
        self.name = name
        self.proc = subprocess.Popen(
            [str(PYTHON), "-u", "-m", "tui_gateway.entry"],
            cwd=str(REPO),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._next = 1
        self.ready()

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"[{self.name}] gateway closed its pipe")
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read()
            if msg and msg.get("method") == "event":
                if msg.get("params", {}).get("type") == "gateway.ready":
                    return
        raise RuntimeError(f"[{self.name}] never announced gateway.ready")

    def call(self, method: str, params: dict, timeout: float = 180.0) -> dict:
        rid = str(self._next)
        self._next += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read()
            if msg and msg.get("id") == rid:
                return msg
        raise RuntimeError(f"[{self.name}] timed out calling {method}")

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def registry(home: Path):
    path = home / "runtime" / "active_sessions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")).get("entries", [])
    except Exception:
        return []


def main() -> int:
    home = REPO / ".probe-home"
    # A fresh profile each run: a lease left by a previous run would make the
    # first check pass or fail for the wrong reason.
    import shutil

    shutil.rmtree(home, ignore_errors=True)
    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    a = Gateway("A", home)
    b = None
    try:
        created = a.call("session.create", {"cols": 80})
        sid_a = created["result"]["session_id"]

        # Opening a chat must not claim anything -- an idle composer is invisible
        # and a slot held by one would fence a real turn for no reason.
        check("session.create claims nothing", registry(home) == [], f"{len(registry(home))} entries")

        a.call("prompt.submit", {"session_id": sid_a, "text": "probe: A takes the session"})
        held = registry(home)
        check("A's first turn claims a session", len(held) == 1, json.dumps(held)[:200])
        if not held:
            raise RuntimeError("A never claimed anything; nothing further can be tested")

        # The STORED key, which only materialises when a turn is first submitted --
        # and which is what the lease must be keyed on. A lease keyed on the live
        # runtime id would fence nothing: two processes resuming one conversation
        # have different runtime ids by construction.
        key = held[0].get("session_id")
        print(f"A live session {sid_a}, stored key {key}")
        check("the lease is keyed on the STORED session, not the runtime handle",
              bool(key) and key != sid_a, f"key={key} runtime={sid_a}")

        refused_b = subprocess.run(
            [str(PYTHON), "-u", "-m", "tui_gateway.entry"], cwd=str(REPO),
            env={**os.environ, "HERMES_HOME": str(home)}, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False)
        check("B cannot start a second writer on A's home", refused_b.returncode == 1
              and "gateway.ready" not in refused_b.stdout, (refused_b.stderr or "")[-200:])

        # A dies without releasing -- the crash case, not a clean handoff.
        a.proc.kill()
        a.proc.wait(timeout=30)

        b = Gateway("B", home)
        resumed = b.call("session.resume", {"session_id": key})
        check("after A dies, B may resume", "result" in resumed,
              json.dumps(resumed.get("error", ""))[:160])
        sid_b = resumed.get("result", {}).get("session_id")
        retried = b.call("prompt.submit", {"session_id": sid_b, "text": "probe: B may write now"})
        check("and B's submit is accepted", retried.get("error") is None,
              json.dumps(retried.get("error", ""))[:200])
    finally:
        if b is not None:
            b.close()
        a.close()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All cross-process checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
