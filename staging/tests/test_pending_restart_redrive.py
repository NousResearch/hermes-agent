#!/usr/bin/env python3
"""Tests for the A1b pending-restart re-drive in runtime-parity-check.py.

Behavior contract (real temp HERMES_HOME, notify + fleet-restart stubbed):
  - empty/absent ledger -> silent (no notifications);
  - a pending+idle label -> re-driven, VERIFIED clears its ledger entry, 1 #logs;
  - a pending+busy label -> stays pending, NOT re-driven (INV-1 idle-only);
  - a stuck pending (age+attempts past threshold) -> #alerts ONCE, not per-tick
    (edge-trigger asserted by driving two consecutive ticks -> single send).

Run: /usr/bin/python3 staging/tests/test_pending_restart_redrive.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # .../staging
MOD_PATH = REPO / "scripts" / "runtime-parity-check.py"

PASS, FAIL = "\u2705", "\u274c"
_ok = True


def check(label, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  {PASS if cond else FAIL} {label}")


def load_mod(home: Path):
    os.environ["HERMES_HOME"] = str(home)
    os.environ["RPC_TREE"] = str(home / "runtime" / "hermes-agent")
    spec = importlib.util.spec_from_file_location(f"rpc_{uuid.uuid4().hex}", MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_home():
    td = Path(tempfile.mkdtemp(prefix="redrive-"))
    (td / "state").mkdir(parents=True, exist_ok=True)
    (td / "fleet").mkdir(parents=True, exist_ok=True)
    return td


def set_idle(home, profile, active):
    if profile == "default":
        sp = home / "gateway_state.json"
    else:
        sp = home / "profiles" / profile / "gateway_state.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"active_agents": active}))


def instrument(m, verify_labels=None):
    """Stub _notify (capture) and _redrive_one (VERIFY only listed labels)."""
    sent = []
    m._notify = lambda ch, title, body: (sent.append((ch, title, body)), True)[1]
    vok = set(verify_labels or [])
    m._redrive_one = lambda label: label in vok
    return sent


# ---------------------------------------------------------------------------
print("\nTEST 1 — empty/absent ledger -> silent")
home = base_home(); m = load_mod(home); sent = instrument(m)
m._redrive_pending_restarts()
check("no ledger -> 0 notifications", len(sent) == 0)


# ---------------------------------------------------------------------------
print("\nTEST 2 — pending + IDLE + verify-ok -> re-driven, entry cleared, 1 #logs")
home = base_home(); m = load_mod(home)
sent = instrument(m, verify_labels=["ai.hermes.gateway-aegis"])
set_idle(home, "aegis", 0)
(home / "state" / "pending-gateway-restart.json").write_text(json.dumps({
    "ai.hermes.gateway-aegis": {"target_sha": "abc123", "since_epoch": "1",
                                 "first_skipped": int(time.time()), "attempts": 0}}))
m._redrive_pending_restarts()
led = json.loads((home / "state" / "pending-gateway-restart.json").read_text())
check("verified label cleared from ledger", "ai.hermes.gateway-aegis" not in led)
check("re-drive -> exactly 1 #logs", len([s for s in sent if s[0] == m.LOGS]) == 1)
check("re-drive -> 0 #alerts", len([s for s in sent if s[0] == m.ALERTS]) == 0)


# ---------------------------------------------------------------------------
print("\nTEST 3 — pending + BUSY -> stays pending, NOT re-driven (INV-1)")
home = base_home(); m = load_mod(home)
calls = {"n": 0}
sent = instrument(m)
def _count_redrive(label):
    calls["n"] += 1
    return True
m._redrive_one = _count_redrive
set_idle(home, "aegis", 2)  # busy
(home / "state" / "pending-gateway-restart.json").write_text(json.dumps({
    "ai.hermes.gateway-aegis": {"target_sha": "abc123", "since_epoch": "1",
                                 "first_skipped": int(time.time()), "attempts": 0}}))
m._redrive_pending_restarts()
led = json.loads((home / "state" / "pending-gateway-restart.json").read_text())
check("busy label NOT re-driven (redrive_one uncalled)", calls["n"] == 0)
check("busy label stays pending in ledger", "ai.hermes.gateway-aegis" in led)
check("busy -> silent (no #logs/#alerts)", len(sent) == 0)


# ---------------------------------------------------------------------------
print("\nTEST 4 — stuck pending (old + retried) -> #alerts ONCE, edge-triggered")
home = base_home(); m = load_mod(home)
sent = instrument(m, verify_labels=[])  # re-drive never verifies (stuck)
set_idle(home, "aegis", 0)  # idle, so it retries every tick and keeps failing
old = int(time.time()) - (7 * 3600)  # 7h ago, past 6h threshold
(home / "state" / "pending-gateway-restart.json").write_text(json.dumps({
    "ai.hermes.gateway-aegis": {"target_sha": "abc123", "since_epoch": "1",
                                 "first_skipped": old, "attempts": 3}}))
m._redrive_pending_restarts()
n1 = len([s for s in sent if s[0] == m.ALERTS])
m._redrive_pending_restarts()  # second consecutive tick
n2 = len([s for s in sent if s[0] == m.ALERTS])
check("stuck -> exactly 1 #alerts on first tick", n1 == 1)
check("stuck -> still 1 #alerts after second tick (edge-triggered, not per-tick)", n2 == 1)


# ---------------------------------------------------------------------------
print("\nTEST 5 — idle+not-yet-stuck+verify-fail -> silent, attempts increments")
home = base_home(); m = load_mod(home)
sent = instrument(m, verify_labels=[])  # never verifies, but young
set_idle(home, "aegis", 0)
(home / "state" / "pending-gateway-restart.json").write_text(json.dumps({
    "ai.hermes.gateway-aegis": {"target_sha": "abc123", "since_epoch": "1",
                                 "first_skipped": int(time.time()), "attempts": 0}}))
m._redrive_pending_restarts()
led = json.loads((home / "state" / "pending-gateway-restart.json").read_text())
check("young+failing -> silent (not stuck yet)", len(sent) == 0)
check("young+failing -> stays pending", "ai.hermes.gateway-aegis" in led)
check("attempts incremented on idle re-drive", led["ai.hermes.gateway-aegis"]["attempts"] == 1)


# ---------------------------------------------------------------------------
print("\n" + ("ALL RE-DRIVE TESTS PASSED " + PASS if _ok else "SOME TESTS FAILED " + FAIL) + "\n")
sys.exit(0 if _ok else 1)
