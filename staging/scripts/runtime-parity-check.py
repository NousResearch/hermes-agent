#!/usr/bin/env python3
"""runtime-parity-check.py — daily runtime-tree parity monitor (UPGRADE of the old
runtime-tree-status hygiene cron). SPEC: ~/.hermes/plans/2026-07-01_runtime-parity-watchdog-SPEC.md v0.4.

Kills the silent un-deploy class (#154): a runtime tree behind fork/main = an un-deployed merge →
🛑 #alerts (was: never tripped at behind-limit 500, never fetched, #logs-only). Preserves the
DIRTY_IMPORT_PATH hygiene advisory → #logs. Read-only (fetch + rev-list); never mutates the tree.

Replaces the routing logic of fleet/runtime-tree-status-cron.sh (same launchd label). no_agent.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERMES = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
TREE = Path(os.environ.get("RPC_TREE", str(HERMES / "runtime" / "hermes-agent")))
DEPLOY_REF = os.environ.get("RPC_DEPLOY_REF", "fork/main")
DETECTOR = HERMES / "fleet" / "runtime_tree_status.py"
NOTIFY = HERMES / "scripts" / "notify.py"
PY = "/usr/bin/python3"
ALERTS = os.environ.get("RPC_ALERTS", "1480528231286181948")   # #alerts
LOGS = os.environ.get("RPC_LOGS", "1480525090331561984")       # #logs
STATE = HERMES / "state" / "runtime-parity.json"
DONE_MARKER = HERMES / "state" / "runtime-parity.done"
LOCK = HERMES / "state" / "runtime-parity.lock"
PENDING_LEDGER = HERMES / "state" / "pending-gateway-restart.json"
FLEET_RESTART = HERMES / "fleet" / "fleet-gateway-restart.sh"
# A pending gateway that keeps failing to come current is a real stuck state:
# escalate to #alerts once it has been pending this long AND been retried at
# least this many times (edge-triggered, not per-tick).
REDRIVE_STUCK_AGE_S = int(os.environ.get("RPC_REDRIVE_STUCK_AGE_S", str(6 * 3600)))
REDRIVE_STUCK_ATTEMPTS = int(os.environ.get("RPC_REDRIVE_STUCK_ATTEMPTS", "3"))
DEADMAN_LABEL = "ai.hermes.runtime-parity-deadman"
AGE_REESCALATE_S = 24 * 3600  # INV-5: re-alert a stable gap every 24h

# pass-4 caution 1: derive the literal kind strings FROM the detector source at runtime, so a
# rename/hyphen drift in the detector fails our routing loudly instead of silently mis-routing.
def _detector_kinds() -> dict:
    try:
        src = DETECTOR.read_text()
        kinds = set(re.findall(r'"kind":\s*"([A-Z_]+)"', src))
    except Exception:
        kinds = set()
    need = {"OFF_DEPLOY_REF_BEHIND", "OFF_DEPLOY_REF_AHEAD", "DIRTY_IMPORT_PATH"}
    missing = need - kinds
    if missing:
        # detector changed its kind vocabulary — fail loud, don't silently mis-route (pass-3 B1)
        _notify(ALERTS, "🛑 runtime-parity-check: detector kind drift",
                f"Detector `runtime_tree_status.py` no longer emits {sorted(missing)} — routing would "
                f"silently break. Reconcile runtime-parity-check.py with the detector.")
        sys.exit(5)
    return {"behind": "OFF_DEPLOY_REF_BEHIND", "ahead": "OFF_DEPLOY_REF_AHEAD", "dirty": "DIRTY_IMPORT_PATH"}


def _notify(channel: str, title: str, body: str) -> bool:
    try:
        r = subprocess.run([PY, str(NOTIFY), "--send", f"**{title}**\n{body}",
                            "--channel", "discord", "--target", channel],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(TREE), *args], capture_output=True, text=True)


def _fetch_verified() -> tuple[bool, str]:
    """INV-1: fetch fork with retry; verify the ref actually resolved (not just exit-0).
    Returns (ok, reason). ok=False distinguishes transient (retry-exhausted) vs config-broken."""
    remote = DEPLOY_REF.split("/", 1)[0]
    for attempt in range(3):
        r = _git("fetch", remote, "--quiet")
        if r.returncode == 0:
            # verify the deploy ref resolves to a real sha post-fetch (auth-expired-cached-ref guard)
            rp = _git("rev-parse", DEPLOY_REF)
            if rp.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", rp.stdout.strip() or ""):
                return True, rp.stdout.strip()
        time.sleep(2 ** attempt)
    # distinguish config-broken (remote/auth) from transient
    remotes = _git("remote").stdout.split()
    if remote not in remotes:
        return False, "config-broken: remote '%s' not configured" % remote
    return False, "transient-or-auth: fetch/rev-parse failed after 3 tries"


def _detect() -> dict:
    r = subprocess.run([PY, str(DETECTOR), "--tree", str(TREE), "--deploy-ref", DEPLOY_REF,
                        "--behind-limit", "0", "--ahead-limit", "500", "--json"],
                       capture_output=True, text=True)
    # true exit contract: rc0 clean / rc1 findings / rc2 detector-error (pass-3 B2)
    try:
        d = json.loads(r.stdout)
    except Exception:
        d = {"ok": False, "clean": None, "findings": [], "_rc": r.returncode,
             "_err": (r.stderr or r.stdout)[-200:]}
    d["_rc"] = r.returncode
    return d


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"gaps": {}}  # fail-safe: don't reset-and-storm


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(STATE)


def _self_heal_deadman() -> None:
    """INV-7: if the deadman launchd job is present-but-unloaded, re-bootstrap it + alert.
    No infinite regress: deadman watches monitor, monitor heals deadman."""
    uid = os.getuid()
    r = subprocess.run(["launchctl", "print", f"gui/{uid}/{DEADMAN_LABEL}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return  # loaded + healthy
    plist = Path.home() / "Library" / "LaunchAgents" / f"{DEADMAN_LABEL}.plist"
    if plist.exists():
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)],
                       capture_output=True, text=True)
        _notify(ALERTS, "⚠️ runtime-parity: deadman was unloaded",
                f"The parity deadman `{DEADMAN_LABEL}` was present-but-unloaded; re-bootstrapped it.")


WRAPPER_GUARD = HERMES / "fleet" / "hermes-wrapper-guard.py"
HEAL_RATE_ALERT_THRESHOLD = int(os.environ.get("RPC_HEAL_RATE_MAX", "2"))  # >N heals in 7d → #alerts (D-7)


def _check_gateway_shadows() -> None:
    """D-6 (SPEC 2026-07-01-alerts-batch): flag ANY live gateway importing FOSSIL first-party
    code via a CWD shadow.

    ``python -m hermes_cli.main … gateway run`` puts the process CWD on ``sys.path[0]``, so a
    gateway whose launchd WorkingDirectory contains a top-level ``hermes_cli/`` (i.e. the stale
    ``~/.hermes/hermes-agent`` dev tree) silently imports 269-commit-old code even though its
    editable venv points at runtime. This bit aegis+argus (2026-07-01): they ran fossil
    ``gateway/run.py`` (no ``reboot_interrupted``) + ``plugins/memory/mem0`` (no
    ``resolve_capture``). The AC-12 / mem0-drift guards only inspect the DEFAULT gateway, so
    they never saw it — this is the all-gateways generalization.

    Routing: 0 shadows → silent; ≥1 shadow → #alerts (one line per offender + the fix). Never
    raises into the parity check (a guard failure must not take down the monitor).
    """
    checker = HERMES / "scripts" / "lib" / "gateway_shadow_check.py"
    if not checker.is_file():
        return
    try:
        p = subprocess.run([PY, str(checker)], capture_output=True, text=True, timeout=90)
        res = json.loads((p.stdout or "").strip() or "{}")
    except Exception as e:  # noqa: BLE001 — a guard failure must never crash the parity monitor
        _notify(LOGS, "gateway-shadow-check: run error", f"could not run gateway_shadow_check: {e}")
        return
    bad = res.get("shadowed", [])
    if not bad:
        return  # all gateways import from runtime — silent
    lines = ["🛑 A live gateway is importing FOSSIL code via a CWD shadow "
             "(python -m puts cwd on sys.path[0]):"]
    for r in bad:
        prof = "default"
        cmd = r.get("cmd", "")
        if "--profile" in cmd:
            try:
                prof = cmd.split("--profile", 1)[1].split()[0]
            except Exception:
                pass
        lines.append(f"  • {prof} (pid {r.get('pid')}): cwd={r.get('cwd')}")
    lines.append("Fix: repoint that gateway's launchd WorkingDirectory to its profile dir (or a "
                 "cwd with NO top-level hermes_cli/), then bootout+bootstrap it.")
    _notify(ALERTS, "🛑 gateway fossil-shadow (un-deployed code running live)", "\n".join(lines))


def _check_stale_gateways() -> None:
    """P3 (SPEC 2026-07-01 pr122-profile-home-fix): flag gateways running PRE-DEPLOY code.

    Import-time binding means a deploy never reaches a running process — a gateway that
    STARTED before the runtime tree's current HEAD adoption (the reflog epoch, NOT ``%ct``
    and NOT module mtime; see gateway_shadow_check.deployed_adoption_epoch) runs older
    code than deployed until restarted. This left athena/daedalus authoring skills with
    pre-#122 create-code (2026-07-01).

    Routing: 0 stale → silent; ≥1 stale → **#logs** (advisory — it self-clears on the next
    natural restart; NOT #alerts, unlike the fossil-shadow which is a correctness bug).
    Reflog unreadable → one-line #logs breadcrumb (degradation observable, never a false
    page). Never raises into the parity check.
    """
    checker = HERMES / "scripts" / "lib" / "gateway_shadow_check.py"
    if not checker.is_file():
        return
    try:
        p = subprocess.run([PY, str(checker), "--stale-check"],
                           capture_output=True, text=True, timeout=90)
        res = json.loads((p.stdout or "").strip() or "{}")
    except Exception as e:  # noqa: BLE001
        _notify(LOGS, "gateway-stale-check: run error", f"could not run stale check: {e}")
        return
    if "stale" not in res:
        return
    if res.get("deployed_epoch") is None:
        _notify(LOGS, "gateway-stale-check: degraded",
                "runtime tree reflog unreadable — stale-gateway detection is OFF this tick "
                "(not flagging anything). Check the runtime tree's .git health.")
        return
    stale = res.get("stale") or []
    if not stale:
        return  # all gateways post-date the deployed code — silent
    lines = [f"ℹ️ {len(stale)} gateway(s) running PRE-DEPLOY code (started before the runtime "
             f"tree's HEAD adoption {res['deployed_epoch']}); self-clears on their next restart:"]
    for r in stale:
        prof = "default"
        cmd = r.get("cmd", "")
        if "--profile" in cmd:
            try:
                prof = cmd.split("--profile", 1)[1].split()[0]
            except Exception:
                pass
        lines.append(f"  • {prof} (pid {r.get('pid')}): {r.get('reason')}")
    lines.append("Advisory: kickstart them when idle to load the deployed code, or let the next "
                 "natural restart pick it up.")
    _notify(LOGS, "gateway stale-deployed-code advisory", "\n".join(lines))


def _load_ledger() -> dict:
    try:
        d = json.loads(PENDING_LEDGER.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}  # fail-safe: absent/corrupt ledger => nothing pending, never crash


def _save_ledger(d: dict) -> None:
    PENDING_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = PENDING_LEDGER.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(PENDING_LEDGER)


def _redrive_one(label: str) -> bool:
    """Restart+verify a single gateway via the shared fleet-gateway-restart.sh
    (INV-1 idle-only, INV-2 verify, INV-3 dynamic enum all inherited). Returns
    True iff the fleet script reports it VERIFIED. Never raises."""
    if not FLEET_RESTART.is_file():
        return False
    try:
        p = subprocess.run(["bash", str(FLEET_RESTART), "--only", label, "--json"],
                           capture_output=True, text=True, timeout=180)
        res = json.loads((p.stdout or "").strip() or "{}")
    except Exception:
        return False
    # verified>=1 and this label's result line says VERIFIED
    if int(res.get("verified", 0)) < 1:
        return False
    for line in res.get("results", []):
        if line.startswith(label + ":") and "VERIFIED" in line:
            return True
    return False


def _redrive_pending_restarts() -> None:
    """A1b: close the busy-skip gap. Read the pending-restart ledger + the live
    stale-set; for each gateway that is pending (or provably stale) AND now
    IDLE, restart+verify it via fleet-gateway-restart.sh and clear its entry.
    A gateway still busy stays pending (no amputation — INV-1). A gateway that
    stays pending past the age+attempt threshold escalates to #alerts once
    (edge-triggered). Routing: nothing re-driven and nothing stuck => silent
    (house rule); ≥1 brought to VERIFIED => single #logs line; a stuck pending
    => #alerts (age+attempt gated). Never raises into the parity check.
    """
    ledger = _load_ledger()
    if not ledger:
        return  # nothing pending — silent (also covers absent/corrupt ledger)

    now = int(time.time())

    # Live idle state per profile (active_agents==0). Reuse fleet's convention:
    # default -> gateway_state.json at HERMES root; others under profiles/<p>/.
    def _profile_of(label: str) -> str:
        if label == "ai.hermes.gateway":
            return "default"
        if label.startswith("ai.hermes.gateway-"):
            return label[len("ai.hermes.gateway-"):]
        return ""

    def _is_idle(label: str) -> bool:
        prof = _profile_of(label)
        if not prof:
            return False
        sp = (HERMES / "gateway_state.json") if prof == "default" \
            else (HERMES / "profiles" / prof / "gateway_state.json")
        try:
            v = json.loads(sp.read_text()).get("active_agents")
            return int(v) == 0
        except Exception:
            return False  # missing/corrupt/busy => not idle (INV-1: never amputate)

    redrove: list[str] = []
    stuck: list[tuple[str, dict]] = []
    changed = False

    for label in list(ledger.keys()):
        entry = ledger[label] if isinstance(ledger.get(label), dict) else {}
        if not _is_idle(label):
            # still busy — stays pending; check only for stuck-escalation below
            pass
        else:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            ledger[label] = entry
            changed = True
            if _redrive_one(label):
                del ledger[label]
                redrove.append(label)
                continue
        # not idle, or re-drive didn't verify -> evaluate stuck escalation
        try:
            first = int(entry.get("first_skipped", now))
        except Exception:
            first = now
        age = now - first
        if age >= REDRIVE_STUCK_AGE_S and int(entry.get("attempts", 0)) >= REDRIVE_STUCK_ATTEMPTS \
                and not entry.get("alerted"):
            stuck.append((label, entry))
            entry["alerted"] = True
            ledger[label] = entry
            changed = True

    if changed:
        _save_ledger(ledger)

    if redrove:
        _notify(LOGS, "gateway pending-restart re-driven",
                "Brought %d busy-skipped gateway(s) current once idle (loaded the deployed "
                "commit): %s. Pending-restart ledger cleared for them."
                % (len(redrove), ", ".join(redrove)))
    for label, entry in stuck:
        _notify(ALERTS, "🛑 gateway stuck pending-restart",
                f"`{label}` has been pending a deploy-skew restart for "
                f"{(now - int(entry.get('first_skipped', now))) // 3600}h across "
                f"{entry.get('attempts', 0)} idle-window attempts and still won't come "
                f"current (target sha {entry.get('target_sha', '?')[:8]}). It should have been "
                f"restartable — a human must chase it: `bash ~/.hermes/fleet/fleet-gateway-restart.sh "
                f"--only {label}` and read its FAILED reason.")


def _check_wrapper() -> None:
    """Wrapper-integrity guard fold (SPEC 2026-07-01-hermes-wrapper-integrity-guard, D-3).

    Self-heals a clobbered ~/.local/bin/hermes (a hermes/pip/console-script reinstall — notably a desktop
    build — can revert it to a bare dev-venv exec, silently re-pinning the dev venv). Runs FIRST in run(),
    before every early return, since a wrapper clobber is independent of fork/main parity. Routing (INV-4):
      correct/no-op        -> silent
      healed + verified    -> #logs (single) ; heal-rate > threshold -> #alerts (recurrence = a real decision)
      un-healable/failed   -> #alerts (one line: fix perms / restore source / build the runtime venv)
    Never raises into the parity check — a guard error must not take down the parity monitor.
    """
    if not WRAPPER_GUARD.is_file():
        return
    try:
        p = subprocess.run([PY, str(WRAPPER_GUARD), "--reassert", "--json"],
                           capture_output=True, text=True, timeout=120)
        res = json.loads((p.stdout or "").strip() or "{}")
    except Exception as e:  # noqa: BLE001 — a guard failure must never crash the parity monitor
        _notify(LOGS, "wrapper-guard: run error", f"could not run hermes-wrapper-guard: {e}")
        return

    state = res.get("state", "unknown")
    rate = int(res.get("heal_rate_7d", 0))
    if res.get("ok") and not res.get("healed"):
        return  # correct — silent
    if res.get("ok") and res.get("healed"):
        if rate > HEAL_RATE_ALERT_THRESHOLD:
            _notify(ALERTS, "🛑 hermes wrapper clobbered repeatedly",
                    f"~/.local/bin/hermes drifted off-runtime and was auto-healed, but this is the "
                    f"**{rate}th heal in 7d** — the upstream writer keeps clobbering it. A recurring heal is "
                    f"the decision to page on: find + fix the writer (likely a desktop build / console-script "
                    f"reinstall). Log: `~/.hermes/state/wrapper-guard/heals.jsonl`.")
        else:
            _notify(LOGS, "wrapper-guard: healed a drifted hermes wrapper",
                    f"~/.local/bin/hermes had drifted off-runtime; auto-healed + verified "
                    f"(reason: {res.get('reason','')}). Heal-rate 7d: {rate}. Investigate if recurring.")
        return
    # not ok → un-healable / verify-failed-and-restored
    _notify(ALERTS, "🛑 hermes wrapper off-runtime and un-healable",
            f"~/.local/bin/hermes is drifted and the guard could NOT heal it (state: {state}): "
            f"{res.get('reason','')}. Manual fix needed (perms on ~/.local/bin, restore "
            f"`~/.hermes/fleet/bin/hermes`, or build the runtime venv).")


def _check_dev_tree_lineage() -> None:
    """Dev-tree lineage guard (SPEC 2026-07-05 fork-clobber). The runtime-parity
    machinery watches the DEPLOY tree; nothing watched the ROOT DEV checkout's
    ``main`` lineage. A ``git reset --hard origin/main`` (bypassing land-on-main.sh,
    which is FF-only and can't stop a bypass) drifts dev-main onto raw upstream,
    dropping the fork's patches — the class that hung the 2026-07-05 backup.

    READ-ONLY sensor (``lib/dev_tree_lineage_check.py``): no fetch, no mutate, no
    auto-heal (realigning a contended dev checkout is a human judgement — it may carry
    in-flight local work). Routing:
      not drifted / behind-only     -> silent (behind is normal pending-land churn)
      drifted (clobber-onto-upstream)-> #logs advisory with the exact heal command
      drifted (unique local work)    -> #logs advisory (land/rebase onto fork/main)
    Never raises into the parity check — a sensor error must not take down the monitor.
    """
    checker = HERMES / "scripts" / "lib" / "dev_tree_lineage_check.py"
    if not checker.is_file():
        return
    try:
        p = subprocess.run([PY, str(checker)], capture_output=True, text=True, timeout=60)
        res = json.loads((p.stdout or "").strip() or "{}")
    except Exception as e:  # noqa: BLE001 — a sensor failure must never crash the parity monitor
        _notify(LOGS, "dev-tree-lineage: run error", f"could not run dev_tree_lineage_check: {e}")
        return
    if not res.get("drifted"):
        return  # on fork/main lineage (or just behind) — silent
    reason = res.get("reason", "")
    dev_head = res.get("dev_head", "?")
    fork_head = res.get("fork_head", "?")
    _notify(LOGS, "dev-tree off fork/main lineage (fork-clobber)",
            f"The root DEV checkout `~/.hermes/hermes-agent` main ({dev_head}) has drifted off "
            f"`fork/main` ({fork_head}):\n{reason}\n\nNot a backup outage (backups import from the "
            f"deploy tree), but the dev tree's own `hermes` runs the wrong code until realigned.")


def run() -> int:
    now = datetime.now(timezone.utc)
    _check_wrapper()  # pass-2 Req #3: FIRST statement, before every early return (wrapper drift ≠ parity)
    _check_gateway_shadows()  # D-6: all-gateways fossil-shadow guard (independent of fork/main parity)
    _check_stale_gateways()  # P3: stale-deployed-code advisory (reflog oracle, #logs)
    _redrive_pending_restarts()  # A1b: re-drive busy-skipped gateways once idle (ledger actuator)
    _check_dev_tree_lineage()  # 2026-07-05: dev-tree fork-clobber sensor (#logs advisory)
    kinds = _detector_kinds()

    ok, ref_or_reason = _fetch_verified()
    if not ok:
        if ref_or_reason.startswith("config-broken"):
            _notify(ALERTS, "🛑 runtime-parity: fetch config-broken", ref_or_reason)
            return 2
        # transient: quiet to #logs, no #alerts storm (INV-4)
        _notify(LOGS, "runtime-parity: transient fetch failure", ref_or_reason)
        DONE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        DONE_MARKER.write_text(now.isoformat())
        _self_heal_deadman()
        return 0

    fork_sha = ref_or_reason[:8]
    d = _detect()
    if d.get("_rc") == 2 or d.get("ok") is False:
        _notify(ALERTS, "🛑 runtime-parity: detector error",
                f"`runtime_tree_status.py` errored (rc={d.get('_rc')}): {d.get('_err','')}")
        return 2

    findings = d.get("findings", [])
    behind = next((f for f in findings if f.get("kind") == kinds["behind"]), None)
    ahead = [f for f in findings if f.get("kind") == kinds["ahead"]]
    dirty = [f for f in findings if f.get("kind") == kinds["dirty"]]

    st = _load_state()
    gaps = st.setdefault("gaps", {})

    # BEHIND → #alerts (the un-deploy outage), de-duped by fork_sha, age-re-escalated (INV-3/5/6)
    if behind:
        # non-FF branch: if HEAD is not an ancestor of the deploy ref, deploy.sh will refuse (exit 4)
        nonff = _git("merge-base", "--is-ancestor", "HEAD", DEPLOY_REF).returncode != 0
        prior = gaps.get(fork_sha)
        due = (prior is None or
               (now - datetime.fromisoformat(prior["last_alerted"])).total_seconds() >= AGE_REESCALATE_S)
        if due:
            if nonff:
                body = (f"⚠️ runtime tree has DIVERGED (non-FF) from {DEPLOY_REF} — `deploy.sh` will "
                        f"refuse (exit 4). Manual reconcile: inspect `git -C {TREE} log --oneline "
                        f"HEAD...{DEPLOY_REF}`, then `git reset --hard {DEPLOY_REF}` after confirming no "
                        f"uncommitted deploy hotfix.")
            else:
                body = (f"{behind['detail']}\nRun `bash ~/.hermes/fleet/deploy.sh` **then restart** "
                        f"(parity ≠ live process reloaded — the running gateway still holds old code "
                        f"until it restarts).")
            sent = _notify(ALERTS, "🛑 runtime tree behind fork/main (un-deployed merge)", body)
            if sent:  # INV-6: advance only on confirmed send
                gaps[fork_sha] = {"first_seen": (prior or {}).get("first_seen", now.isoformat()),
                                  "last_alerted": now.isoformat(), "kind": "behind"}
    # resolved gaps drop out so they re-alert if they ever return
    for sha in list(gaps):
        if not behind or sha != fork_sha:
            del gaps[sha]

    # DIRTY / AHEAD → #logs hygiene advisory (unchanged behavior, no downgrade)
    logs_lines = [f"  - {f['kind']}: {f['detail']}" for f in (dirty + ahead)]
    if logs_lines:
        _notify(LOGS, "Runtime-tree hygiene",
                "The runtime import tree has a non-outage finding (dev-in-place / ahead-of-deploy-ref):\n"
                + "\n".join(logs_lines))

    _save_state(st)
    DONE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    DONE_MARKER.write_text(now.isoformat())  # INV-7 liveness marker, LAST (before self-heal)
    _self_heal_deadman()
    return 0


def main() -> int:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lockf = open(LOCK, "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another parity check is running; exiting", file=sys.stderr)
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
