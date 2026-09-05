#!/usr/bin/env python3
"""Temporarily route the coder profile to Codex after repeated worker failures.

This is a deterministic engineering-board watchdog. It counts trailing closed
coder runs (including timeouts and provider rate limits), activates Codex Sol at
medium reasoning after the fourth consecutive failure, and restores the exact
profile settings captured at activation after one hour. Auto-blocked coder
cards are retried on each transition; PRODUCT_SIGNOFF cards are never touched.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROFILE = "coder"
BOARD = "engineering"
THRESHOLD = 4  # "more than 3"
WINDOW_SECONDS = 60 * 60
FALLBACK_MODEL = "gpt-5.6-sol"
FALLBACK_PROVIDER = "openai-codex"
FALLBACK_REASONING = "medium"
FAILURE_OUTCOMES = {
    "crashed",
    "timed_out",
    "spawn_failed",
    "gave_up",
    "failed",
    "rate_limited",
    "reclaimed",
}
DEFAULT_DB = Path.home() / ".hermes/kanban/boards/engineering/kanban.db"
DEFAULT_STATE = Path.home() / ".hermes/engineering/coder-fallback-state.json"
DEFAULT_CODER_HOME = Path.home() / ".hermes/profiles/coder"


def _run_hermes(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["hermes", *args], capture_output=True, text=True, check=check
    )


def _config_get(key: str, *, optional: bool = False) -> str | None:
    result = _run_hermes("-p", PROFILE, "config", "get", key, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    if optional:
        return None
    raise RuntimeError(
        f"cannot read coder config {key}: {result.stderr.strip() or result.stdout.strip()}"
    )


def _config_set(key: str, value: str) -> None:
    _run_hermes("-p", PROFILE, "config", "set", key, value)


def _config_restore(key: str, value: str | None) -> None:
    if value is None:
        _run_hermes("-p", PROFILE, "config", "unset", key)
    else:
        _config_set(key, value)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _trailing_failures(conn: sqlite3.Connection) -> tuple[int, int | None]:
    rows = conn.execute(
        "SELECT r.id, r.outcome FROM task_runs r "
        "JOIN tasks t ON t.id = r.task_id "
        "WHERE COALESCE(r.profile, t.assignee) = ? AND r.ended_at IS NOT NULL "
        "ORDER BY r.id DESC LIMIT 100",
        (PROFILE,),
    ).fetchall()
    count = 0
    latest_id = int(rows[0][0]) if rows else None
    for _, raw_outcome in rows:
        outcome = str(raw_outcome or "").lower()
        if outcome in FAILURE_OUTCOMES:
            count += 1
            continue
        break
    return count, latest_id


def _retry_auto_blocked(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT t.id, t.title, t.body FROM tasks t "
        "WHERE t.assignee = ? AND t.status = 'blocked' "
        "AND (SELECT e.kind FROM task_events e WHERE e.task_id = t.id "
        "     ORDER BY e.id DESC LIMIT 1) = 'gave_up'",
        (PROFILE,),
    ).fetchall()
    retried: list[str] = []
    for task_id, title, body in rows:
        text = f"{title or ''}\n{body or ''}".upper()
        if "PRODUCT_SIGNOFF" in text:
            continue
        _run_hermes("kanban", "--board", BOARD, "unblock", str(task_id))
        retried.append(str(task_id))
    return retried


def _apply_fallback() -> None:
    _config_set("model.default", FALLBACK_MODEL)
    _config_set("model.provider", FALLBACK_PROVIDER)
    _config_set("agent.reasoning_effort", FALLBACK_REASONING)


def _restore_primary(primary: dict[str, str | None]) -> None:
    _config_restore("model.default", primary["model"])
    _config_restore("model.provider", primary["provider"])
    _config_restore("agent.reasoning_effort", primary.get("reasoning_effort"))


def main() -> int:
    db_path = Path(os.environ.get("CODER_FALLBACK_DB", str(DEFAULT_DB)))
    state_path = Path(os.environ.get("CODER_FALLBACK_STATE", str(DEFAULT_STATE)))
    now = int(float(os.environ.get("CODER_FALLBACK_NOW", str(time.time()))))
    state = _read_state(state_path)

    conn = sqlite3.connect(db_path)
    try:
        failures, latest_run_id = _trailing_failures(conn)

        # A prior process may have died between recording the original profile
        # and committing fallback mode. Restore before evaluating the trigger
        # again so a partial write can never become the new "primary".
        if state.get("mode") == "activating" and state.get("primary"):
            _restore_primary(state["primary"])
            state = {
                "mode": "primary",
                "consecutive_failures": failures,
                "latest_run_id": latest_run_id,
                "recovered_activation_at": now,
            }
            _write_state(state_path, state)

        if state.get("mode") == "fallback":
            # Provider-declared reset windows outrank the legacy one-hour
            # watchdog timer.  Consult the same profile-scoped circuit used by
            # cron and Kanban; at reset this invocation atomically owns the one
            # bounded primary probe before restoring the route.
            primary = state.get("primary") or {}
            claimed_probe = None
            if primary.get("provider") and primary.get("model"):
                from agent.provider_health import ProviderHealthStore, ProviderRoute

                coder_home = Path(
                    os.environ.get("CODER_FALLBACK_PROFILE_HOME", str(DEFAULT_CODER_HOME))
                )
                store = ProviderHealthStore(coder_home)
                primary_route = ProviderRoute(str(primary["provider"]), str(primary["model"]))
                probe_owner = f"watchdog:coder-fallback:{os.getpid()}:{now}"
                decision = store.decide(
                    [primary_route],
                    owner=probe_owner,
                    now=datetime.fromtimestamp(now, tz=timezone.utc),
                )
                if decision.probe:
                    claimed_probe = (store, primary_route, probe_owner)
                if decision.route is None and decision.deferred_until is not None:
                    state["fallback_until"] = max(
                        int(state.get("fallback_until", 0)),
                        int(decision.deferred_until.timestamp()),
                    )
                    state["provider_deferred_until"] = int(
                        decision.deferred_until.timestamp()
                    )
                    state["consecutive_failures"] = failures
                    state["latest_run_id"] = latest_run_id
                    _write_state(state_path, state)
                    return 0
            if now < int(state.get("fallback_until", 0)):
                state["consecutive_failures"] = failures
                state["latest_run_id"] = latest_run_id
                _write_state(state_path, state)
                return 0
            _restore_primary(state["primary"])
            if claimed_probe is not None:
                # The watchdog only restores configuration; the first actual
                # Kanban task after reset must own the bounded model probe.
                probe_store, probe_route, probe_owner = claimed_probe
                probe_store.release_probe(probe_route, owner=probe_owner)
            retried = _retry_auto_blocked(conn)
            state.update({
                "mode": "primary",
                "reverted_at": now,
                "consecutive_failures": failures,
                "latest_run_id": latest_run_id,
            })
            _write_state(state_path, state)
            print(
                f"coder fallback expired; restored {state['primary']['provider']}/"
                f"{state['primary']['model']}; retried={','.join(retried) or 'none'}"
            )
            return 0

        last_trigger = state.get("last_trigger_run_id")
        if failures >= THRESHOLD and latest_run_id != last_trigger:
            primary = {
                "model": _config_get("model.default"),
                "provider": _config_get("model.provider"),
                "reasoning_effort": _config_get("agent.reasoning_effort", optional=True),
            }
            activating = {
                "mode": "activating",
                "primary": primary,
                "activated_at": now,
                "fallback_until": now + WINDOW_SECONDS,
                "last_trigger_run_id": latest_run_id,
                "latest_run_id": latest_run_id,
                "consecutive_failures": failures,
            }
            _write_state(state_path, activating)
            try:
                _apply_fallback()
            except Exception:
                _restore_primary(primary)
                activating["mode"] = "primary"
                activating["activation_failed_at"] = now
                _write_state(state_path, activating)
                raise
            activating["mode"] = "fallback"
            _write_state(state_path, activating)
            retried = _retry_auto_blocked(conn)
            print(
                f"coder failed {failures} consecutive runs; activated "
                f"{FALLBACK_PROVIDER}/{FALLBACK_MODEL} until "
                f"{activating['fallback_until']}; retried={','.join(retried) or 'none'}"
            )
            return 0

        state.update({
            "mode": "primary",
            "consecutive_failures": failures,
            "latest_run_id": latest_run_id,
        })
        _write_state(state_path, state)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
