#!/usr/bin/env python3
"""WalletDB tick-watchdog notifier.

Runs walletdb ops tick-watchdog, compares against persisted state, and if needed
sends a concise Telegram DM via Clawdbot (delegated to the calling agent).

This script prints a single line of JSON to stdout with:
  {
    "ok": bool,
    "reasons": [..],
    "metrics": {...},
    "notify": "fail"|"recovery"|null,
    "message": "..."|null,
    "state_path": "...",
    "state_written": {...}
  }

No direct messaging is done here (to keep it usable in cron + agent wrapper).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

STATE_PATH_DEFAULT = "/home/yeqiuqiu/projects/walletdb/data/watchdog_notifier_state.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, timeout: int = 120) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _normalize_reasons(reasons: Any) -> List[str]:
    if reasons is None:
        return []
    if isinstance(reasons, list):
        return [str(x) for x in reasons]
    return [str(reasons)]


def _extract_metrics(payload: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ["last_tick_age_sec", "queue_nonterminal", "cursor_stalls", "warnings"]:
        if k in payload:
            out[k] = payload.get(k)
    # Some versions might nest metrics
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for k in ["last_tick_age_sec", "queue_nonterminal", "cursor_stalls", "warnings"]:
        if k not in out and k in metrics:
            out[k] = metrics.get(k)
    return out


def _journal_excerpt(text: str, *, max_lines: int = 12, max_chars: int = 1000) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    tail = lines[-max_lines:]
    joined = "\n".join(tail).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:].lstrip()


def build_message(prefix: str, reasons: List[str], metrics: Dict[str, Any], journal: Dict[str, str]) -> str:
    parts: List[str] = [prefix]

    if reasons:
        reason_lines = "\n".join(f"- {str(reason).strip()}" for reason in reasons if str(reason).strip())
        if reason_lines:
            parts.append("Why this fired:\n" + reason_lines)

    metric_lines: List[str] = []
    if "last_tick_age_sec" in metrics:
        metric_lines.append(f"- Last tick age: {metrics.get('last_tick_age_sec')}s")
    if "queue_nonterminal" in metrics:
        metric_lines.append(f"- Queue backlog (non-terminal): {metrics.get('queue_nonterminal')}")
    if "cursor_stalls" in metrics:
        metric_lines.append(f"- Cursor stalls: {metrics.get('cursor_stalls')}")
    if "warnings" in metrics and metrics.get("warnings"):
        w = metrics.get("warnings")
        if isinstance(w, list):
            warning_text = ", ".join(str(x) for x in w)
        else:
            warning_text = str(w)
        metric_lines.append(f"- Warnings: {warning_text}")
    if metric_lines:
        parts.append("Health snapshot:\n" + "\n".join(metric_lines))

    for unit, txt in journal.items():
        excerpt = _journal_excerpt(txt)
        if excerpt:
            parts.append(f"{unit} log tail (recent):\n{excerpt}")

    return "\n\n".join(parts).strip()


def main() -> None:
    repo = "/home/yeqiuqiu/projects/walletdb"
    db_path = "data/walletdb.sqlite"

    state_path = os.environ.get("WALLETDB_WATCHDOG_NOTIFIER_STATE", STATE_PATH_DEFAULT)

    env = os.environ.copy()
    env.setdefault("WALLETDB_TICK_WATCHDOG_CURSOR_STALL_TICKS", "30")

    cmd = [
        os.path.join(repo, ".venv", "bin", "python"),
        "-m",
        "walletdb.cli",
        "ops",
        "tick-watchdog",
        "--db-path",
        os.path.join(repo, db_path),
        "--json",
    ]

    rc, out, err = _run(cmd, cwd=repo, env=env, timeout=180)
    payload: Dict[str, Any] = {}
    if out.strip():
        try:
            payload = json.loads(out)
        except Exception:
            payload = {"ok": False, "reasons": ["tick-watchdog returned non-JSON"], "raw": out[:2000]}
    else:
        payload = {"ok": False, "reasons": ["tick-watchdog produced no output"], "stderr": err[:2000]}

    ok = bool(payload.get("ok")) if isinstance(payload.get("ok"), (bool, int)) else False
    reasons = _normalize_reasons(payload.get("reasons"))
    metrics = _extract_metrics(payload)

    prev = _read_json(state_path) or {}
    last_ok = prev.get("last_ok")
    last_reasons = _normalize_reasons(prev.get("last_reasons"))

    notify: Optional[str] = None
    prefix: Optional[str] = None

    if not ok:
        if last_ok is True and ok is False:
            notify = "fail"
        elif last_ok is False and ok is False:
            if reasons != last_reasons:
                notify = "fail"
        elif last_ok is None:
            # First run: only notify on failure if reasons exist
            if reasons:
                notify = "fail"
    else:
        if last_ok is False:
            notify = "recovery"

    journal: Dict[str, str] = {}
    if notify in ("fail", "recovery"):
        for unit in ["walletdb-tick.service", "walletdb-tick-watchdog.service"]:
            jrc, jout, _jerr = _run(["journalctl", "-u", unit, "-n", "40", "--no-pager"], timeout=15)
            if jrc == 0 and jout.strip():
                journal[unit] = jout

        if notify == "fail":
            prefix = "WalletDB watchdog FAIL"
        else:
            prefix = "WalletDB watchdog RECOVERY"

    msg = build_message(prefix, reasons, metrics, journal) if prefix else None

    new_state = {
        "last_ok": ok,
        "last_reasons": reasons,
        "last_ts": utc_now_iso(),
    }
    _write_json(state_path, new_state)

    result = {
        "ok": ok,
        "reasons": reasons,
        "metrics": metrics,
        "notify": notify,
        "message": msg,
        "state_path": state_path,
        "state_written": new_state,
        "tick_watchdog_rc": rc,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
