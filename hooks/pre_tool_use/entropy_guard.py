#!/usr/bin/env python3
"""
Hermes PreToolUse hook — entropy signal detection.

Reads a JSON payload from stdin (Claude Code PreToolUse hook format):
  {"tool_name": "...", "tool_input": {...}, "session_id": "..."}

Exit 0 = NOMINAL (no entropy signals detected or all signals suppressed for
Tier-1 read-only tools). Exit 2 = HIGH or CRITICAL entropy detected; a
structured JSON report is written to stderr.

Signals detected:
  unknown_confidence   (CRITICAL) — HERMES_CONFIDENCE env var is "UNKNOWN"
  tool_loop            (HIGH)     — same tool+input hash seen ≥ 2× this session
  stale_memory         (HIGH)     — tool_input contains "[STALE]" or "STALE:" marker
  conflicting_sources  (HIGH)     — HERMES_CONFLICTING_SOURCES env var set to "1"
  unverified_claim     (HIGH)     — HERMES_CONFIDENCE is "LOW" and tool is non-Tier-1

Tier-1 tools (read-only): web_search, fetch_url, read_file, recall_memory,
list_tasks. HIGH signals are suppressed for Tier-1 tools; CRITICAL signals
fire regardless of tier.

Stdlib only: hashlib, json, os, pathlib, sys. No LLM calls. <50ms target.

Configuration (all optional — safe defaults provided):
  HERMES_CONFIDENCE           Confidence level [HIGH, MEDIUM, LOW, UNKNOWN] (default: HIGH)
  HERMES_CONFLICTING_SOURCES  Set to "1" to assert conflicting-source signal (default: unset)
  HERMES_LEDGER_PATH          Path to tool-call ledger JSON file
                              (default: ~/.hermes/hooks/call_ledger.json)
  HERMES_MEMORY_DB            Path to memory SQLite DB for fact-confidence lookup
                              (default: ~/.hermes/memory_store.db)

Installation (Claude Code hook):
  Add to ~/.claude/settings.json under "hooks":
    "PreToolUse": [{"hooks": [{"type": "command",
      "command": "python3 ~/.hermes/hooks/pre_tool_use/entropy_guard.py"}]}]
"""
import hashlib
import json
import os
import pathlib
import sys

TIER1_TOOLS = frozenset(
    {"web_search", "fetch_url", "read_file", "recall_memory", "list_tasks"}
)

LEDGER_PATH = pathlib.Path(
    os.environ.get(
        "HERMES_LEDGER_PATH",
        str(pathlib.Path.home() / ".hermes" / "hooks" / "call_ledger.json"),
    )
)
MEMORY_DB = pathlib.Path(
    os.environ.get(
        "HERMES_MEMORY_DB",
        str(pathlib.Path.home() / ".hermes" / "memory_store.db"),
    )
)


def _load_ledger():
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _get_fact_confidence(fact_id: str) -> str:
    """Read confidence_tier for a fact_id from SQLite memory DB. Returns 'UNKNOWN' on any error."""
    try:
        import sqlite3

        conn = sqlite3.connect(str(MEMORY_DB), timeout=0.004)
        cur = conn.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        fact_table = next(
            (t for t in tables if "fact" in t.lower() or "memor" in t.lower()),
            None,
        )
        if fact_table is None:
            return "UNKNOWN"
        pragma = cur.execute(
            f"PRAGMA table_info({fact_table})"
        ).fetchall()
        pk_col = next(
            (r[1] for r in pragma if r[1] in ("fact_id", "id")),
            "fact_id",
        )
        row = cur.execute(
            f"SELECT confidence_tier FROM {fact_table}"
            f" WHERE {pk_col} = ?",
            (fact_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def detect_signals(tool_name, tool_input, session_id):
    signals = []
    confidence = os.environ.get("HERMES_CONFIDENCE", "HIGH").strip().upper()

    # unknown_confidence: CRITICAL
    if confidence == "UNKNOWN":
        signals.append(("unknown_confidence", "CRITICAL"))

    # Fact-based confidence from memory DB (for recall_memory tool)
    if tool_name == "recall_memory":
        fact_id = tool_input.get("fact_id", "")
        if fact_id:
            fact_tier = _get_fact_confidence(fact_id)
            if fact_tier == "UNKNOWN":
                signals.append(("unknown_confidence", "CRITICAL"))
            elif fact_tier in ("LOW",) and confidence != "UNKNOWN":
                signals.append(("unverified_claim", "HIGH"))

    # tool_loop: same tool+input hash seen ≥ 2× this session — HIGH
    ledger = _load_ledger()
    hash_key = hashlib.sha256(
        json.dumps(tool_input, sort_keys=True).encode()
    ).hexdigest()[:16]
    entry_key = f"{tool_name}:{hash_key}"
    session_entries = ledger.get(session_id, [])
    if session_entries.count(entry_key) >= 2:
        signals.append(("tool_loop", "HIGH"))

    # stale_memory: tool_input contains [STALE] or STALE: marker — HIGH
    tool_input_str = json.dumps(tool_input)
    if "[STALE]" in tool_input_str or "STALE:" in tool_input_str:
        signals.append(("stale_memory", "HIGH"))

    # conflicting_sources: env flag — HIGH
    if os.environ.get("HERMES_CONFLICTING_SOURCES", "").strip() == "1":
        signals.append(("conflicting_sources", "HIGH"))

    # unverified_claim: LOW confidence + non-Tier-1 tool — HIGH
    if confidence == "LOW" and tool_name not in TIER1_TOOLS:
        signals.append(("unverified_claim", "HIGH"))

    return signals


def _resolution(signal):
    resolutions = {
        "unknown_confidence": "Set HERMES_CONFIDENCE to HIGH or MEDIUM after sourcing the claim, or discard it.",
        "tool_loop": "The same query returned no new result. Replan: change the query or escalate to a different source.",
        "stale_memory": "Memory is marked stale. Refresh the fact from a current source before proceeding.",
        "conflicting_sources": "Sources conflict. Resolve the conflict before acting: pick the higher-authority source.",
        "unverified_claim": "Claim is unverified (LOW confidence). Find a second source or escalate research.",
    }
    return resolutions.get(signal, "Resolve entropy before proceeding.")


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    session_id = payload.get("session_id", "default")

    is_tier1 = tool_name in TIER1_TOOLS
    signals = detect_signals(tool_name, tool_input, session_id)

    if not signals:
        sys.exit(0)

    # CRITICAL fires regardless of tier; HIGH suppressed for Tier-1
    critical = [(s, l) for s, l in signals if l == "CRITICAL"]
    high = [(s, l) for s, l in signals if l == "HIGH"]

    if critical:
        signal_name, level = critical[0]
    elif high and not is_tier1:
        signal_name, level = high[0]
    else:
        sys.exit(0)

    report = {
        "signal": signal_name,
        "level": level,
        "recommended_resolution": _resolution(signal_name),
    }
    print(json.dumps(report), file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
