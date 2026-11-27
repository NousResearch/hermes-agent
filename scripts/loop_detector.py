#!/usr/bin/env python3
"""
Loop Detector — real-time loop detection for Hermes sessions.

Reads the latest session from state.db, detects loop patterns:
- Same tool call with identical args (exact repeat)
- Same normalized approach (fingerprint repeat)
- Sliding window failure rate above threshold
- Repeated error patterns

Output: JSON alerts for cron/Telegram delivery.

Usage:
    python3 loop_detector.py                  # Check latest session
    python3 loop_detector.py --session ID     # Check specific session
    python3 loop_detector.py --days 1         # Check all sessions from last N days
    python3 loop_detector.py --json           # JSON output (for cron)
"""

import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
DB_PATH = HERMES_HOME / "state.db"

# --- Error classification (reused from agent-health) ---

_ERR_SIGS = [
    "error:", "traceback", "exception:", "failed to",
    "command not found", "permission denied", "no such file",
    "timed out", "timeout", "connection refused",
    "404 not found", "500 internal", "rate limit",
    "unauthorized", "access denied",
    "enoent", "eacces", "etimedout",
    "could not", "unable to", "syntax error",
    "blocked:", "exit code",
]
_ERR_RE = re.compile("|".join(re.escape(s) for s in _ERR_SIGS), re.IGNORECASE)

# Transient errors that should NOT count as failures
_TRANSIENT_RE = re.compile(
    r"(?i)(rate.?limit|429|throttl|connection.?reset|network.?unreachable|"
    r"ECONNRESET|EPIPE|temporary.?fail)", re.IGNORECASE
)


def is_error(content: str) -> bool:
    """Classify tool result as error."""
    if not content:
        return False
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(content[start:end + 1])
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if obj.get("error"):
                return True
            if obj.get("success") is True:
                return False
            if "exit_code" in obj:
                return obj.get("exit_code") not in (None, 0)
            if "status" in obj:
                status = str(obj.get("status")).lower()
                if status == "timeout" and obj.get("process_running"):
                    return False
                return status in ("failed", "error", "timeout")
    return bool(_ERR_RE.search(content[:2000]))


# Shell init errors — environment issues, not agent reasoning failures
_SHELL_INIT_RE = re.compile(
    r"(?i)(cd:.*no such file|bash.*line \d+:|\.bashrc|\.profile|"
    r"pihole|/home/nd/\.\w+)", re.IGNORECASE
)


def is_shell_init_error(content: str) -> bool:
    """Detect errors from shell initialization, not agent actions."""
    if not content:
        return False
    return bool(_SHELL_INIT_RE.search(content[:500]))


def is_transient(content: str) -> bool:
    """Check if error is transient (should not count toward failure threshold)."""
    if not content:
        return False
    return bool(_TRANSIENT_RE.search(content[:1000]))


# --- Approach fingerprinting ---

# Commands that are semantically equivalent
_EQUIV_MAP = {
    "docker compose": "docker-compose",
    "docker-compose": "docker compose",
    "systemctl restart": "service restart",
    "service restart": "systemctl restart",
}


def normalize_command(cmd: str) -> str:
    """Normalize a command for fingerprinting."""
    c = cmd.strip()
    # Remove common prefixes
    for prefix in ["cd /tmp && ", "cd ~ && ", "sudo "]:
        if c.startswith(prefix):
            c = c[len(prefix):]
    # Normalize whitespace
    c = " ".join(c.split())
    # Normalize equivalent commands
    for old, new in _EQUIV_MAP.items():
        c = c.replace(old, new)
    return c


def fingerprint(tool_name: str, args: dict) -> str:
    """Create a normalized fingerprint for a tool call."""
    if tool_name == "terminal":
        cmd = args.get("command", "")
        return f"terminal:{normalize_command(cmd)}"
    elif tool_name == "web_search":
        q = args.get("query", "")
        return f"web_search:{q.lower().strip()}"
    elif tool_name == "web_extract":
        urls = args.get("urls", [])
        return f"web_extract:{len(urls)}_urls"
    elif tool_name == "read_file":
        path = args.get("path", "")
        return f"read_file:{path}"
    elif tool_name == "write_file":
        path = args.get("path", "")
        return f"write_file:{path}"
    elif tool_name == "patch":
        path = args.get("path", "")
        return f"patch:{path}"
    elif tool_name == "terminal":
        return f"terminal:{args.get('command', '')[:50]}"
    else:
        # Generic: tool name + sorted args hash
        arg_str = json.dumps(args, sort_keys=True, default=str)[:100]
        return f"{tool_name}:{arg_str}"


def extract_args_from_tool_calls(tool_calls_json: str) -> dict:
    """Extract arguments from assistant tool_calls JSON."""
    if not tool_calls_json:
        return {}
    try:
        calls = json.loads(tool_calls_json)
        if isinstance(calls, list) and calls:
            fn = calls[0].get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                return json.loads(args)
            return args
    except Exception:
        pass
    return {}


# --- Sliding window analysis ---

def analyze_session(messages: list[dict], window: int = 10) -> dict:
    """Analyze a session's messages for loop patterns.
    
    Args:
        messages: list of message dicts (sorted by timestamp)
        window: sliding window size for failure rate calculation
    """
    alerts = []
    
    # Build tool call sequences
    tool_calls = []  # (timestamp, tool_name, fingerprint, is_error, content_preview, is_shell_init)
    
    # Track user messages for intervention detection
    user_message_timestamps = [m.get("timestamp", 0) for m in messages if m.get("role") == "user"]
    
    # Match assistant tool_calls with tool results
    # Key: call_id from tool_calls JSON -> fingerprint
    pending_calls = {}
    
    for m in messages:
        role = m.get("role", "")
        
        if role == "assistant" and m.get("tool_calls"):
            tc_str = m.get("tool_calls", "")
            try:
                tc_list = json.loads(tc_str)
                if isinstance(tc_list, list):
                    for tc_item in tc_list:
                        call_id = tc_item.get("call_id") or tc_item.get("id", "")
                        fn = tc_item.get("function", {})
                        tool_name = fn.get("name", "")
                        args_raw = fn.get("arguments", "{}")
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        fp = fingerprint(tool_name, args)
                        if call_id:
                            pending_calls[call_id] = (tool_name, fp)
            except Exception:
                pass
        
        elif role == "tool" and m.get("tool_name"):
            tool_name = m["tool_name"]
            content = m.get("content", "") or ""
            err = is_error(content)
            transient = is_transient(content) if err else False
            shell_init = is_shell_init_error(content) if err else False
            # Fallback: use tool_name when call_id matching fails
            tcid = m.get("tool_call_id", "")
            fp_tuple = pending_calls.pop(tcid, None)
            if fp_tuple:
                fp = fp_tuple[1]
            else:
                # No matching assistant call — use tool name as coarse fingerprint
                fp = f"{tool_name}:?<unknown_args>"
            
            tool_calls.append({
                "timestamp": m.get("timestamp", 0),
                "tool": tool_name,
                "fingerprint": fp,
                "is_error": err,
                "is_transient": transient,
                "is_shell_init": shell_init,
                "content_preview": content[:150].replace("\n", " "),
            })
    
    if not tool_calls:
        return {"status": "no_tool_calls", "alerts": []}
    
    # --- Detection 1: Exact fingerprint repeat (≥3 times) ---
    # Exclude shell init errors and check if approach changed after failure
    fp_counts = Counter()
    fp_first_seen = {}
    fp_last_error = {}
    fp_had_success_after = set()  # fingerprints that had a success after failure
    prev_fp = None
    prev_was_error = False
    
    for tc in tool_calls:
        fp = tc["fingerprint"]
        
        # Track if approach changed after error (workaround detection)
        if prev_was_error and not tc["is_error"] and fp != prev_fp:
            fp_had_success_after.add(prev_fp)
        
        fp_counts[fp] += 1
        if fp not in fp_first_seen:
            fp_first_seen[fp] = tc["timestamp"]
        if tc["is_error"] and not tc["is_shell_init"]:
            fp_last_error[fp] = tc["content_preview"]
        
        prev_fp = fp
        prev_was_error = tc["is_error"]
    
    for fp, count in fp_counts.items():
        # Skip if: approach had success after error (workaround) OR is shell init
        if fp in fp_had_success_after:
            continue
        # Count only non-shell-init errors for this fingerprint
        real_errors = sum(1 for tc in tool_calls if tc["fingerprint"] == fp 
                         and tc["is_error"] and not tc["is_shell_init"])
        if real_errors < 2:  # Need at least 2 real errors to flag
            continue
        if count >= 3:
            alerts.append({
                "type": "exact_repeat",
                "severity": "high" if count >= 5 else "medium",
                "fingerprint": fp,
                "count": count,
                "real_errors": real_errors,
                "error_sample": fp_last_error.get(fp, ""),
                "message": f"Same approach repeated {count}x ({real_errors} errors): {fp[:80]}",
            })
    
    # --- Detection 2: Sliding window failure rate (excluding shell init) ---
    real_errors = [tc for tc in tool_calls if not tc["is_transient"] and not tc["is_shell_init"]]
    if len(real_errors) >= window:
        recent = real_errors[-window:]
        err_count = sum(1 for tc in recent if tc["is_error"])
        err_rate = err_count / window
        if err_rate >= 0.6:
            alerts.append({
                "type": "high_failure_rate",
                "severity": "high",
                "window": window,
                "errors": err_count,
                "total": window,
                "rate": round(err_rate * 100, 1),
                "message": f"High failure rate: {err_count}/{window} ({err_rate*100:.0f}%) in last {window} calls",
            })
    
    # --- Detection 3: Same error message repeated (excluding shell init) ---
    err_msgs = Counter()
    for tc in tool_calls:
        if tc["is_error"] and not tc["is_transient"] and not tc["is_shell_init"]:
            msg = tc["content_preview"].lower()[:100]
            err_msgs[msg] += 1
    for msg, count in err_msgs.items():
        if count >= 3:
            alerts.append({
                "type": "repeated_error",
                "severity": "high",
                "error_message": msg[:100],
                "count": count,
                "message": f"Same error repeated {count}x: {msg[:80]}",
            })
    
    # --- Detection 4: Alternating pattern (A-B-A-B) ---
    # With false-positive filters: different errors, time gaps, or progress
    if len(tool_calls) >= 4:
        recent = tool_calls[-6:]
        fps = [tc["fingerprint"] for tc in recent]
        if len(fps) >= 4:
            for i in range(len(fps) - 3):
                if fps[i] == fps[i+2] and fps[i+1] == fps[i+3] and fps[i] != fps[i+1]:
                    # Filter 1: Different error messages → likely debugging, not loop
                    err_a = next((tc for tc in recent if tc["fingerprint"] == fps[i] and tc["is_error"]), None)
                    err_b = next((tc for tc in recent if tc["fingerprint"] == fps[i+1] and tc["is_error"]), None)
                    if err_a and err_b:
                        err_a_msg = err_a["content_preview"][:80].lower()
                        err_b_msg = err_b["content_preview"][:80].lower()
                        if err_a_msg != err_b_msg:
                            continue  # Different errors = debugging, not loop
                    
                    # Filter 2: Success between errors → task progress
                    has_success = any(not tc["is_error"] for tc in recent[i:i+4])
                    if has_success:
                        continue
                    
                    # Filter 3: Time gap > 60s between A and B → deliberate retry
                    ts_a = [tc["timestamp"] for tc in recent if tc["fingerprint"] == fps[i]]
                    ts_b = [tc["timestamp"] for tc in recent if tc["fingerprint"] == fps[i+1]]
                    if ts_a and ts_b:
                        time_gap = min(abs(t2 - t1) for t1 in ts_a for t2 in ts_b)
                        if time_gap > 60:
                            continue  # Significant time gap = deliberate debugging
                    
                    alerts.append({
                        "type": "alternating_loop",
                        "severity": "high",
                        "pattern": f"{fps[i][:40]} → {fps[i+1][:40]}",
                        "message": f"Alternating loop detected: {fps[i][:40]} ↔ {fps[i+1][:40]}",
                    })
                    break
    
    # --- Detection 5: Escalation needed (consecutive non-transient, non-shell-init errors) ---
    consecutive_errors = 0
    for tc in reversed(real_errors):
        if tc["is_error"]:
            consecutive_errors += 1
        else:
            break
    if consecutive_errors >= 3:
        alerts.append({
            "type": "consecutive_errors",
            "severity": "critical",
            "count": consecutive_errors,
            "last_error": real_errors[-1]["content_preview"] if real_errors else "",
            "message": f"{consecutive_errors} consecutive errors — consider model escalation",
        })
    
    # --- Summary stats ---
    total_calls = len(tool_calls)
    total_errors = sum(1 for tc in tool_calls if tc["is_error"])
    transient_errors = sum(1 for tc in tool_calls if tc["is_transient"])
    unique_fingerprints = len(fp_counts)
    
    return {
        "status": "alerts" if alerts else "ok",
        "total_calls": total_calls,
        "total_errors": total_errors,
        "transient_errors": transient_errors,
        "permanent_errors": total_errors - transient_errors,
        "unique_approaches": unique_fingerprints,
        "error_rate": round(total_errors / max(total_calls, 1) * 100, 1),
        "alerts": alerts,
    }


def get_session_messages(conn, session_id: str = None, days: int = None) -> list[dict]:
    """Get messages from a session."""
    if session_id:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    elif days:
        cutoff = time.time() - days * 86400
        rows = conn.execute(
            """SELECT m.* FROM messages m 
               JOIN sessions s ON m.session_id = s.id 
               WHERE s.started_at > ? 
               ORDER BY m.timestamp""",
            (cutoff,),
        ).fetchall()
    else:
        # Latest session
        rows = conn.execute(
            """SELECT m.* FROM messages m 
               JOIN sessions s ON m.session_id = s.id 
               WHERE s.archived = 0 
               ORDER BY m.timestamp DESC LIMIT 200""",
        ).fetchall()
        rows = list(reversed(rows))
    
    return [dict(r) for r in rows]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Loop Detector")
    p.add_argument("--session", help="Specific session ID")
    p.add_argument("--days", type=int, help="Check sessions from last N days")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--window", type=int, default=10, help="Sliding window size")
    args = p.parse_args()
    
    if not DB_PATH.exists():
        print(json.dumps({"error": f"Database not found: {DB_PATH}"}))
        sys.exit(1)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    messages = get_session_messages(conn, args.session, args.days)
    conn.close()
    
    if not messages:
        result = {"status": "no_data", "message": "No messages found"}
        if args.json:
            print(json.dumps(result))
        else:
            print("No messages found.")
        return
    
    result = analyze_session(messages, window=args.window)
    
    # Add session context
    if messages:
        result["session_id"] = messages[0].get("session_id", "unknown")
        result["message_count"] = len(messages)
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        # Human-readable output
        if result["status"] == "ok":
            print(f"✅ No loops detected. {result['total_calls']} calls, "
                  f"{result['error_rate']}% error rate, "
                  f"{result['unique_approaches']} unique approaches.")
        else:
            print(f"⚠️ {len(result['alerts'])} alert(s) detected!\n")
            for a in result["alerts"]:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(a["severity"], "⚪")
                print(f"{severity_icon} [{a['type']}] {a['message']}")
            print(f"\n📊 {result['total_calls']} calls, {result['error_rate']}% error rate, "
                  f"{result['unique_approaches']} unique approaches")


if __name__ == "__main__":
    main()
