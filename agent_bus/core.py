"""Agent-to-agent task bus core API.

Both Hermes (native tool) and OpenClaw (via CLI) call these functions.
Every state change posts to Slack #ops-evolution and logs an event.
"""

import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_bus import finalizer, storage

logger = logging.getLogger(__name__)

VALID_AGENTS = {"hermes", "openclaw"}
VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}
VALID_TERMINAL_STATUSES = {"done", "fail", "timeout"}


def _run_close_gate(
    *,
    task_id: str,
    outcome: str,
    summary: str,
    current_status: str,
) -> str:
    """Run finalizer Gate A + Gate B. Returns ``"proceed"`` or ``"idempotent"``.

    Raises ``ValueError`` (with the finalizer error code as the message) when
    the gate rejects and the mode is enforcing. In ``off`` or ``advisory`` mode,
    rejections are logged but treated as ``"proceed"`` so legacy behaviour is
    preserved until full rollout (see `finalizer-gate-spec-v0.md` §11 Rollout).
    """
    decision, code = finalizer.enforce_close(
        {"task_id": task_id, "outcome": outcome, "summary": summary},
        current_status=current_status,
        logger=logger,
    )
    if decision == "idempotent":
        return "idempotent"
    if decision == "proceed":
        return "proceed"
    # decision == "reject"
    if finalizer.is_enforcing():
        raise ValueError(code or "FINALIZER_REJECTED")
    logger.warning(
        "finalizer[%s]: would reject close task=%s code=%s — proceeding in legacy mode",
        finalizer.get_mode(), task_id, code,
    )
    return "proceed"

SLACK_CHANNEL_OPS_EVOLUTION = "C0AT8C6GJTH"
SLACK_CHANNEL_HERMES_INBOX = "C0ATLDC284D"

# Minimum seconds between consecutive push-nudges to the same agent for the
# same task, so we don't spam on retries / rapid state changes.
_NOTIFY_COOLDOWN_SECONDS = 60.0
_last_notify_at: Dict[str, float] = {}  # key: f"{agent}:{task_id}"


def _now() -> float:
    return time.time()


def _gen_task_id() -> str:
    return f"T-{uuid.uuid4().hex[:6].upper()}"


def _validate_agent(agent: str, field: str = "agent") -> None:
    if agent not in VALID_AGENTS:
        raise ValueError(f"Invalid {field}: {agent!r} (must be one of {sorted(VALID_AGENTS)})")


def _task_view(task_id: str, include_events: bool = True) -> Dict[str, Any]:
    task = storage.get_task_row(task_id)
    if not task:
        return None
    if include_events:
        task["events"] = storage.get_task_events(task_id)
    return task


def _extract_keywords(goal: str, max_words: int = 4) -> List[str]:
    """Pull a few meaningful words from a goal for prior-learning lookup.
    Skip short/stop words, keep alnum words >3 chars, keep CJK chunks
    that are ≥2 chars (single Han chars match too much junk).
    Good enough for `rg -l <word>` searches.
    """
    tokens = re.split(r"[\s,.;:!?！？。，、：；（）()\"'`\[\]]+", goal.strip())
    stop = {
        "the", "and", "for", "with", "from", "into", "這", "那", "用", "把",
        "請", "和", "或", "以", "一個", "任務", "的", "嗎", "沒有", "可以",
        "需要", "一下", "試一下", "看看",
    }
    picked: List[str] = []
    for t in tokens:
        t = t.strip()
        if not t or t.lower() in stop:
            continue
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", t))
        # CJK chunks need ≥2 chars to avoid noisy matches; ASCII needs >3 chars
        # (but keep paths like /etc/hostname that contain punctuation).
        if has_cjk and len(t) < 2:
            continue
        if not has_cjk and len(t) <= 3 and "/" not in t:
            continue
        picked.append(t[:40])
        if len(picked) >= max_words:
            break
    return picked


def _gather_prior_learnings(goal: str, *, max_hits: int = 3) -> str:
    """Look up wiki learnings relevant to the goal text. Returns a compact
    markdown block for attaching to the task context, or "" if nothing found.
    Fails soft — any error returns empty string.
    """
    try:
        hits: List[Dict[str, Any]] = []
        seen: set = set()
        for kw in _extract_keywords(goal):
            for h in wiki_query(kw, limit=5):
                p = h.get("path")
                if not p or p in seen:
                    continue
                seen.add(p)
                hits.append(h)
                if len(hits) >= max_hits * 2:
                    break
            if len(hits) >= max_hits * 2:
                break
        if not hits:
            return ""
        # Prefer agent-bus learnings + recent memory files
        def _score(h):
            p = h.get("path", "")
            s = 0
            if "agent-bus" in p:
                s += 3
            if "/memory/" in p or p.startswith("memory/"):
                s += 2
            return s
        hits.sort(key=_score, reverse=True)
        top = hits[:max_hits]
        lines = ["### Prior learnings (auto-retrieved)", ""]
        for h in top:
            snippet = (h.get("snippet") or "").strip().replace("\n", " ")[:200]
            lines.append(f"- `{h['path']}` — {h.get('title', '')}")
            if snippet:
                lines.append(f"  > {snippet}")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("agent_bus: prior-learning lookup failed: %s", exc)
        return ""


def assign_task(
    *,
    from_agent: str,
    to_agent: str,
    goal: str,
    success_criteria: Optional[str] = None,
    context: Optional[str] = None,
    priority: str = "P2",
    deadline_minutes: Optional[int] = None,
    parent_task_id: Optional[str] = None,
    skip_prior_learnings: bool = False,
) -> Dict[str, Any]:
    """Assign a new task. Returns the full task record."""
    _validate_agent(from_agent, "from_agent")
    _validate_agent(to_agent, "to_agent")
    if from_agent == to_agent:
        raise ValueError("from_agent and to_agent must differ")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"priority must be one of {sorted(VALID_PRIORITIES)}")
    if not goal or not goal.strip():
        raise ValueError("goal is required")

    task_id = _gen_task_id()
    now = _now()
    deadline = now + deadline_minutes * 60 if deadline_minutes else None

    # L2 self-evolution: auto-retrieve relevant prior learnings from the wiki
    # and attach them to the task context so the recipient starts warm
    # instead of rediscovering the same pitfalls. Can be suppressed with
    # skip_prior_learnings=True for ping/test tasks.
    if not skip_prior_learnings:
        prior = _gather_prior_learnings(goal)
        if prior:
            if context and context.strip():
                context = f"{context.rstrip()}\n\n{prior}"
            else:
                context = prior

    task = {
        "task_id": task_id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "goal": goal.strip(),
        "success_criteria": success_criteria,
        "context": context,
        "priority": priority,
        "status": "pending",
        "created_at": now,
        "deadline": deadline,
        "parent_task_id": parent_task_id,
    }
    storage.insert_task(task)
    storage.add_event(task_id, from_agent, "created", {
        "goal": goal, "success_criteria": success_criteria,
        "deadline_minutes": deadline_minutes, "priority": priority,
        "prior_learnings_attached": bool(context and "Prior learnings" in (context or "")),
    })

    # Post to Slack and attach thread_ts for follow-up replies
    thread_ts, channel = _slack_post_assignment(task)
    if thread_ts:
        storage.update_task(task_id, slack_thread_ts=thread_ts, slack_channel=channel)
        task["slack_thread_ts"] = thread_ts
        task["slack_channel"] = channel

    # Push-notify the recipient so they pick it up without polling.
    _notify_agent(to_agent, task, kind="new")
    return _task_view(task_id)


def ack_task(*, task_id: str, agent: str, note: Optional[str] = None) -> Dict[str, Any]:
    """Recipient acknowledges the task."""
    _validate_agent(agent, "agent")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can ack; caller was {agent}"
        )
    if task["status"] in VALID_TERMINAL_STATUSES:
        raise ValueError(f"Task {task_id} is already terminal ({task['status']})")

    storage.update_task(task_id, status="ack", acked_at=_now())
    storage.add_event(task_id, agent, "ack", {"note": note})
    _slack_reply(task, f"🟡 `{task_id}` ACK by *{agent}*" + (f"\n> {note}" if note else ""))
    return _task_view(task_id)


def progress_task(*, task_id: str, agent: str, note: str) -> Dict[str, Any]:
    _validate_agent(agent, "agent")
    if not note or not note.strip():
        raise ValueError("note is required for progress updates")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can report progress; caller was {agent}"
        )
    if task["status"] in VALID_TERMINAL_STATUSES:
        raise ValueError(f"Task {task_id} is already terminal ({task['status']})")

    # Move to progress if currently pending/ack
    if task["status"] in ("pending", "ack"):
        storage.update_task(task_id, status="progress", acked_at=task["acked_at"] or _now())
    storage.add_event(task_id, agent, "progress", {"note": note})
    _slack_reply(task, f"🔵 `{task_id}` progress\n> {note.strip()}")
    return _task_view(task_id)


def complete_task(
    *,
    task_id: str,
    agent: str,
    result: str,
    learning: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_agent(agent, "agent")
    if not result or not result.strip():
        raise ValueError("result is required")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can complete; caller was {agent}"
        )

    decision = _run_close_gate(
        task_id=task_id, outcome="done", summary=result, current_status=task["status"],
    )
    if decision == "idempotent":
        logger.info("complete_task: idempotent re-close for %s", task_id)
        return _task_view(task_id)

    storage.update_task(task_id, status="done", result=result, completed_at=_now())
    storage.add_event(task_id, agent, "done", {"result": result, "learning": learning})
    # Refresh task now that result/completed_at/status are persisted so the
    # broadcast below carries the authoritative row.
    task = storage.get_task_row(task_id) or task
    broadcast_ok = _slack_reply(
        task,
        f"✅ `{task_id}` DONE by *{agent}*\n> {result.strip()[:500]}"
        + (f"\n\n_learning_: {learning.strip()[:500]}" if learning else "")
    )
    if broadcast_ok:
        storage.update_task(task_id, terminal_broadcast_ok=1)
    learning_path = None
    if learning:
        learning_path = _write_learning(task_id, agent, learning, outcome="done")
        if learning_path:
            storage.update_task(task_id, learning_wiki_path=str(learning_path))
    # Wake the originator so they see the result without polling.
    # Exception: when Hermes is the originator, the user-facing ping below
    # already covers it — don't spawn a second LLM turn for the same fact.
    updated = _task_view(task_id)
    if task["from_agent"] != "hermes":
        _notify_agent(
            task["from_agent"], updated, kind="update",
            extra=f"任務已完成。結果：{result.strip()[:300]}",
        )
    # Proactively tell the human boss when Hermes was the dispatcher.
    if _notify_user_of_outcome(updated):
        storage.update_task(task_id, user_notified=1)
    return updated


def fail_task(
    *,
    task_id: str,
    agent: str,
    reason: str,
    learning: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_agent(agent, "agent")
    if not reason or not reason.strip():
        raise ValueError("reason is required")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can fail; caller was {agent}"
        )

    decision = _run_close_gate(
        task_id=task_id, outcome="fail", summary=reason, current_status=task["status"],
    )
    if decision == "idempotent":
        logger.info("fail_task: idempotent re-close for %s", task_id)
        return _task_view(task_id)

    storage.update_task(task_id, status="fail", result=reason, completed_at=_now())
    storage.add_event(task_id, agent, "fail", {"reason": reason, "learning": learning})
    task = storage.get_task_row(task_id) or task
    broadcast_ok = _slack_reply(
        task,
        f"❌ `{task_id}` FAILED by *{agent}*\n> {reason.strip()[:500]}"
        + (f"\n\n_learning_: {learning.strip()[:500]}" if learning else "")
    )
    if broadcast_ok:
        storage.update_task(task_id, terminal_broadcast_ok=1)
    if learning:
        learning_path = _write_learning(task_id, agent, learning, outcome="fail")
        if learning_path:
            storage.update_task(task_id, learning_wiki_path=str(learning_path))
    updated = _task_view(task_id)
    if task["from_agent"] != "hermes":
        _notify_agent(
            task["from_agent"], updated, kind="update",
            extra=f"任務失敗。原因：{reason.strip()[:300]}",
        )
    if _notify_user_of_outcome(updated):
        storage.update_task(task_id, user_notified=1)
    return updated


def keep_alive_task(
    *,
    task_id: str,
    agent: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark a long-running task as still alive and extend its deadline.

    Not a terminal state — pairs with `finalizer-gate-spec-v0.md` §4.
    Valid from ack / progress / keep-alive. Pending tasks must ack first.
    Terminal tasks cannot go back to keep-alive.

    Extends ``deadline`` by ``FINALIZER_KEEPALIVE_TIMEOUT_SEC`` (default 1800s).
    Does not broadcast to Slack by default (noise reduction).
    """
    _validate_agent(agent, "agent")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can keep-alive; caller was {agent}"
        )

    ok, code = finalizer.validate_transition(task["status"], finalizer.OUTCOME_KEEP_ALIVE)
    if not ok:
        if finalizer.is_enforcing():
            raise ValueError(code or "FINALIZER_REJECTED")
        logger.warning(
            "finalizer[%s]: would reject keep_alive task=%s code=%s — proceeding in legacy mode",
            finalizer.get_mode(), task_id, code,
        )

    extend = finalizer.keepalive_timeout_sec()
    new_deadline = _now() + extend
    storage.update_task(task_id, status="keep-alive", deadline=new_deadline)
    storage.add_event(
        task_id, agent, "keep_alive",
        {"note": note, "extended_sec": extend},
    )
    return _task_view(task_id)


def amend_learning(
    *,
    task_id: str,
    agent: str,
    learning: str,
) -> Dict[str, Any]:
    """Attach or replace the learning file for an already-terminal task.

    Decouples the close path from learning persistence — if the original
    close swallowed a learning-write error, the recipient can retry later
    without re-closing the task.
    """
    _validate_agent(agent, "agent")
    if not learning or not learning.strip():
        raise ValueError("learning is required")
    task = storage.get_task_row(task_id)
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    if task["to_agent"] != agent:
        raise ValueError(
            f"Only the recipient ({task['to_agent']}) can amend learning; caller was {agent}"
        )
    if task["status"] not in VALID_TERMINAL_STATUSES:
        raise ValueError(
            f"amend_learning requires a terminal task (status={task['status']!r})"
        )

    outcome = task["status"] if task["status"] in ("done", "fail") else "done"
    learning_path = _write_learning(task_id, agent, learning, outcome=outcome)
    if learning_path:
        storage.update_task(task_id, learning_wiki_path=str(learning_path))
    storage.add_event(task_id, agent, "amend_learning", {"path": str(learning_path) if learning_path else None})
    return _task_view(task_id)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_view(task_id)


def get_outstanding(agent: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Tasks this agent still needs to act on (assigned TO them, not terminal)."""
    _validate_agent(agent, "agent")
    return storage.query_tasks(
        to_agent=agent,
        status_in=["pending", "ack", "progress"],
        limit=limit,
    )


def get_sent(agent: str, limit: int = 50, include_terminal: bool = False) -> List[Dict[str, Any]]:
    """Tasks this agent has dispatched to the other."""
    _validate_agent(agent, "agent")
    status_in = None if include_terminal else ["pending", "ack", "progress"]
    return storage.query_tasks(from_agent=agent, status_in=status_in, limit=limit)


def list_recent(limit: int = 20) -> List[Dict[str, Any]]:
    return storage.query_tasks(limit=limit)


def wiki_query(query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Grep `~/wiki/` for past learnings/notes matching `query`.

    Returns a list of `{path, title, snippet}` dicts. Intended for the
    Hermes tool so Hermes can pull prior learnings before dispatching a
    similar task — "what have we done like this before?"
    """
    dir_ = _wiki_memory_dir()
    if dir_ is None:
        return []
    vault = dir_.parent
    if not vault.exists():
        return []
    # Prefer ripgrep if available — much faster on large vaults.
    rg = shutil.which("rg")
    results: List[Dict[str, Any]] = []
    seen: set = set()
    if rg:
        try:
            out = subprocess.run(
                [rg, "--no-messages", "-i", "-l", "--glob", "*.md", query, str(vault)],
                capture_output=True, text=True, timeout=15,
            )
            paths = [Path(p) for p in out.stdout.splitlines() if p.strip()]
        except Exception as exc:
            logger.warning("agent_bus: wiki rg failed: %s", exc)
            paths = []
    else:
        # Fallback: simple recursive walk
        paths = list(vault.rglob("*.md"))
        q_lower = query.lower()
        paths = [p for p in paths if q_lower in p.read_text(encoding="utf-8", errors="ignore").lower()]

    for p in paths[:limit]:
        if p in seen:
            continue
        seen.add(p)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # Title from frontmatter or filename
        title = p.stem
        if text.startswith("---"):
            m = re.search(r"^title:\s*(.+)$", text[:400], re.MULTILINE)
            if m:
                title = m.group(1).strip()
        # Snippet: first line containing the query, plus neighbors
        q_lower = query.lower()
        lines = text.splitlines()
        snippet = ""
        for i, line in enumerate(lines):
            if q_lower in line.lower():
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                snippet = "\n".join(lines[start:end]).strip()
                break
        if not snippet:
            # Fall back to first non-frontmatter paragraph
            body = text
            if text.startswith("---"):
                body = text.split("---", 2)[-1]
            snippet = body.strip().split("\n\n", 1)[0][:400]
        results.append({
            "path": str(p.relative_to(vault)),
            "title": title,
            "snippet": snippet[:600],
        })
    return results


def check_timeouts() -> List[Dict[str, Any]]:
    """Mark past-deadline tasks as timed out, alert in Slack. Returns the list."""
    timed_out = storage.query_timed_out(_now())
    for task in timed_out:
        storage.update_task(
            task["task_id"], status="timeout", completed_at=_now()
        )
        storage.add_event(
            task["task_id"], "system", "timeout",
            {"deadline": task["deadline"]},
        )
        _slack_reply(
            task,
            f"⏰ `{task['task_id']}` TIMEOUT — *{task['to_agent']}* did not complete by deadline"
        )
        # Also wake the originator agent so it knows — unless Hermes was the
        # originator, in which case the user-facing ping below is enough.
        if task["from_agent"] != "hermes":
            _notify_agent(
                task["from_agent"], task, kind="update",
                extra=f"⚠️ 任務超時，{task['to_agent']} 未在 deadline 前完成。",
            )
        # Ping the user directly when Hermes was the dispatcher.
        refreshed = storage.get_task_row(task["task_id"]) or task
        if _notify_user_of_outcome(refreshed):
            storage.update_task(task["task_id"], user_notified=1)
    return timed_out


def ensure_side_effects() -> Dict[str, int]:
    """Rebroadcast pending side-effects (Slack posts, wiki writes) that
    failed because the caller's sandbox blocked them.

    Called by the Hermes gateway watchdog. Hermes runs unsandboxed so it can
    complete anything OpenClaw's sandbox blocked: posting terminal-state
    Slack replies and writing wiki learning files.
    """
    stats = {"slack_posted": 0, "wiki_written": 0}
    # Terminal broadcasts
    for task in storage.query_tasks(
        status_in=["done", "fail", "timeout"], limit=200,
    ):
        if task.get("terminal_broadcast_ok"):
            continue
        if not task.get("slack_thread_ts"):
            continue
        events = storage.get_task_events(task["task_id"])
        terminal = next(
            (e for e in reversed(events)
             if e["event_type"] in ("done", "fail", "timeout")),
            None,
        )
        if not terminal:
            continue
        result = (task.get("result") or "").strip()[:500]
        icon = {"done": "✅", "fail": "❌", "timeout": "⏰"}[task["status"]]
        verb = {"done": "DONE", "fail": "FAILED", "timeout": "TIMEOUT"}[task["status"]]
        line = f"{icon} `{task['task_id']}` {verb} by *{terminal['agent']}* (relayed)"
        if result:
            line += f"\n> {result}"
        if _slack_reply(task, line):
            storage.update_task(task["task_id"], terminal_broadcast_ok=1)
            stats["slack_posted"] += 1

    # Wiki learning writes
    for task in storage.query_tasks(
        status_in=["done", "fail"], limit=200,
    ):
        if task.get("learning_wiki_path"):
            continue
        events = storage.get_task_events(task["task_id"])
        terminal = next(
            (e for e in reversed(events)
             if e["event_type"] in ("done", "fail")),
            None,
        )
        if not terminal:
            continue
        try:
            import json as _json
            payload = _json.loads(terminal.get("payload") or "{}")
        except Exception:
            payload = {}
        learning = (payload.get("learning") or "").strip()
        if not learning:
            continue
        outcome = "done" if terminal["event_type"] == "done" else "fail"
        path = _write_learning(
            task["task_id"], terminal["agent"], learning, outcome=outcome,
        )
        if path:
            storage.update_task(task["task_id"], learning_wiki_path=str(path))
            stats["wiki_written"] += 1
    return stats


def nudge_stale_pending(*, stale_after_seconds: int = 180) -> List[Dict[str, Any]]:
    """Re-push any pending/ack task whose recipient hasn't progressed in a while.

    Called by the gateway watchdog every few minutes as a safety net when the
    initial push notification was missed (agent was down, crashed, etc.).
    Respects the per-task notify cooldown so it won't spam.
    """
    nudged: List[Dict[str, Any]] = []
    now = _now()
    # Check both recipient sides
    for recipient in VALID_AGENTS:
        for task in storage.query_tasks(
            to_agent=recipient,
            status_in=["pending", "ack"],
            limit=100,
        ):
            created = task.get("created_at") or 0
            acked = task.get("acked_at") or 0
            last_touch = max(acked, created)
            if now - last_touch < stale_after_seconds:
                continue
            _notify_agent(recipient, task, kind="reminder")
            nudged.append(task)
    return nudged


# ---------------------------------------------------------------------------
# Slack broadcasting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Active push notifications (cross-agent wake-up)
# ---------------------------------------------------------------------------

def _should_notify(agent: str, task_id: str, force: bool = False) -> bool:
    key = f"{agent}:{task_id}"
    now = _now()
    last = _last_notify_at.get(key, 0.0)
    if not force and (now - last) < _NOTIFY_COOLDOWN_SECONDS:
        return False
    _last_notify_at[key] = now
    return True


def _notify_openclaw(message: str) -> None:
    """Fire-and-forget: wake OpenClaw's main agent with a message.

    Uses `openclaw agent --agent main --message "..."`. Spawned as a
    detached subprocess so the caller never blocks waiting for OpenClaw's
    reply. stdout/stderr are redirected to a log file for post-mortem.
    """
    bin_path = shutil.which("openclaw")
    if not bin_path:
        logger.debug("agent_bus: openclaw binary not found; skip push")
        return
    log_dir = Path.home() / ".hermes" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "agent_bus_push.log", "ab", buffering=0)
    try:
        subprocess.Popen(
            [bin_path, "agent", "--agent", "main", "--message", message,
             "--timeout", "300"],
            stdout=log_file, stderr=log_file, stdin=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        logger.info("agent_bus: nudged openclaw (%d bytes)", len(message))
    except Exception as exc:
        logger.warning("agent_bus: failed to nudge openclaw: %s", exc)
    finally:
        # Popen keeps the fd open; close our handle so we don't leak.
        try:
            log_file.close()
        except Exception:
            pass


def _outbox_dir() -> Path:
    return Path.home() / ".openclaw" / "workspace" / ".agent-bus" / "outbox"


def _write_outbox_intent(*, target: str, message: str, task_id: Optional[str] = None) -> Optional[Path]:
    """Write a durable intent file. Always writable from both sandboxes
    because it lives inside OpenClaw's workspace.
    """
    import json as _json
    outbox = _outbox_dir()
    try:
        outbox.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("agent_bus: cannot create outbox dir: %s", exc)
        return None
    payload = {
        "intent": "notify_agent",
        "target": target,
        "task_id": task_id,
        "message": message,
        "created_at": _now(),
    }
    ts_ms = int(_now() * 1000)
    fname = f"{ts_ms:013d}_{task_id or 'nil'}_{uuid.uuid4().hex[:6]}.json"
    path = outbox / fname
    try:
        path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning("agent_bus: failed to write outbox intent: %s", exc)
        return None


def _notify_hermes_via_slack(message: str, *, task_id: Optional[str] = None) -> None:
    """Post a message to #hermes-inbox so Hermes processes it as a user
    message. Strategy:

    1. Try direct Slack post (works when caller is Hermes itself).
    2. On failure (e.g. OpenClaw sandbox blocks network), write an intent
       file to the shared outbox. Hermes's outbox watcher will pick it up
       within seconds and post from its unsandboxed side.
    """
    user_token = (
        os.environ.get("SLACK_USER_OAUTH_TOKEN")
        or _load_token_from_env_file("SLACK_USER_OAUTH_TOKEN")
    )
    bot_token = (
        os.environ.get("SLACK_BOT_TOKEN")
        or _load_token_from_env_file("SLACK_BOT_TOKEN")
    )
    token = user_token or bot_token
    if token:
        try:
            from slack_sdk import WebClient
            client = WebClient(token=token)
            client.chat_postMessage(
                channel=SLACK_CHANNEL_HERMES_INBOX,
                text=message,
                unfurl_links=False, unfurl_media=False,
            )
            logger.info("agent_bus: nudged hermes via slack (direct, %d bytes)", len(message))
            return
        except Exception as exc:
            logger.info(
                "agent_bus: direct slack post failed (%s); falling back to outbox",
                exc,
            )
    # Fallback: outbox intent
    path = _write_outbox_intent(target="hermes", message=message, task_id=task_id)
    if path:
        logger.info("agent_bus: wrote outbox intent for hermes at %s", path.name)


# ---------------------------------------------------------------------------
# User-facing proactive notifications ("boss gets pinged when X ships")
# ---------------------------------------------------------------------------

def _user_mention() -> str:
    """Return the first allowed Slack user as `<@UID>` mention, or empty."""
    allowed = (
        os.environ.get("SLACK_ALLOWED_USERS")
        or _load_token_from_env_file("SLACK_ALLOWED_USERS")
        or ""
    )
    first = (allowed.split(",") or [""])[0].strip()
    return f"<@{first}> " if first else ""


def _home_channel() -> str:
    return (
        os.environ.get("SLACK_HOME_CHANNEL")
        or _load_token_from_env_file("SLACK_HOME_CHANNEL")
        or SLACK_CHANNEL_HERMES_INBOX
    )


def _similar_fail_count(task: Dict[str, Any]) -> int:
    """Count prior failures/timeouts with similar goal directed at the same
    recipient. Used to escalate repeated flaps to the user with stronger
    wording ("third time in a row — maybe I should just do it").
    """
    try:
        goal_key = (task.get("goal") or "").strip()[:60]
        if not goal_key:
            return 0
        recent = storage.query_tasks(
            from_agent=task["from_agent"],
            status_in=["fail", "timeout"],
            limit=50,
        )
        return sum(
            1 for t in recent
            if t["task_id"] != task["task_id"]
            and (t.get("goal") or "").strip()[:60] == goal_key
            and t.get("to_agent") == task.get("to_agent")
        )
    except Exception:
        return 0


def _notify_user_of_outcome(task: Dict[str, Any]) -> bool:
    """Post a human-facing update to Hermes's home channel for a terminal
    task that Hermes originated. Returns True if posted.
    """
    if task.get("from_agent") != "hermes":
        return False  # only notify when Hermes was the dispatcher
    status = task.get("status")
    if status not in ("done", "fail", "timeout"):
        return False

    token = (
        os.environ.get("SLACK_BOT_TOKEN")
        or _load_token_from_env_file("SLACK_BOT_TOKEN")
    )
    if not token:
        return False

    tid = task["task_id"]
    to_agent = task.get("to_agent") or "?"
    goal = (task.get("goal") or "").strip()
    if len(goal) > 120:
        goal = goal[:117] + "…"
    result = (task.get("result") or "").strip()[:600]
    mention = _user_mention()

    if status == "done":
        text = (
            f"✅ {mention}`{tid}` 已完成 (by *{to_agent}*)\n"
            f"*Goal:* {goal}\n"
            f"*結果:* {result or '(空)'}"
        )
    elif status == "fail":
        prior = _similar_fail_count(task)
        escalation = ""
        if prior >= 1:
            escalation = f"\n⚠️ 這是同類任務第 {prior + 1} 次失敗。要我直接接手嗎？"
        text = (
            f"❌ {mention}`{tid}` 失敗 (by *{to_agent}*)\n"
            f"*Goal:* {goal}\n"
            f"*原因:* {result or '(未說明)'}"
            f"{escalation}"
        )
    else:  # timeout
        prior = _similar_fail_count(task)
        escalation = ""
        if prior >= 1:
            escalation = (
                f"\n⚠️ 這是同類任務第 {prior + 1} 次卡住。"
                f"*{to_agent}* 可能不適合這種任務，要我自己來嗎？"
            )
        else:
            escalation = f"\n{to_agent} 沒在 deadline 內回應，要我自己處理或延期？"
        text = (
            f"⏰ {mention}`{tid}` 超時 (assigned to *{to_agent}*)\n"
            f"*Goal:* {goal}"
            f"{escalation}"
        )

    try:
        from slack_sdk import WebClient
        WebClient(token=token).chat_postMessage(
            channel=_home_channel(),
            text=text,
            unfurl_links=False, unfurl_media=False,
        )
        return True
    except Exception as exc:
        logger.warning("agent_bus: user notification failed: %s", exc)
        return False


def ensure_user_notifications() -> int:
    """Watchdog hook: for every terminal task Hermes originated that hasn't
    had the user informed, try to send the notification now. Returns the
    count of fresh notifications sent."""
    sent = 0
    for task in storage.query_tasks(
        from_agent="hermes",
        status_in=["done", "fail", "timeout"],
        limit=200,
    ):
        if task.get("user_notified"):
            continue
        if _notify_user_of_outcome(task):
            storage.update_task(task["task_id"], user_notified=1)
            sent += 1
    return sent


def process_outbox() -> int:
    """Process pending outbox intents (typically written by OpenClaw).

    Runs from Hermes's unsandboxed context so network calls work. Returns
    the number of intents processed.
    """
    outbox = _outbox_dir()
    if not outbox.exists():
        return 0
    import json as _json
    processed = 0
    for path in sorted(outbox.glob("*.json")):
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("agent_bus: bad outbox intent %s: %s", path.name, exc)
            # Move aside so we don't keep hitting it
            try:
                path.rename(path.with_suffix(".json.bad"))
            except Exception:
                pass
            continue

        target = data.get("target")
        message = data.get("message") or ""
        if target == "hermes":
            token = (
                os.environ.get("SLACK_USER_OAUTH_TOKEN")
                or _load_token_from_env_file("SLACK_USER_OAUTH_TOKEN")
                or os.environ.get("SLACK_BOT_TOKEN")
                or _load_token_from_env_file("SLACK_BOT_TOKEN")
            )
            if not token:
                logger.warning("agent_bus: no slack token; leaving outbox intent %s", path.name)
                continue
            try:
                from slack_sdk import WebClient
                WebClient(token=token).chat_postMessage(
                    channel=SLACK_CHANNEL_HERMES_INBOX,
                    text=message,
                    unfurl_links=False, unfurl_media=False,
                )
            except Exception as exc:
                logger.warning("agent_bus: outbox slack post failed (%s); leaving %s",
                               exc, path.name)
                continue
        else:
            logger.info("agent_bus: unknown outbox target %r in %s", target, path.name)

        try:
            path.unlink()
        except Exception:
            pass
        processed += 1
    return processed


def _notify_agent(agent: str, task: Dict[str, Any], *, kind: str, extra: str = "") -> None:
    """Wake the given agent about this task. `kind` is 'new' | 'update' | 'reminder'."""
    tid = task["task_id"]
    if not _should_notify(agent, tid, force=(kind == "new")):
        return

    prefix_map = {
        "new": "🆕 新任務",
        "update": "🔄 任務更新",
        "reminder": "⏰ 提醒",
    }
    prefix = prefix_map.get(kind, "📬")
    goal = (task.get("goal") or "").strip()
    msg_core = (
        f"{prefix} `{tid}` ({task['from_agent']} → {task['to_agent']})\n"
        f"Goal: {goal}\n"
    )
    if task.get("success_criteria"):
        msg_core += f"Success: {task['success_criteria']}\n"
    if extra:
        msg_core += f"\n{extra}\n"

    if agent == "openclaw":
        hint = (
            "\n下一步：先執行 `hermes-bus inbox --agent openclaw --json` 確認任務，"
            f"然後 `hermes-bus ack {tid} --as openclaw`，處理完用 "
            f"`hermes-bus done {tid} --as openclaw --result \"...\" --learning \"...\"` 回報。"
        )
        _notify_openclaw(msg_core + hint)
    elif agent == "hermes":
        hint = (
            f"\n請用 `agent_bus` 工具：action=ack/progress/done/fail 回應這個任務 "
            f"(task_id={tid})。"
        )
        _notify_hermes_via_slack(msg_core + hint, task_id=tid)


def _load_token_from_env_file(key: str) -> Optional[str]:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return None
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'") or None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Slack broadcasting
# ---------------------------------------------------------------------------

def _slack_client():
    token = os.environ.get("SLACK_BOT_TOKEN") or _load_token_from_env_file("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("agent_bus: no SLACK_BOT_TOKEN found; skipping Slack broadcast")
        return None
    try:
        from slack_sdk import WebClient
        return WebClient(token=token)
    except Exception as exc:
        logger.warning("agent_bus: slack_sdk unavailable: %s", exc)
        return None


def _slack_post_assignment(task: Dict[str, Any]):
    client = _slack_client()
    if client is None:
        return None, None

    lines = [
        f"📬 *New task* `{task['task_id']}` — *{task['from_agent']}* → *{task['to_agent']}*",
        f"*Priority:* {task['priority']}"
        + (f"  *Deadline:* <!date^{int(task['deadline'])}^{{time}}|unknown>" if task.get("deadline") else ""),
        f"*Goal:* {task['goal']}",
    ]
    if task.get("success_criteria"):
        lines.append(f"*Success:* {task['success_criteria']}")
    if task.get("context"):
        lines.append(f"*Context:*\n```{task['context'][:800]}```")
    text = "\n".join(lines)
    try:
        resp = client.chat_postMessage(
            channel=SLACK_CHANNEL_OPS_EVOLUTION,
            text=text,
            unfurl_links=False,
            unfurl_media=False,
        )
        return resp.get("ts"), resp.get("channel")
    except Exception as exc:
        logger.warning("agent_bus: failed to post Slack assignment: %s", exc)
        return None, None


def _slack_reply(task: Dict[str, Any], text: str) -> bool:
    client = _slack_client()
    if client is None:
        return False
    thread_ts = task.get("slack_thread_ts")
    channel = task.get("slack_channel") or SLACK_CHANNEL_OPS_EVOLUTION
    try:
        client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
            unfurl_links=False,
            unfurl_media=False,
        )
        return True
    except Exception as exc:
        logger.warning("agent_bus: failed to post Slack reply: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Wiki learning sync
# ---------------------------------------------------------------------------

def _wiki_memory_dir() -> Optional[Path]:
    """Return ~/wiki/memory/ if it exists, else None."""
    vault = os.environ.get("OBSIDIAN_VAULT_PATH") or str(Path.home() / "wiki")
    path = Path(vault).expanduser() / "memory"
    return path if path.exists() else None


def _write_learning(task_id: str, agent: str, learning: str, *, outcome: str) -> Optional[Path]:
    """Write a learning file in Obsidian-schema format.

    The vault at ~/wiki/ follows a strict schema (see ~/wiki/SCHEMA.md):
    every page has title/created/updated/type/tags frontmatter and should
    wikilink out to related pages so the graph stays connected. We also
    append a one-line entry to log.md so changes show up in the action log.
    """
    dir_ = _wiki_memory_dir()
    if dir_ is None:
        logger.info("agent_bus: wiki memory dir missing, skipping learning sync")
        return None
    ts = time.strftime("%Y-%m-%d", time.localtime())
    fname = f"{ts}_agent-bus_{task_id}_{outcome}.md"
    title = f"Agent-bus {task_id} ({outcome})"

    # Grab the task so we can show context and cross-agent wikilinks
    task = storage.get_task_row(task_id) or {}
    from_agent = task.get("from_agent") or ""
    to_agent = task.get("to_agent") or agent
    goal = (task.get("goal") or "").strip()
    result = (task.get("result") or "").strip()

    role_link = {
        "hermes": "[[hermes-role]]",
        "openclaw": "[[openclaw-role]]",
    }
    from_link = role_link.get(from_agent, from_agent)
    to_link = role_link.get(to_agent, to_agent)
    agent_link = role_link.get(agent, agent)

    tags = ["agent-bus", f"outcome/{outcome}", f"agent/{agent}"]
    tags_yaml = "[" + ", ".join(tags) + "]"

    body = (
        f"---\n"
        f"title: {title}\n"
        f"created: {ts}\n"
        f"updated: {ts}\n"
        f"type: memory\n"
        f"tags: {tags_yaml}\n"
        f"task_id: {task_id}\n"
        f"agent: {agent}\n"
        f"outcome: {outcome}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"Task `{task_id}` — {from_link} → {to_link}, reported by {agent_link}.\n\n"
        f"Part of [[agent-bus-moc]] (two-agent task bus).\n"
        f"See protocol: [[collaboration-protocol]].\n\n"
        f"## Goal\n\n"
        f"{goal or '(no goal captured)'}\n\n"
    )
    if result:
        heading = {"done": "Result", "fail": "Failure reason", "timeout": "Timeout context"}[outcome] if outcome in ("done", "fail", "timeout") else "Result"
        body += f"## {heading}\n\n{result[:800]}\n\n"
    body += f"## Learning\n\n{learning.strip()}\n"

    path = dir_ / fname
    try:
        path.write_text(body, encoding="utf-8")
    except Exception as exc:
        logger.warning("agent_bus: failed to write learning file: %s", exc)
        return None

    # Append to log.md so the action log sees it.
    _append_wiki_log(
        f"[[memory/{fname[:-3]}]] — agent-bus {outcome} by {agent} "
        f"({from_agent}→{to_agent}, {task_id})"
    )
    # Regenerate the MOC so new learning is linked from the hub page.
    _rebuild_agent_bus_moc()
    return path


def _append_wiki_log(line: str) -> None:
    log_path = None
    dir_ = _wiki_memory_dir()
    if dir_ is None:
        return
    vault = dir_.parent
    log_path = vault / "log.md"
    if not log_path.exists():
        return
    ts = time.strftime("%Y-%m-%d", time.localtime())
    entry = f"\n- {ts} — {line}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.debug("agent_bus: log.md append failed: %s", exc)


def _rebuild_agent_bus_moc() -> None:
    """Rebuild ~/wiki/agent-bus-moc.md listing all agent-bus learnings,
    grouped by outcome and date. Runs every time a new learning is written.
    Lightweight: just scans memory/ for filename pattern.
    """
    dir_ = _wiki_memory_dir()
    if dir_ is None:
        return
    vault = dir_.parent
    moc_path = vault / "agent-bus-moc.md"
    try:
        entries = sorted(dir_.glob("*_agent-bus_*.md"), reverse=True)
    except Exception:
        return

    by_outcome: Dict[str, List[Path]] = {"done": [], "fail": [], "timeout": []}
    for p in entries:
        # filename: YYYY-MM-DD_agent-bus_T-XXXXXX_<outcome>.md
        stem = p.stem
        for oc in ("done", "fail", "timeout"):
            if stem.endswith("_" + oc):
                by_outcome.setdefault(oc, []).append(p)
                break

    ts = time.strftime("%Y-%m-%d", time.localtime())
    lines: List[str] = [
        "---",
        "title: Agent-Bus MOC",
        f"updated: {ts}",
        "type: summary",
        "tags: [agent-bus, moc]",
        "---",
        "",
        "# Agent-Bus MOC",
        "",
        "Map of Content for the two-agent task bus ([[hermes-role]] ↔ [[openclaw-role]]).",
        "",
        "See [[collaboration-protocol]] for the handoff rules, and "
        "`~/wiki/memory/` for the full ledger of learnings.",
        "",
        "## Recent learnings",
        "",
    ]
    for outcome, icon in [("done", "✅"), ("fail", "❌"), ("timeout", "⏰")]:
        rows = by_outcome.get(outcome, [])
        if not rows:
            continue
        lines.append(f"### {icon} {outcome} ({len(rows)})")
        lines.append("")
        for p in rows[:30]:
            lines.append(f"- [[memory/{p.stem}]]")
        if len(rows) > 30:
            lines.append(f"- _(+{len(rows) - 30} older entries)_")
        lines.append("")
    lines.append("## Architecture reference")
    lines.append("")
    lines.append(
        "The task bus lives in `~/dev/hermes-agent/agent_bus/` "
        "(SQLite at `~/.openclaw/workspace/.agent-bus/agent_bus.db`). "
        "Each `done`/`fail` with `--learning` writes a file into "
        "`~/wiki/memory/` and this MOC auto-rebuilds."
    )
    try:
        moc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.debug("agent_bus: MOC write failed: %s", exc)
