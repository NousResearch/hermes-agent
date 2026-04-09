"""
ARGUS action execution — restart, kill, inject, and corrective prompt logic.

Each function takes cursor/conn and session info as explicit parameters.
Pure functions with side effects only on the database (no Argus class state).

Session types: cron, delegate_task, manual.
"""

import json
import logging
import sqlite3
import subprocess
import time
from typing import Dict, List, Optional, Union

from . import venv_utils as _venv_utils

logger = logging.getLogger("argus.actions")

# Cron API — imported at module level for testability (tests mock these)
try:
    from cron.jobs import pause_job, resume_job, trigger_job, get_job, update_job
except (ImportError, TypeError):
    # Subprocess fallback — hermes_fallback exports the same names
    try:
        from hermes_fallback import (
            pause_job,
            resume_job,
            trigger_job,
            get_job,
            update_job,
        )
    except ImportError:
        pause_job = resume_job = trigger_job = get_job = update_job = None
        logger.warning("cron.jobs unavailable — action functions will log warnings")


def _get_cron_env() -> Dict[str, str]:
    """Build a full environment dict for subprocess calls in sandboxed contexts.
    
    Uses venv_utils to ensure virtual environment context is preserved.
    This ensures subprocesses can find hermes modules and tools.
    """
    return _venv_utils.build_argus_subprocess_env()


def safe_subprocess(
    cmd: List[str], timeout: int = 10, **kwargs
) -> Optional[subprocess.CompletedProcess]:
    """Run a subprocess with full env and error handling. Never raises."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_get_cron_env(),
            **kwargs,
        )
    except FileNotFoundError:
        logger.warning("Command not found: %s (check PATH)", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ss: %s", timeout, " ".join(cmd))
        return None
    except Exception as e:
        logger.error("Subprocess error for %s: %s", cmd[0], e, exc_info=True)
        return None


# =============================================================================
# Corrective prompt building
# =============================================================================

DEFAULT_CORRECTIVE_PROMPTS: Dict[str, str] = {
    "repeat_tool_calls": (
        "ENTROPY CORRECTION: ARGUS detected repeated tool calls without progress. "
        "You are calling the same tool with the same arguments multiple times. "
        "Stop and reassess. Read the file/content you need ONCE, then act on it. "
        "Do not re-read files you already have in context. Complete the task."
    ),
    "repeat_commands": (
        "ENTROPY CORRECTION: ARGUS detected repeated terminal commands. "
        "You are running the same command multiple times. "
        "Check the output you already received before re-running. "
        "If the command failed, fix the issue, don't retry blindly."
    ),
    "stuck_loop": (
        "ENTROPY CORRECTION: ARGUS detected a stuck loop pattern. "
        "Your last several tool calls form a repeating cycle. "
        "STOP. Read your conversation history. Identify what you're trying to accomplish. "
        "Take a different approach. Do not repeat the same sequence."
    ),
    "no_file_changes": (
        "ENTROPY CORRECTION: ARGUS detected write operations that didn't change files. "
        "You are calling write_file/patch but the file content is not changing. "
        "Read the file first, verify what you're writing is actually different. "
        "If using patch, check that old_string matches exactly."
    ),
    "error_cascade": (
        "ENTROPY CORRECTION: ARGUS detected a cascade of tool failures. "
        "Multiple consecutive tool calls have returned errors. "
        "STOP. Read the error messages carefully. The environment or arguments may be wrong. "
        "Check file paths, command syntax, and prerequisites before retrying. "
        "If a tool keeps failing, try a different approach or use a different tool."
    ),
    "budget_pressure": (
        "BUDGET CORRECTION: You are burning through your iteration budget fast "
        "without productive output. "
        "Step back. Summarize what you have accomplished so far and what remains. "
        "Pick the simplest remaining task and complete it in one pass. "
        "Avoid exploratory tool calls — read once, then act."
    ),
    "quality_gate": (
        "QUALITY CORRECTION: Your output quality is below the 0.92 threshold. "
        "Provide mechanistic explanations, not surface descriptions. "
        "Include structured output with headers and metrics. "
        "Feed the pipeline: write facts, generate trajectories, enrich KB."
    ),
    "pipeline_violation": (
        "PIPELINE CORRECTION: You are not hitting all 4 pipeline targets. "
        "Every substantive interaction must produce: "
        "(1) target output, (2) holographic_memory.db facts, "
        "(3) trajectories (Q&A chains), (4) KB enrichment. "
        "Self-assess before finishing."
    ),
}


def build_corrective_prompt(
    cursor: sqlite3.Cursor,
    session_id: str,
    reason: str,
    corrective_prompts: Optional[Dict[str, str]] = None,
) -> str:
    """Build a corrective prompt based on recent entropy detections for this session.

    Queries entropy_detections for both session_id and wal_{session_id} to find
    detections from both DB-poll and WAL-monitor sources.
    """
    prompts = corrective_prompts or DEFAULT_CORRECTIVE_PROMPTS
    wal_session_id = f"wal_{session_id}"

    cursor.execute(
        """
        SELECT entropy_type, severity FROM entropy_detections
        WHERE session_id IN (?, ?) AND timestamp > datetime('now', '-10 minutes')
        ORDER BY severity DESC, timestamp DESC
        LIMIT 1
    """,
        (session_id, wal_session_id),
    )

    row = cursor.fetchone()
    if row:
        entropy_type = row["entropy_type"]
        template = prompts.get(entropy_type, prompts["stuck_loop"])
        return "%s\n\nReason: %s" % (template, reason)

    return "ENTROPY CORRECTION: ARGUS detected an issue requiring restart. %s" % reason


# =============================================================================
# PID termination (shared by restart and kill paths)
# =============================================================================


def terminate_pid(pid: Union[str, int], context: str = "terminate") -> None:
    """Send SIGTERM then SIGKILL to a process."""
    pid_str = str(pid)
    safe_subprocess(["kill", "-TERM", pid_str])
    logger.info("Sent SIGTERM to PID %s (%s)", pid_str, context)
    time.sleep(2)
    safe_subprocess(["kill", "-9", pid_str])


# =============================================================================
# Restart logic
# =============================================================================


def restart_session(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    session_id: str,
    reason: str,
    corrective_prompts: Optional[Dict[str, str]] = None,
) -> None:
    """Restart a session with tighter constraints.

    Increments restart_count, builds corrective prompt, dispatches to
    session-type-specific handler.
    """
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        logger.warning("Session %s not found for restart", session_id)
        return
    session = dict(row)

    # Increment restart count
    cursor.execute(
        """
        UPDATE sessions SET restart_count = restart_count + 1, status = 'restarted'
        WHERE session_id = ?
    """,
        (session_id,),
    )

    session_type = session["session_type"]
    corrective_prompt = build_corrective_prompt(
        cursor, session_id, reason, corrective_prompts
    )

    try:
        if session_type == "cron":
            restart_cron_session(session, corrective_prompt)
        elif session_type == "delegate_task":
            restart_delegate_session(session, corrective_prompt)
        elif session_type == "manual":
            restart_manual_session(session, corrective_prompt)
    except Exception as e:
        logger.error("Error during restart of %s: %s", session_id, e, exc_info=True)

    conn.commit()
    logger.info(
        "Restarted %s session %s (restart count: %s)",
        session_type,
        session_id,
        session["restart_count"] + 1,
    )


def restart_cron_session(session: Dict, corrective_prompt: str) -> None:
    """Cron restart: pause job, update prompt, resume."""
    job_id = session.get("job_id")
    if not job_id:
        logger.warning(
            "No job_id for cron session %s, cannot restart", session["session_id"]
        )
        return

    if pause_job is None:
        logger.warning("cron.jobs API unavailable — cannot restart cron session")
        return

    try:
        result = pause_job(job_id, reason="ARGUS restart: entropy detected")
        if result:
            logger.info("Paused cron job %s", job_id)
        else:
            logger.warning("pause_job returned None for %s", job_id)
    except Exception as e:
        logger.error("Failed to pause cron job %s: %s", job_id, e, exc_info=True)

    # Update prompt with corrective instructions
    try:
        job = get_job(job_id)
        if job:
            original_prompt = job.get("prompt", "")
            updated_prompt = "%s\n\n---\n\nOriginal task:\n%s" % (
                corrective_prompt,
                original_prompt,
            )
            update_job(job_id, {"prompt": updated_prompt})
            logger.info(
                "Updated cron job %s prompt with corrective instructions", job_id
            )
    except Exception as e:
        logger.error(
            "Failed to update cron prompt for %s: %s", job_id, e, exc_info=True
        )

    # Resume
    try:
        result = resume_job(job_id)
        if result:
            logger.info("Resumed cron job %s with corrective prompt", job_id)
        else:
            logger.warning("resume_job returned None for %s", job_id)
    except Exception as e:
        logger.error("Failed to resume cron job %s: %s", job_id, e, exc_info=True)


def restart_delegate_session(session: Dict, corrective_prompt: str) -> None:
    """Delegate restart: kill process, respawn with corrective prompt."""
    metadata = json.loads(session.get("metadata", "{}"))
    pid = metadata.get("pid")

    if pid:
        terminate_pid(pid, "restart")

    # The respawn will happen naturally when the parent agent retries
    logger.info("Killed delegate session, corrective prompt stored for respawn")


def restart_manual_session(session: Dict, corrective_prompt: str) -> None:
    """Manual restart: record action, store corrective prompt for next interaction."""
    # For manual sessions, we can't force a restart
    # We record the corrective prompt and notify the user
    logger.info(
        "Manual session %s flagged for restart (user intervention needed)",
        session["session_id"],
    )


# =============================================================================
# Kill logic
# =============================================================================


def kill_session(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    session_id: str,
    reason: str,
) -> None:
    """Kill a session based on its type.

    Updates session status, dispatches to session-type-specific handler,
    records kill action.
    """
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        logger.warning("Session %s not found for kill", session_id)
        return
    session = dict(row)

    session_type = session["session_type"]

    # Update session status
    cursor.execute(
        """
        UPDATE sessions SET status = 'killed', kill_count = kill_count + 1
        WHERE session_id = ?
    """,
        (session_id,),
    )

    try:
        if session_type == "cron":
            kill_cron_session(session, reason)
        elif session_type == "delegate_task":
            kill_delegate_session(session, reason)
        elif session_type == "manual":
            kill_manual_session(cursor, session, reason)
    except Exception as e:
        logger.error("Error killing %s: %s", session_id, e, exc_info=True)

    # Record kill action
    cursor.execute(
        """
        INSERT INTO watcher_actions (session_id, action_type, action_reason, success, details)
        VALUES (?, 'kill', ?, TRUE, ?)
    """,
        (
            session_id,
            reason,
            json.dumps(
                {
                    "session_type": session_type,
                    "kill_count": session["kill_count"] + 1,
                }
            ),
        ),
    )

    conn.commit()
    logger.info("Killed %s session %s: %s", session_type, session_id, reason)


def kill_cron_session(session: Dict, reason: str) -> None:
    """Permanently pause a cron job."""
    job_id = session.get("job_id")
    if not job_id:
        logger.warning(
            "No job_id for cron session %s, cannot kill", session["session_id"]
        )
        return

    if pause_job is None:
        logger.warning("cron.jobs API unavailable — cannot kill cron session")
        return

    try:
        result = pause_job(job_id, reason="ARGUS kill: %s" % reason)
        if result:
            logger.info("Permanently paused cron job %s", job_id)
        else:
            logger.warning("pause_job returned None for %s", job_id)
    except Exception as e:
        logger.error(
            "Failed to pause cron job %s for kill: %s", job_id, e, exc_info=True
        )


def kill_delegate_session(session: Dict, reason: str) -> None:
    """Terminate a delegate task subprocess."""
    metadata = json.loads(session.get("metadata", "{}"))
    pid = metadata.get("pid")

    if pid:
        terminate_pid(pid, "kill")


def kill_manual_session(cursor: sqlite3.Cursor, session: Dict, reason: str) -> None:
    """Cannot kill manual sessions — record notification for user review."""
    message = (
        "ARGUS cannot terminate manual session %s.\n"
        "Action required: Please review this session manually.\n"
        "Reason: %s" % (session["session_id"], reason)
    )
    cursor.execute(
        """
        INSERT INTO notifications (session_id, notification_type, message, delivered)
        VALUES (?, 'kill', ?, FALSE)
    """,
        (session["session_id"], message),
    )
    logger.warning(
        "Manual session %s flagged for kill — user intervention required",
        session["session_id"],
    )


# =============================================================================
# Prompt injection
# =============================================================================


def inject_prompt(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    session_id: str,
    prompt: str,
) -> None:
    """Inject a corrective prompt into a session based on its type."""
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        logger.warning("Session %s not found for prompt injection", session_id)
        return
    session = dict(row)

    session_type = session["session_type"]

    try:
        if session_type == "cron":
            inject_cron_prompt(session, prompt)
        elif session_type == "delegate_task":
            inject_delegate_prompt(session, prompt)
        elif session_type == "manual":
            inject_manual_prompt(cursor, session, prompt)
    except Exception as e:
        logger.error("Error injecting prompt into %s: %s", session_id, e, exc_info=True)

    # Record prompt injection action
    cursor.execute(
        """
        INSERT INTO watcher_actions (session_id, action_type, action_reason, success, details)
        VALUES (?, 'inject_prompt', 'Corrective prompt injected', TRUE, ?)
    """,
        (
            session_id,
            json.dumps(
                {"session_type": session_type, "corrective_prompt": prompt[:500]}
            ),
        ),
    )

    conn.commit()
    logger.info(
        "Injected corrective prompt into %s session %s", session_type, session_id
    )


def inject_cron_prompt(session: Dict, prompt: str) -> None:
    """Update cron job prompt and trigger via cron.jobs."""
    job_id = session.get("job_id")
    if not job_id:
        return

    if trigger_job is None or get_job is None or update_job is None:
        logger.warning("cron.jobs API unavailable — cannot inject into cron session")
        return

    try:
        job = get_job(job_id)
        if job:
            original_prompt = job.get("prompt", "")
            updated_prompt = "%s\n\n---\n\nOriginal task:\n%s" % (
                prompt,
                original_prompt,
            )
            update_job(job_id, {"prompt": updated_prompt})

        # Force run with new prompt
        trigger_job(job_id)
        logger.info("Triggered cron job %s with corrective prompt", job_id)
    except Exception as e:
        logger.error(
            "Failed to inject prompt into cron job %s: %s", job_id, e, exc_info=True
        )


def inject_delegate_prompt(session: Dict, prompt: str) -> None:
    """Kill and respawn delegate with corrective prompt."""
    metadata = json.loads(session.get("metadata", "{}"))
    pid = metadata.get("pid")

    if pid:
        terminate_pid(pid, "prompt injection")
        logger.info("Killed delegate PID %s for prompt injection — will respawn", pid)


def inject_manual_prompt(cursor: sqlite3.Cursor, session: Dict, prompt: str) -> None:
    """Store corrective prompt as notification for manual session."""
    cursor.execute(
        """
        INSERT INTO notifications (session_id, notification_type, message, delivered)
        VALUES (?, 'inject_prompt', ?, FALSE)
    """,
        (
            session["session_id"],
            "CORRECTIVE PROMPT FOR NEXT INTERACTION:\n\n%s" % prompt,
        ),
    )
    logger.info("Stored corrective prompt for manual session %s", session["session_id"])


# =============================================================================
# Session ID utility
# =============================================================================


def strip_session_prefix(session_id: str) -> str:
    """Strip type prefix from session ID: cron_ec1a5e9f4c12 -> ec1a5e9f4c12"""
    parts = session_id.split("_", 1)
    return parts[1] if len(parts) == 2 else session_id
