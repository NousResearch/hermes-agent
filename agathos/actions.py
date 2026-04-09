"""
Agathos action execution — restart, kill, inject, and corrective prompt logic.

Each function takes cursor/conn and session info as explicit parameters.
Pure functions with side effects only on the database (no Agathos class state).

Session types: cron, delegate_task, manual.
"""

import json
import logging
import sqlite3
import subprocess
import time
from typing import Dict, List, Optional, Union

from . import venv_utils as _venv_utils

logger = logging.getLogger("agathos.actions")

# Cron API — imported at module level for testability (tests mock these)
try:
    from cron.jobs import pause_job, resume_job, trigger_job, get_job, update_job
except (ImportError, TypeError):
    # Subprocess fallback — hermes_fallback exports the same names
    try:
        from .hermes_fallback import (
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
    return _venv_utils.build_agathos_subprocess_env()


def safe_subprocess(
    cmd: List[str], timeout: int = 10, **kwargs
) -> Optional[subprocess.CompletedProcess]:
    """Run a subprocess with full environment and error handling. Never raises.

    Wraps subprocess.run with:
    - Virtual environment context (via _get_cron_env)
    - Timeout enforcement
    - Exception catching (FileNotFound, Timeout, generic)
    - Logging of all failure modes

    Args:
        cmd: Command and arguments as list
        timeout: Seconds to wait before killing (default: 10)
        **kwargs: Additional subprocess.run arguments

    Returns:
        CompletedProcess on success, None on any failure

    Side effects: Logs warnings/errors via agathos.actions logger.
    """
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
        "ENTROPY CORRECTION: Agathos detected repeated tool calls without progress. "
        "You are calling the same tool with the same arguments multiple times. "
        "Stop and reassess. Read the file/content you need ONCE, then act on it. "
        "Do not re-read files you already have in context. Complete the task."
    ),
    "repeat_commands": (
        "ENTROPY CORRECTION: Agathos detected repeated terminal commands. "
        "You are running the same command multiple times. "
        "Check the output you already received before re-running. "
        "If the command failed, fix the issue, don't retry blindly."
    ),
    "stuck_loop": (
        "ENTROPY CORRECTION: Agathos detected a stuck loop pattern. "
        "Your last several tool calls form a repeating cycle. "
        "STOP. Read your conversation history. Identify what you're trying to accomplish. "
        "Take a different approach. Do not repeat the same sequence."
    ),
    "no_file_changes": (
        "ENTROPY CORRECTION: Agathos detected write operations that didn't change files. "
        "You are calling write_file/patch but the file content is not changing. "
        "Read the file first, verify what you're writing is actually different. "
        "If using patch, check that old_string matches exactly."
    ),
    "error_cascade": (
        "ENTROPY CORRECTION: Agathos detected a cascade of tool failures. "
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

    Queries entropy_detections table for both session_id and wal_{session_id}
    (WAL monitor prefix) to find detections from both polling sources. Returns
    a contextual corrective prompt that explains what entropy was detected
    and how to correct it.

    Args:
        cursor: Database cursor for agathos.db
        session_id: Session to build prompt for
        reason: Human-readable reason for the corrective action
        corrective_prompts: Optional dict mapping entropy_type to prompt templates.
            Defaults to DEFAULT_CORRECTIVE_PROMPTS.

    Returns:
        Formatted corrective prompt string combining the template and reason.
        If no recent entropy detection found, returns generic restart message.

    Query scope: Last 10 minutes of entropy_detections table.
    Query sources: Both session_id and wal_session_id (for WAL monitor coverage).
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

    return "ENTROPY CORRECTION: Agathos detected an issue requiring restart. %s" % reason


# =============================================================================
# PID termination (shared by restart and kill paths)
# =============================================================================


def terminate_pid(pid: Union[str, int], context: str = "terminate") -> None:
    """Send SIGTERM then SIGKILL to a process with graceful fallback.

    First sends SIGTERM (-TERM) and waits 2 seconds for graceful shutdown.
    Then sends SIGKILL (-9) for forced termination if process still exists.

    Args:
        pid: Process ID to terminate (int or string)
        context: Description of why process is being terminated (for logging)

    Side effects:
        - Sends signals to PID
        - Logs info via agathos.actions logger
        - 2-second delay between signals

    Returns:
        None (errors logged, never raises)
    """
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
    """Restart a session with tighter constraints and corrective guidance.

    Orchestrates the restart workflow:
    1. Fetches session from database
    2. Increments restart_count and updates status to 'restarted'
    3. Builds corrective prompt based on recent entropy detections
    4. Dispatches to session-type-specific restart handler (cron/delegate/manual)
    5. Commits database changes and logs result

    Args:
        cursor: Database cursor for agathos.db
        conn: Database connection for commit
        session_id: Session to restart
        reason: Human-readable reason for restart (logged and used in prompts)
        corrective_prompts: Optional dict mapping entropy_type to prompt templates

    Side effects:
        - Updates sessions table (restart_count, status)
        - Dispatches to restart_cron_session, restart_delegate_session,
          or restart_manual_session based on session_type
        - Commits connection
        - Logs info via agathos.actions logger
        - Logs errors if restart handlers fail

    Returns:
        None (errors logged, never raises)

    Session types handled: cron, delegate_task, manual
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
    """Restart a cron session: pause job, update prompt, resume.

    Implementation for cron session restarts:
    1. Pauses the cron job via pause_job API
    2. Updates job prompt with corrective instructions prepended
    3. Resumes the job via resume_job API

    Args:
        session: Session dict with 'session_id' and 'job_id' keys
        corrective_prompt: Instructions to prepend to job prompt

    Side effects:
        - Calls pause_job, get_job, update_job, resume_job APIs
        - Logs info/warning/error via agathos.actions logger

    Returns:
        None (errors logged, never raises)

    Fallback behavior: If cron.jobs API unavailable, logs warning and returns.
    """
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
        result = pause_job(job_id, reason="Agathos restart: entropy detected")
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
    """Restart a delegate task session: kill process, prepare for respawn.

    Implementation for delegate_task session restarts:
    1. Extracts PID from session metadata
    2. Terminates the process via terminate_pid
    3. Logs that corrective prompt is stored for respawn

    Note: The actual respawn happens when the parent agent retries.
    Agathos does not spawn processes directly — it only terminates
    and relies on Hermes retry logic.

    Args:
        session: Session dict with 'session_id' and 'metadata' (JSON with 'pid')
        corrective_prompt: Instructions stored for next respawn

    Side effects:
        - Calls terminate_pid on the delegate process
        - Logs info via agathos.actions logger

    Returns:
        None (errors logged via terminate_pid, never raises)
    """
    metadata = json.loads(session.get("metadata", "{}"))
    pid = metadata.get("pid")

    if pid:
        terminate_pid(pid, "restart")

    # The respawn will happen naturally when the parent agent retries
    logger.info("Killed delegate session, corrective prompt stored for respawn")


def restart_manual_session(session: Dict, corrective_prompt: str) -> None:
    """Restart a manual session: flag for user intervention.

    Implementation for manual session restarts:
    Manual sessions cannot be forcibly restarted (they are interactive
    user sessions). Agathos records the corrective prompt and flags
    the session status, but the user must manually restart or correct.

    Args:
        session: Session dict with 'session_id'
        corrective_prompt: Instructions that would be shown if restart were possible

    Side effects:
        - Logs info via agathos.actions logger

    Returns:
        None (no actual restart performed, user must intervene)

    Note: Manual sessions rely on user awareness, not automated action.
    The corrective prompt is logged but not automatically injected.
    """
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
    """Kill (terminate) a session based on its type.

    Orchestrates the kill workflow:
    1. Fetches session from database
    2. Updates session status to 'killed' and increments kill_count
    3. Dispatches to session-type-specific kill handler (cron/delegate/manual)
    4. Records kill action in watcher_actions table
    5. Commits database changes

    Args:
        cursor: Database cursor for agathos.db
        conn: Database connection for commit
        session_id: Session to kill
        reason: Human-readable reason for kill (logged and recorded)

    Side effects:
        - Updates sessions table (kill_count, status)
        - Inserts into watcher_actions table
        - Dispatches to kill_cron_session, kill_delegate_session,
          or kill_manual_session based on session_type
        - Commits connection
        - Logs info/error via agathos.actions logger

    Returns:
        None (errors logged, never raises)

    Session types handled: cron, delegate_task, manual
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
    """Kill a cron session: permanently pause the job.

    Implementation for cron session kills:
    1. Extracts job_id from session
    2. Pauses the cron job via pause_job API with kill reason
    3. Logs result

    Args:
        session: Session dict with 'session_id' and 'job_id' keys
        reason: Human-readable reason for kill (included in pause reason)

    Side effects:
        - Calls pause_job API
        - Logs info/warning/error via agathos.actions logger

    Returns:
        None (errors logged, never raises)

    Difference from restart: Kill is permanent (no resume), restart is temporary.
    Fallback: If cron.jobs API unavailable, logs warning and returns.
    """
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
        result = pause_job(job_id, reason="Agathos kill: %s" % reason)
        if result:
            logger.info("Permanently paused cron job %s", job_id)
        else:
            logger.warning("pause_job returned None for %s", job_id)
    except Exception as e:
        logger.error(
            "Failed to pause cron job %s for kill: %s", job_id, e, exc_info=True
        )


def kill_delegate_session(session: Dict, reason: str) -> None:
    """Kill a delegate task session: terminate the subprocess.

    Implementation for delegate_task session kills:
    1. Extracts PID from session metadata
    2. Terminates the process via terminate_pid
    3. Logs result

    Args:
        session: Session dict with 'session_id' and 'metadata' (JSON with 'pid')
        reason: Human-readable reason for kill (for logging context)

    Side effects:
        - Calls terminate_pid on the delegate process
        - Logs info via agathos.actions logger

    Returns:
        None (errors logged via terminate_pid, never raises)

    Difference from restart: Kill terminates without preparing for respawn.
    The process is dead; parent must handle failure.
    """
    metadata = json.loads(session.get("metadata", "{}"))
    pid = metadata.get("pid")

    if pid:
        terminate_pid(pid, "kill")


def kill_manual_session(cursor: sqlite3.Cursor, session: Dict, reason: str) -> None:
    """Kill a manual session: record notification for user review.

    Implementation for manual session kills:
    Manual sessions cannot be forcibly killed (they are interactive
    user sessions). Agathos records a notification in the notifications
    table requesting user review.

    Args:
        cursor: Database cursor for agathos.db
        session: Session dict with 'session_id'
        reason: Human-readable reason for kill (included in notification)

    Side effects:
        - Inserts into notifications table (type='kill', delivered=FALSE)
        - Logs info via agathos.actions logger

    Returns:
        None (no actual kill performed, user must intervene)

    Note: Manual sessions rely on user awareness, not automated action.
    The notification is queued for delivery via notification system.
    """
    message = (
        "Agathos cannot terminate manual session %s.\n"
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
