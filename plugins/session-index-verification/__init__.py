"""
Session Index Verification Plugin for Hermes

Automatically detects and reconstructs sessions that exist as JSON files
but are not indexed in the SQLite database. This commonly happens when
Hermes crashes or restarts before the session is fully persisted.

Usage:
    The plugin runs automatically on every session start via the 
    `on_session_start` hook. No manual intervention required.

Configuration:
    None required. The plugin uses the standard Hermes paths:
    - Sessions: ~/.hermes/sessions/
    - Database: ~/.hermes/state.db

Author: Hermes Community
Version: 1.0.0
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Standard Hermes paths
DEFAULT_DB_PATH = Path.home() / ".hermes" / "state.db"
SESSIONS_DIR = Path.home() / ".hermes" / "sessions"


def get_missing_sessions() -> List[Tuple[str, Path]]:
    """
    Find sessions that exist as JSON but not in SQLite database.
    
    Returns:
        List of (session_id, json_file_path) tuples
    """
    if not DEFAULT_DB_PATH.exists():
        logger.warning("Database not found: %s", DEFAULT_DB_PATH)
        return []
    
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # Get all session IDs from database
    cursor.execute("SELECT id FROM sessions")
    db_sessions = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    # Check JSON files
    missing = []
    if SESSIONS_DIR.exists():
        for json_file in SESSIONS_DIR.glob("session_*.json"):
            session_id = json_file.stem.replace("session_", "")
            if session_id not in db_sessions:
                missing.append((session_id, json_file))
    
    return missing


def reconstruct_session(session_id: str, json_file: Path) -> Tuple[bool, int]:
    """
    Reconstruct a single missing session in the database.
    
    Args:
        session_id: The session ID
        json_file: Path to the JSON session file
        
    Returns:
        (success: bool, messages_inserted: int)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", json_file, e)
        return False, 0
    
    # Extract metadata
    session_start = data.get('session_start')
    platform = data.get('platform', 'cli')
    model = data.get('model', 'unknown')
    title = data.get('title')
    messages = data.get('messages', [])
    
    # Parse timestamp
    try:
        if session_start:
            dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            started_ts = dt.timestamp()
        else:
            started_ts = time.time()
    except:
        started_ts = time.time()
    
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # Insert session
    try:
        cursor.execute(
            """INSERT INTO sessions 
               (id, source, model, started_at, ended_at, title, message_count, tool_call_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, platform, model, started_ts, None, title, len(messages), 0)
        )
    except sqlite3.IntegrityError:
        pass  # Session already exists
    except Exception as e:
        logger.error("Failed to insert session %s: %s", session_id, e)
        conn.close()
        return False, 0
    
    # Insert messages
    messages_inserted = 0
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        tool_name = msg.get('tool_name')
        
        try:
            cursor.execute(
                """INSERT INTO messages 
                   (session_id, role, content, tool_name, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, tool_name, time.time())
            )
            messages_inserted += 1
        except Exception:
            pass  # Skip individual message errors
    
    conn.commit()
    conn.close()
    
    return True, messages_inserted


def verify_and_reconstruct() -> Tuple[int, int]:
    """
    Main verification routine. Finds and reconstructs all missing sessions.
    
    Returns:
        (sessions_reconstructed: int, messages_inserted: int)
    """
    missing = get_missing_sessions()
    
    if not missing:
        return 0, 0
    
    logger.info("Found %d unindexed sessions, reconstructing...", len(missing))
    
    total_sessions = 0
    total_messages = 0
    
    for session_id, json_file in missing:
        success, msg_count = reconstruct_session(session_id, json_file)
        if success:
            total_sessions += 1
            total_messages += msg_count
            logger.debug("Reconstructed %s (%d messages)", session_id, msg_count)
    
    # Optimize FTS5 index
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("FTS5 optimization failed: %s", e)
    
    if total_sessions > 0:
        logger.info(
            "Session reconstruction complete: %d sessions, %d messages",
            total_sessions, total_messages
        )
    
    return total_sessions, total_messages


def register(ctx):
    """
    Plugin registration entry point.
    
    Called by Hermes plugin system on startup. Registers the on_session_start
    hook to automatically verify and reconstruct missing sessions.
    """
    # Register the verification function as a hook
    ctx.register_hook("on_session_start", _on_session_start)
    
    # Also register as a tool for manual use
    ctx.register_tool(
        name="verify_session_index",
        toolset="maintenance",
        schema={
            "name": "verify_session_index",
            "description": (
                "Verify and reconstruct missing session index. "
                "Checks for sessions saved as JSON but not indexed in SQLite, "
                "and reconstructs them automatically. Returns count of fixed sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        handler=_handle_verify_tool,
        emoji="🔍",
    )
    
    logger.debug("Session index verification plugin registered")


def _on_session_start(**kwargs):
    """
    Hook callback executed at the start of each Hermes session.
    
    Silently verifies and reconstructs any missing sessions.
    Only logs if there were sessions to fix.
    """
    try:
        sessions_fixed, messages_fixed = verify_and_reconstruct()
        
        if sessions_fixed > 0:
            # Use agent's output mechanism if available
            agent = kwargs.get('agent')
            if agent and hasattr(agent, 'console'):
                agent.console.print(
                    f"[dim]🔍 Session index: reconstructed {sessions_fixed} sessions "
                    f"({messages_fixed} messages)[/dim]"
                )
    except Exception as e:
        # Fail silently - don't block session start
        logger.debug("Session verification failed: %s", e)


def _handle_verify_tool(args: dict, **kwargs) -> str:
    """
    Tool handler for manual verification.
    
    Returns:
        JSON string with results
    """
    import json as jsonlib
    
    try:
        sessions_fixed, messages_fixed = verify_and_reconstruct()
        
        return jsonlib.dumps({
            "success": True,
            "sessions_reconstructed": sessions_fixed,
            "messages_inserted": messages_fixed,
            "message": (
                f"Reconstructed {sessions_fixed} sessions with {messages_fixed} messages. "
                "All sessions are now searchable via session_search."
                if sessions_fixed > 0
                else "All sessions are already properly indexed."
            )
        })
    except Exception as e:
        return jsonlib.dumps({
            "success": False,
            "error": str(e)
        })
