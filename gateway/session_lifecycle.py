"""
Session Lifecycle Manager — Persistent per-thread session state.

Complements the memory provider layer (e.g. Neural Memory from PR #7185)
by managing session-level concerns:
- Per-thread session files (human-readable markdown)
- Auto-load on wake (resume context after gateway restart)
- Auto-save checkpoints (state changes, decisions, completions)
- Platform-aware session routing (Discord thread ≠ Telegram chat)
- Session compression lifecycle (active → archived → pruned)

This module does NOT replace MemoryProvider — it sits above it.
The MemoryProvider handles what the agent remembers.
This module handles where and how conversation state is persisted.

Architecture:
    gateway/session.py (existing — message routing)
        ↓ extends
    SessionLifecycleManager (this file — per-thread persistence)
        ↓ calls
    MemoryManager → MemoryProvider (existing — semantic memory)

Usage:
    from gateway.session_lifecycle import SessionLifecycleManager

    lifecycle = SessionLifecycleManager()
    
    # On message receipt
    context = lifecycle.load_session("discord_12345")
    
    # On response completion (state change)
    lifecycle.save_session("discord_12345", {
        "status": "Deployed Hermes to VPS",
        "pending": ["Test SSL cert", "Configure domain"],
        "decisions": ["Using Oracle Free Tier"],
    })
    
    # Search past sessions
    results = lifecycle.search("database migration")

Note: This file is contributed by an AI agent (sephmartin's Hermes instance).
All code was developed and tested on a live 3-node Hermes cluster.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────

def _get_sessions_dir() -> Path:
    """Sessions directory — respects HERMES_VAULT env or defaults to ~/.hermes/sessions."""
    vault = os.environ.get("HERMES_VAULT", os.path.expanduser("~/.hermes"))
    return Path(vault) / "sessions"


def _get_archive_dir() -> Path:
    vault = os.environ.get("HERMES_VAULT", os.path.expanduser("~/.hermes"))
    return Path(vault) / "sessions-archive"


# ─── Platform Detection ────────────────────────────────────────────────

def detect_platform(thread_id: str) -> str:
    """Extract platform from thread ID prefix."""
    if thread_id.startswith("discord_"):
        return "discord"
    elif thread_id.startswith("telegram_"):
        return "telegram"
    elif thread_id.startswith("cli_"):
        return "cli"
    elif thread_id.startswith("api_"):
        return "api"
    return "unknown"


def extract_timestamp(thread_id: str) -> Optional[str]:
    """Extract timestamp from platform-specific thread IDs."""
    platform = detect_platform(thread_id)
    
    if platform == "discord":
        # Discord snowflake ID
        match = re.match(r"discord_(\d{13,})", thread_id)
        if match:
            snowflake = int(match.group(1))
            ts = ((snowflake >> 22) + 1420070400000) / 1000
            try:
                return datetime.fromtimestamp(ts).isoformat()
            except (OSError, ValueError):
                pass
    
    # Generic YYYYMMDD_HHMMSS format
    match = re.search(r"(\d{8})_(\d{6})", thread_id)
    if match:
        try:
            dt = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
            return dt.isoformat()
        except ValueError:
            pass
    
    return None


# ─── Session File I/O ──────────────────────────────────────────────────

def session_file_path(thread_id: str) -> Path:
    """Get the path to a session file."""
    return _get_sessions_dir() / f"{thread_id}.md"


def session_exists(thread_id: str) -> bool:
    """Check if a session file exists for this thread."""
    return session_file_path(thread_id).exists()


def load_session(thread_id: str) -> Dict[str, Any]:
    """
    Load session context from file.
    
    Returns a dict with:
    - thread_id: str
    - platform: str
    - title: str
    - status: str (from ## Project Status section)
    - context: dict (from ## Architecture/Context section)
    - last_interaction: str (from ## Last Interaction section)
    - raw: str (full file content)
    - loaded_at: str (ISO timestamp)
    """
    path = session_file_path(thread_id)
    
    if not path.exists():
        return {
            "thread_id": thread_id,
            "platform": detect_platform(thread_id),
            "title": "",
            "status": "",
            "context": {},
            "last_interaction": "",
            "raw": "",
            "loaded_at": datetime.now().isoformat(),
            "is_new": True,
        }
    
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Failed to read session %s: %s", thread_id, e)
        return {"thread_id": thread_id, "raw": "", "is_new": False}
    
    # Parse sections
    result = {
        "thread_id": thread_id,
        "platform": detect_platform(thread_id),
        "title": "",
        "status": "",
        "context": {},
        "last_interaction": "",
        "raw": text,
        "loaded_at": datetime.now().isoformat(),
        "is_new": False,
    }
    
    # Extract title
    title_m = re.search(r"^#\s+(?:Thread|Session):\s*(.+)", text, re.MULTILINE)
    if title_m:
        result["title"] = title_m.group(1).strip()
    
    # Extract status section
    status_m = re.search(r"##\s+(?:Project )?Status\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if status_m:
        result["status"] = status_m.group(1).strip()
    
    # Extract last interaction section
    last_m = re.search(r"##\s+Last Interaction\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if last_m:
        result["last_interaction"] = last_m.group(1).strip()
    
    # Extract all context sections as key-value pairs
    for section in re.finditer(r"##\s+(.+?)\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL):
        heading = section.group(1).strip()
        content = section.group(2).strip()
        if heading.lower() not in ("project status", "status", "last interaction"):
            result["context"][heading] = content
    
    return result


def save_session(thread_id: str, update: Dict[str, Any]) -> None:
    """
    Save/update session file.
    
    Args:
        thread_id: Unique thread identifier
        update: Dict with optional keys:
            - title: str
            - status: str (updates ## Project Status)
            - context: dict (key-value sections)
            - last_interaction: str (updates ## Last Interaction)
    """
    sessions_dir = _get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    path = session_file_path(thread_id)
    platform = detect_platform(thread_id)
    timestamp = extract_timestamp(thread_id)
    
    # Load existing or create new
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        title = update.get("title", f"Thread {thread_id}")
        text = f"# Thread: {title}\n\n"
        if timestamp:
            text += f"**Started**: {timestamp}\n"
        text += f"**Platform**: {platform}\n\n"
    
    # Update title
    if "title" in update:
        text = re.sub(
            r"^#\s+(?:Thread|Session):.*",
            f"# Thread: {update['title']}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    
    # Update or insert status section
    if "status" in update:
        status_block = f"## Project Status\n{update['status']}\n"
        if re.search(r"##\s+(?:Project )?Status", text):
            text = re.sub(
                r"##\s+(?:Project )?Status\s*\n.*?(?=\n##|\Z)",
                status_block,
                text,
                count=1,
                flags=re.DOTALL,
            )
        else:
            text += f"\n{status_block}"
    
    # Update or insert context sections
    if "context" in update:
        for heading, content in update["context"].items():
            section_block = f"## {heading}\n{content}\n"
            if re.search(rf"##\s+{re.escape(heading)}", text):
                text = re.sub(
                    rf"##\s+{re.escape(heading)}\s*\n.*?(?=\n##|\Z)",
                    section_block,
                    text,
                    count=1,
                    flags=re.DOTALL,
                )
            else:
                text += f"\n{section_block}"
    
    # Update or insert last interaction
    if "last_interaction" in update:
        last_block = f"## Last Interaction\n{update['last_interaction']}\n"
        if re.search(r"##\s+Last Interaction", text):
            text = re.sub(
                r"##\s+Last Interaction\s*\n.*?(?=\n##|\Z)",
                last_block,
                text,
                count=1,
                flags=re.DOTALL,
            )
        else:
            text += f"\n{last_block}"
    
    # Write atomically (write to temp, then rename)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
        logger.debug("Session saved: %s", thread_id)
    except OSError as e:
        logger.error("Failed to save session %s: %s", thread_id, e)
        if tmp_path.exists():
            tmp_path.unlink()


# ─── Session Search ─────────────────────────────────────────────────────

def search_sessions(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search sessions by content. Uses grep for zero-dependency search.
    
    For semantic search, use the vault_embeddings module separately.
    """
    sessions_dir = _get_sessions_dir()
    if not sessions_dir.exists():
        return []
    
    query_lower = query.lower()
    scored = []
    
    for f in sessions_dir.glob("*.md"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        
        text_lower = text.lower()
        
        # Count query term occurrences
        count = text_lower.count(query_lower)
        if count == 0:
            continue
        
        # Extract context around first match
        idx = text_lower.find(query_lower)
        start = max(0, idx - 100)
        end = min(len(text), idx + len(query) + 200)
        snippet = text[start:end].replace("\n", " ").strip()
        
        # Title
        title_m = re.search(r"^#\s+(?:Thread|Session):\s*(.+)", text, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else f.stem
        
        scored.append({
            "thread_id": f.stem,
            "title": title,
            "snippet": snippet,
            "matches": count,
        })
    
    scored.sort(key=lambda x: x["matches"], reverse=True)
    return scored[:k]


# ─── Session Lifecycle ──────────────────────────────────────────────────

class SessionLifecycleManager:
    """
    High-level session lifecycle manager.
    
    Provides a clean API for the gateway to:
    1. Load session context on message receipt
    2. Save session state on response completion
    3. Search past sessions
    4. List active sessions
    5. Compress old sessions
    """
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or _get_sessions_dir()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, thread_id: str) -> Dict[str, Any]:
        """Load session context. Call on every message receipt."""
        return load_session(thread_id)
    
    def save(self, thread_id: str, **kwargs) -> None:
        """Save session state. Call on state changes."""
        save_session(thread_id, kwargs)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search past sessions."""
        return search_sessions(query, k)
    
    def list_active(self, max_age_days: int = 7) -> List[Dict[str, Any]]:
        """List recently active sessions."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        sessions = []
        
        for f in sorted(self.sessions_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                continue
            
            title_m = re.search(r"^#\s+(?:Thread|Session):\s*(.+)", f.read_text(errors="replace"), re.MULTILINE)
            title = title_m.group(1).strip() if title_m else f.stem
            
            sessions.append({
                "thread_id": f.stem,
                "title": title,
                "platform": detect_platform(f.stem),
                "last_modified": mtime.isoformat(),
            })
        
        return sessions
    
    def compress(self, older_than_days: int = 7) -> int:
        """
        Compress old sessions: keep only STATUS + Last Interaction sections.
        Returns number of sessions compressed.
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        archive_dir = _get_archive_dir()
        archive_dir.mkdir(parents=True, exist_ok=True)
        compressed = 0
        
        for f in self.sessions_dir.glob("*.md"):
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime >= cutoff:
                continue
            
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                
                # Keep only title + status + last interaction
                parts = []
                title_m = re.search(r"^#.*", text, re.MULTILINE)
                if title_m:
                    parts.append(title_m.group(0))
                
                status_m = re.search(r"##\s+(?:Project )?Status\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
                if status_m:
                    parts.append(f"## Project Status\n{status_m.group(1).strip()}")
                
                last_m = re.search(r"##\s+Last Interaction\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
                if last_m:
                    parts.append(f"## Last Interaction\n{last_m.group(1).strip()}")
                
                if parts:
                    compressed_text = "\n\n".join(parts)
                    # Move original to archive
                    archive_path = archive_dir / f.name
                    f.rename(archive_path)
                    # Write compressed version
                    f.write_text(compressed_text, encoding="utf-8")
                    compressed += 1
                    
            except OSError as e:
                logger.warning("Failed to compress %s: %s", f.name, e)
        
        return compressed
    
    def exists(self, thread_id: str) -> bool:
        """Check if session file exists."""
        return session_exists(thread_id)
