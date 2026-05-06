"""
Obsidian memory plugin — continuous offload system.

Writes memory entries as Markdown notes in the vault's Memory/ folder, creates
daily session logs, and surfaces vault context on session start.

NEW: Continuous memory offload
  - Raw checkpoints every N turns → Memory/sessions/{id}-raw.md
  - Summarize command → reads raw, writes clean summary, deletes raw
  - Prefetch on startup → loads latest clean summary
  - Dream cycle → nightly condenses all summaries to daily/{date}.md

Config via environment variables (set in ~/.hermes/.env):
  OBSIDIAN_VAULT_PATH — Path to the Obsidian vault root directory
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
RAW_CHECKPOINT_EVERY_TURNS = 5   # Write raw checkpoint every N user turns
SUMMARY_MAX_AGE_TURNS = 10       # Force summarize if raw is this old


def _env_value(name: str, default: str = "") -> str:
    """Read from process env, then from ~/.hermes/.env for CLI entrypoints."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    env_file = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))) / ".env"
    if not env_file.exists():
        return default
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key.strip() == name:
                return raw.strip().strip('"').strip("'")
    except OSError:
        return default
    return default


class ObsidianMemoryProvider(MemoryProvider):
    """Mirrors Hermes memory to an Obsidian vault with continuous offload."""

    def __init__(self) -> None:
        self._vault_path: Optional[Path] = None
        self._memory_dir: Optional[Path] = None
        self._daily_dir: Optional[Path] = None
        self._sessions_dir: Optional[Path] = None
        self._agents_dir: Optional[Path] = None
        self._session_id: str = ""
        self._session_start: Optional[datetime] = None
        # Turn counters
        self._turn_count: int = 0
        self._last_checkpoint_turn: int = 0
        self._raw_checkpoint_path: Optional[Path] = None
        # In-memory buffer for current session
        self._message_buffer: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "obsidian"

    def is_available(self) -> bool:
        vault = _env_value("OBSIDIAN_VAULT_PATH")
        if not vault:
            logger.debug("OBSIDIAN_VAULT_PATH not set")
            return False
        path = Path(vault)
        if not path.is_dir():
            logger.debug("Obsidian vault not found at %s", path)
            return False
        if not os.access(path, os.W_OK):
            logger.debug("Obsidian vault not writable at %s", path)
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        vault = _env_value("OBSIDIAN_VAULT_PATH")
        self._vault_path = Path(vault)
        self._memory_dir = self._vault_path / "Memory"
        self._daily_dir = self._memory_dir / "daily"
        self._sessions_dir = self._memory_dir / "sessions"
        self._agents_dir = self._memory_dir / "agents"
        self._session_id = session_id
        self._session_start = datetime.now(timezone.utc)
        self._raw_checkpoint_path = self._sessions_dir / f"{session_id}-raw.md"

        # Ensure dirs exist
        for d in (self._memory_dir, self._daily_dir, self._sessions_dir, self._agents_dir):
            d.mkdir(exist_ok=True)

        # Ensure mirror files exist
        for name in ("MEMORY.md", "USER.md"):
            vault_file = self._memory_dir / name
            if not vault_file.exists():
                vault_file.write_text("")

        logger.info(
            "Obsidian memory initialized — vault: %s, session: %s",
            self._vault_path, session_id,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """No additional tools — mirrors the built-in memory tool."""
        return []

    def shutdown(self) -> None:
        """Final checkpoint on shutdown, then summarize if needed."""
        if self._message_buffer and self._raw_checkpoint_path:
            self._write_raw_checkpoint(force=True)
        logger.debug("Obsidian memory provider shut down")

    # ------------------------------------------------------------------
    # Turn-based checkpointing
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Count turns and write raw checkpoints periodically."""
        self._turn_count = turn_number

        # Every N turns, write a raw checkpoint
        if turn_number - self._last_checkpoint_turn >= RAW_CHECKPOINT_EVERY_TURNS:
            self._write_raw_checkpoint()
            self._last_checkpoint_turn = turn_number

    def on_turn_end(self, turn_number: int, messages: List[Dict[str, Any]], **kwargs) -> None:
        """Buffer messages for checkpointing."""
        self._message_buffer.extend(messages)
        # Keep buffer from growing unbounded — only last 20 messages
        if len(self._message_buffer) > 20:
            self._message_buffer = self._message_buffer[-20:]

    # ------------------------------------------------------------------
    # Raw checkpoint writer
    # ------------------------------------------------------------------

    def _write_raw_checkpoint(self, force: bool = False) -> None:
        """Append buffered messages to the raw checkpoint file."""
        if not self._raw_checkpoint_path or not self._message_buffer:
            return

        lines = [f"\n--- Checkpoint @ turn {self._turn_count} ---\n"]
        for m in self._message_buffer:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, str):
                # Truncate very long content
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                lines.append(f"[{role}] {content}\n")
            elif isinstance(content, list):
                # Tool results — just note them
                lines.append(f"[{role}] <tool results>\n")

        try:
            with open(self._raw_checkpoint_path, "a") as f:
                f.write("".join(lines))
            logger.debug("Raw checkpoint written: %s", self._raw_checkpoint_path.name)
        except OSError as e:
            logger.debug("Raw checkpoint failed: %s", e)

    # ------------------------------------------------------------------
    # Summarize command (triggered by user or schedule)
    # ------------------------------------------------------------------

    def summarize_session(self) -> Optional[Path]:
        """Read raw checkpoint, write clean summary, delete raw. Returns summary path."""
        if not self._raw_checkpoint_path or not self._raw_checkpoint_path.exists():
            return None

        try:
            raw_content = self._raw_checkpoint_path.read_text()
            if not raw_content.strip():
                self._raw_checkpoint_path.unlink(missing_ok=True)
                return None

            summary = self._condense_raw_to_summary(raw_content)
            if not summary.strip():
                self._raw_checkpoint_path.unlink(missing_ok=True)
                return None

            summary_path = self._sessions_dir / f"{self._session_id}.md"
            summary_path.write_text(summary)

            # Delete raw after successful summary
            self._raw_checkpoint_path.unlink(missing_ok=True)
            logger.info("Session summarized: %s", summary_path.name)
            return summary_path

        except OSError as e:
            logger.debug("Summarize failed: %s", e)
            return None

    def _condense_raw_to_summary(self, raw: str) -> str:
        """Convert raw checkpoint text into a clean summary."""
        lines = raw.splitlines()
        user_requests: List[str] = []
        assistant_actions: List[str] = []
        decisions: List[str] = []
        projects: set = set()

        for line in lines:
            line = line.strip()
            if line.startswith("[user]"):
                text = line[6:].strip()
                if len(text) > 10 and text.lower() not in ("ok", "hello", "hi", "thanks", "ty", "yes", "no"):
                    user_requests.append(text[:150])
                    # Detect project names
                    for proj in ("Empire", "Standby", "OpenClaw", "Claw3D", "SEO", "Legal"):
                        if proj.lower() in text.lower():
                            projects.add(proj)
            elif line.startswith("[assistant]"):
                text = line[11:].strip()
                if len(text) > 20:
                    assistant_actions.append(text[:150])
                # Detect decisions
                if any(marker in text.lower() for marker in ("decided", "decision", "will ", "going to", "let's", "plan:", "approve")):
                    decisions.append(text[:150])

        # Build summary
        parts = [
            f"# Session {self._session_id[:8]}",
            f"Started: {self._session_start.isoformat() if self._session_start else 'unknown'}",
            f"Turns: {self._turn_count}",
            "",
        ]

        if projects:
            parts.append(f"**Projects:** {', '.join(sorted(projects))}")
            parts.append("")

        if user_requests:
            parts.append("**Requests:**")
            for r in user_requests[-8:]:  # Last 8
                parts.append(f"- {r}")
            parts.append("")

        if decisions:
            parts.append("**Decisions:**")
            for d in decisions[-5:]:  # Last 5
                parts.append(f"- {d}")
            parts.append("")

        if assistant_actions:
            parts.append("**Actions:**")
            for a in assistant_actions[-5:]:
                parts.append(f"- {a}")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Tell the agent the vault is available."""
        if not self._vault_path:
            return ""
        return (
            "Obsidian vault is active. Memory writes are persisted to the vault. "
            "You can read vault notes with read_file or search_files by prefixing "
            f"the vault path ({self._memory_dir})."
        )

    # ------------------------------------------------------------------
    # Context prefetch — load latest session summary on startup
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return latest session summary + today's daily context."""
        if not self._sessions_dir:
            return ""

        result_parts: List[str] = []

        # 1. Load latest session summary
        try:
            summaries = sorted(
                [f for f in self._sessions_dir.iterdir() if f.suffix == ".md" and not f.name.endswith("-raw.md")],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if summaries:
                latest = summaries[0]
                content = latest.read_text().strip()
                if content:
                    # Take first ~500 chars of summary
                    result_parts.append(f"## Previous session ({latest.stem}):\n")
                    result_parts.append(content[:500] + "\n")
        except OSError:
            pass

        # 2. Load today's daily note (last part)
        if self._daily_dir:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily = self._daily_dir / f"{today}.md"
            if daily.exists():
                try:
                    content = daily.read_text().strip()
                    if content:
                        lines = content.splitlines()
                        meaningful = [l for l in lines if l.strip() and not l.startswith("-") and not l.startswith("#")]
                        if meaningful:
                            recent = "\n".join(meaningful[-8:])
                            if len(recent) > 300:
                                recent = recent[-300:]
                            result_parts.append(f"## Today's activity ({today}):\n{recent}\n")
                except OSError:
                    pass

        return "\n".join(result_parts)

    # ------------------------------------------------------------------
    # Memory write mirroring
    # ------------------------------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to vault files."""
        if not self._memory_dir:
            return

        filename = "MEMORY.md" if target == "memory" else "USER.md"
        vault_file = self._memory_dir / filename

        try:
            if action == "add":
                current = vault_file.read_text() if vault_file.exists() else ""
                entry = f"{content}\n"
                vault_file.write_text((current + entry).strip() + "\n")

            elif action == "replace":
                old_text = (metadata or {}).get("old_text", "")
                current = vault_file.read_text() if vault_file.exists() else ""
                if old_text and old_text in current:
                    vault_file.write_text(current.replace(old_text, content))
                else:
                    vault_file.write_text((current + f"\n{content}").strip() + "\n")

            elif action == "remove":
                old_text = (metadata or {}).get("old_text", "")
                current = vault_file.read_text() if vault_file.exists() else ""
                if old_text and old_text in current:
                    vault_file.write_text(current.replace(old_text, "").strip() + "\n")

        except OSError as e:
            logger.debug("Obsidian mirror write failed for %s: %s", filename, e)

    # ------------------------------------------------------------------
    # Session end — write to daily log
    # ------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Write a concise session summary to the daily note."""
        if not self._daily_dir or not self._session_start:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = self._daily_dir / f"{today}.md"

        # First, summarize any remaining raw checkpoint
        self.summarize_session()

        # Collect from messages (fallback if summarize didn't catch everything)
        actions = []
        decisions = []
        projects = set()

        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                content = m["content"].strip()
                if len(content) < 10 or content.lower() in ("ok", "hello", "hi", "thanks", "ty"):
                    continue
                actions.append(content[:120])
                for proj in ("Empire", "Standby", "OpenClaw", "Claw3D", "SEO", "Legal"):
                    if proj.lower() in content.lower():
                        projects.add(proj)

            elif m.get("role") == "assistant" and isinstance(m.get("content"), str):
                content = m["content"].strip()
                if any(marker in content.lower() for marker in ("decided", "decision", "will", "going to", "let's")):
                    decisions.append(content[:120])

        # Build entry — only if there's actual content
        entry_parts = []
        if actions or decisions or projects:
            entry_parts.append(f"\n## Session {self._session_id[:8]}")

            if projects:
                entry_parts.append(f"\n**Projects:** {', '.join(sorted(projects))}")

            if actions:
                entry_parts.append("\n**Requests:**")
                for a in actions[-5:]:
                    entry_parts.append(f"- {a}")

            if decisions:
                entry_parts.append("\n**Decisions:**")
                for d in decisions[-3:]:
                    entry_parts.append(f"- {d}")

            entry = "\n".join(entry_parts) + "\n"
            try:
                with open(daily, "a") as f:
                    f.write(entry)
            except OSError as e:
                logger.debug("Failed to write session-end daily note: %s", e)

    # ------------------------------------------------------------------
    # Dream cycle — nightly condensation
    # ------------------------------------------------------------------

    def dream_cycle(self) -> None:
        """Condense all session summaries into the daily note. Called by scheduler."""
        if not self._sessions_dir or not self._daily_dir:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = self._daily_dir / f"{today}.md"

        try:
            # Find all session summaries (both ID-based and date-based names)
            summaries = []
            for f in self._sessions_dir.iterdir():
                if f.suffix == ".md" and not f.name.endswith("-raw.md"):
                    # Check if created today OR filename matches today's date
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    mtime_date = mtime.strftime("%Y-%m-%d")
                    fname_date = self._extract_date_from_filename(f.stem)
                    
                    if mtime_date == today or fname_date == today:
                        summaries.append(f)

            if not summaries:
                return

            # Build condensed daily entry
            lines = [f"\n# Daily Summary — {today}\n"]
            for s in sorted(summaries, key=lambda f: f.stat().st_mtime):
                content = s.read_text().strip()
                if content:
                    # Extract key lines
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("**") or line.startswith("- "):
                            lines.append(line)
                    lines.append("")

            # Append to daily note
            with open(daily, "a") as f:
                f.write("\n".join(lines) + "\n")

            # Optionally: archive old summaries (keep last 7 days)
            self._archive_old_summaries()

            logger.info("Dream cycle completed — %d summaries condensed", len(summaries))

        except OSError as e:
            logger.debug("Dream cycle failed: %s", e)

    def _extract_date_from_filename(self, stem: str) -> str:
        """Extract YYYY-MM-DD from filename if present."""
        # Match patterns like 2026-05-06 or 20260506
        match = re.search(r'(\d{4})[-]?(\d{2})[-]?(\d{2})', stem)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return ""

    def on_agent_complete(self, agent_name: str, task: str, result: str) -> None:
        """Log subagent work to Memory/agents/{date}/{agent_name}.md"""
        if not self._agents_dir:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        agent_dir = self._agents_dir / today
        agent_dir.mkdir(exist_ok=True)

        log_file = agent_dir / f"{agent_name}.md"
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = f"""
## {timestamp}
**Task:** {task[:200]}
**Result:** {result[:500]}
---
"""
        try:
            with open(log_file, "a") as f:
                f.write(entry)
            logger.debug("Agent log written: %s", log_file.name)
        except OSError as e:
            logger.debug("Agent log failed: %s", e)

    def search_agent_memory(self, query: str, limit: int = 5) -> str:
        """Semantic search across all agent logs, sessions, and daily notes.
        
        Uses simple keyword + recency scoring. Returns top matches.
        """
        if not self._memory_dir:
            return ""
        
        import re
        from datetime import datetime, timezone
        
        query_terms = set(query.lower().split())
        matches = []
        
        # Search all .md files in Memory/
        for root in (self._sessions_dir, self._agents_dir, self._daily_dir):
            if not root or not root.exists():
                continue
            for f in root.rglob("*.md"):
                if f.name.endswith("-raw.md"):
                    continue
                try:
                    content = f.read_text().lower()
                    score = sum(1 for term in query_terms if term in content)
                    if score > 0:
                        # Recency boost
                        mtime = f.stat().st_mtime
                        age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400
                        recency_boost = max(0, 1 - (age_days / 30))  # Decay over 30 days
                        score += recency_boost
                        
                        # Extract relevant snippet
                        lines = content.splitlines()
                        snippet_lines = []
                        for i, line in enumerate(lines):
                            if any(term in line for term in query_terms):
                                # Get context: 1 line before, the line, 1 after
                                start = max(0, i-1)
                                end = min(len(lines), i+2)
                                snippet_lines.extend(lines[start:end])
                                snippet_lines.append("---")
                        
                        snippet = "\n".join(snippet_lines[:20])  # Limit snippet length
                        matches.append((score, f"{f.relative_to(self._memory_dir)}", snippet))
                except OSError:
                    continue
        
        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        
        if not matches:
            return f"No matches for '{query}'"
        
        results = [f"## Search: '{query}'\n"]
        for score, path, snippet in matches[:limit]:
            results.append(f"**{path}** (score: {score:.1f})")
            results.append(f"```\n{snippet[:400]}\n```")
            results.append("")
        
        return "\n".join(results)

    def _archive_old_summaries(self, keep_days: int = 7) -> None:
        """Move summaries older than keep_days to an archive folder."""
        if not self._sessions_dir:
            return

        archive_dir = self._sessions_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        cutoff = datetime.now(timezone.utc).timestamp() - (keep_days * 86400)

        for f in self._sessions_dir.iterdir():
            if f.suffix == ".md" and not f.name.endswith("-raw.md"):
                if f.stat().st_mtime < cutoff:
                    try:
                        f.rename(archive_dir / f.name)
                    except OSError:
                        pass
