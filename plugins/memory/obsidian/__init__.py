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

try:
    from hermes_cli.config import cfg_get, load_config
except Exception:  # pragma: no cover - Hermes imports can vary in tests
    cfg_get = None
    load_config = None

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


def _cfg_value(*paths: str, default: str = "") -> str:
    """Return the first non-empty Hermes config value from candidate paths."""
    if not load_config or not cfg_get:
        return default
    try:
        config = load_config()
    except Exception:
        return default
    for path in paths:
        parts = tuple(part for part in path.split(".") if part)
        if not parts:
            continue
        try:
            value = cfg_get(config, *parts)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _vault_path_value() -> str:
    """Resolve the Obsidian vault path from env or Hermes config.

    Chris's live config stores this under plugins.obsidian.vault_path; older
    drafts used root obsidian.vault_path. Support both so the plugin survives
    config migration and profile-specific HERMES_HOME usage.
    """
    return _env_value(
        "OBSIDIAN_VAULT_PATH",
        _cfg_value("obsidian.vault_path", "plugins.obsidian.vault_path"),
    )


def _memory_folder_value() -> str:
    return _env_value(
        "OBSIDIAN_MEMORY_FOLDER",
        _cfg_value("obsidian.memory_folder", "plugins.obsidian.memory_folder", default="Memory"),
    )


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
        self._last_user_content: str = ""

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "obsidian"

    def is_available(self) -> bool:
        vault = _vault_path_value()
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
        vault = _vault_path_value()
        memory_folder = _memory_folder_value() or "Memory"
        self._vault_path = Path(vault)
        self._memory_dir = self._vault_path / memory_folder
        self._daily_dir = self._memory_dir / "daily"
        self._sessions_dir = self._memory_dir / "sessions"
        self._agents_dir = self._memory_dir / "agents"
        self._session_id = session_id
        self._session_start = datetime.now(timezone.utc)
        self._raw_checkpoint_path = self._sessions_dir / f"{session_id}-raw.md"

        # Ensure dirs exist
        for d in (self._memory_dir, self._daily_dir, self._sessions_dir, self._agents_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Ensure mirror files exist
        for name in ("MEMORY.md", "USER.md"):
            vault_file = self._memory_dir / name
            if not vault_file.exists():
                vault_file.write_text("", encoding="utf-8")

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
        """Count turns and buffer the user message for the current turn."""
        self._turn_count = turn_number
        if message:
            self._buffer_message("user", message)
            self._last_user_content = message

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to the raw checkpoint buffer.

        MemoryManager calls this after the assistant response, so this is the
        reliable point to include both sides of the turn and checkpoint every
        fifth completed user turn.
        """
        if session_id and session_id != self._session_id:
            self.on_session_switch(session_id)

        if user_content and user_content != self._last_user_content:
            self._buffer_message("user", user_content)
            self._last_user_content = user_content
        if assistant_content:
            self._buffer_message("assistant", assistant_content)

        if self._turn_count - self._last_checkpoint_turn >= RAW_CHECKPOINT_EVERY_TURNS:
            if self._write_raw_checkpoint():
                self._last_checkpoint_turn = self._turn_count
                self.summarize_session()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Start writing checkpoints under the new Hermes session id."""
        if not new_session_id or new_session_id == self._session_id:
            return
        if self._message_buffer:
            self._write_raw_checkpoint(force=True)
        self._session_id = new_session_id
        self._session_start = datetime.now(timezone.utc)
        self._turn_count = 0
        self._last_checkpoint_turn = 0
        self._last_user_content = ""
        self._message_buffer = []
        if self._sessions_dir:
            self._raw_checkpoint_path = self._sessions_dir / f"{new_session_id}-raw.md"

    def _buffer_message(self, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        self._message_buffer.append({"role": role, "content": text[:2000]})
        if len(self._message_buffer) > 40:
            self._message_buffer = self._message_buffer[-40:]

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize raw checkpoint before context compression."""
        # Summarize the current session
        summary_path = self.summarize_session()
        if summary_path:
            return f"Session summarized: {summary_path}"
        return ""

    def _extract_requests_with_timestamps(self, raw_content: str) -> list:
        """Extract user requests from raw checkpoints with timestamps."""
        requests = []
        # Pattern: --- Checkpoint @ turn N --- followed by [user] messages
        checkpoint_pattern = r'--- Checkpoint @ turn (\d+) ---\n((?:\n|.)*?)(?=\n--- Checkpoint @ turn \d+ ---|\Z)'
        checkpoints = re.findall(checkpoint_pattern, raw_content)

        for turn_num, block in checkpoints:
            # Find user messages in this block — must be exact [user] tag at start of line
            user_lines = re.findall(r'^[ ]*\[user\] (.*)', block, re.MULTILINE)
            for line in user_lines:
                line = line.strip()
                # Skip lines that look like pasted chat history (start with [date] or [HH:MM])
                if re.match(r'^\[\d{2}[/-]\d{2}', line):
                    continue
                if line and not line.startswith('<') and len(line) > 10:
                    # Check if it's a request (imperative or question)
                    if any(line.lower().startswith(w) for w in ['can you', 'could you', 'please', 'want', 'need', 'fix', 'build', 'create', 'update', 'delete', 'add', 'remove', 'uninstall', 'install', 'restart', 'stop', 'start', 'do ', 'make ', 'help ', 'try ', 'let\'s ', 'let us']):
                        requests.append({
                            'turn': int(turn_num),
                            'text': line,
                            'timestamp': self._session_start.isoformat() if self._session_start else datetime.now(timezone.utc).isoformat()
                        })

        # Deduplicate similar requests
        seen = set()
        unique = []
        for r in requests:
            key = r['text'][:80].lower()
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    # ------------------------------------------------------------------
    # Raw checkpoint writer
    # ------------------------------------------------------------------

    def _write_raw_checkpoint(self, force: bool = False) -> bool:
        """Append buffered messages to the raw checkpoint file."""
        if not self._raw_checkpoint_path or not self._message_buffer:
            return False

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
            with open(self._raw_checkpoint_path, "a", encoding="utf-8") as f:
                f.write("".join(lines))
            self._message_buffer = []
            logger.info("Raw checkpoint written: %s", self._raw_checkpoint_path.name)
            return True
        except OSError as e:
            logger.debug("Raw checkpoint failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Summarize command (triggered by user or schedule)
    # ------------------------------------------------------------------

    def summarize_session(self) -> Optional[Path]:
        """Read raw checkpoint, write clean summary, delete raw. Returns summary path."""
        if not self._raw_checkpoint_path or not self._raw_checkpoint_path.exists():
            return None

        try:
            raw_content = self._raw_checkpoint_path.read_text(encoding="utf-8")
            if not raw_content.strip():
                self._raw_checkpoint_path.unlink(missing_ok=True)
                return None

            summary = self._condense_raw_to_summary(raw_content)
            if not summary.strip():
                self._raw_checkpoint_path.unlink(missing_ok=True)
                return None

            summary_path = self._sessions_dir / f"{self._session_id}.md"
            if summary_path.exists():
                existing = summary_path.read_text(encoding="utf-8")
                combined = existing.rstrip() + "\n\n---\n\n" + summary
                summary_path.write_text(combined, encoding="utf-8")
            else:
                summary_path.write_text(summary, encoding="utf-8")

            # Delete raw after successful summary
            self._raw_checkpoint_path.unlink(missing_ok=True)
            logger.info("Session summarized: %s", summary_path.name)
            return summary_path

        except OSError as e:
            logger.debug("Summarize failed: %s", e)
            return None

    def _condense_raw_to_summary(self, raw: str) -> str:
        """Convert raw checkpoint text into a clean summary.
        
        Captures:
        - User requests (with document detection)
        - Completed actions (commits, file changes, builds)
        - Decisions made
        - File paths created/modified
        """
        lines = raw.splitlines()
        user_requests: List[str] = []
        completed_actions: List[str] = []
        decisions: List[str] = []
        files_touched: set = set()
        projects: set = set()
        documents_received: List[str] = []
        
        in_document = False
        document_title = ""
        
        for line in lines:
            line = line.strip()
            
            # Detect document sends
            if "[The user sent a text document:" in line:
                in_document = True
                # Extract document name
                match = re.search(r"'([^']+\.md)'", line)
                if match:
                    document_title = match.group(1)
                    documents_received.append(document_title)
                continue
            
            if in_document:
                if line.startswith("[user]") or line.startswith("[assistant]"):
                    in_document = False
                    document_title = ""
                elif line and not line.startswith("["):
                    # Skip document content lines in raw, we'll note the doc name only
                    continue
            
            if line.startswith("[user]"):
                text = line[6:].strip()
                if len(text) > 10 and text.lower() not in ("ok", "hello", "hi", "thanks", "ty", "yes", "no"):
                    # Skip document content markers
                    if "content has been included below" not in text and "file is also saved at" not in text:
                        user_requests.append(text[:200])
                    # Detect project names
                    for proj in ("Empire", "Standby", "OpenClaw", "Claw3D", "SEO", "Legal", "Hermes"):
                        if proj.lower() in text.lower():
                            projects.add(proj)
                            
            elif line.startswith("[assistant]"):
                text = line[11:].strip()
                
                # Detect completed actions
                if any(marker in text.lower() for marker in ("committed", "pushed", "deployed", "built", "created", "done", "finished", "saved to", "written to")):
                    completed_actions.append(text[:200])
                    # Extract file paths
                    paths = re.findall(r'`([^`]+\.(?:py|md|yaml|json|sh))`', text)
                    files_touched.update(paths)
                    
                # Detect decisions
                if any(marker in text.lower() for marker in ("decided", "decision", "will ", "going to", "let's", "plan:", "approve", "recommend")):
                    decisions.append(text[:200])
                    
                # Detect file paths in code blocks or backticks
                paths = re.findall(r'`([^`]+\.(?:py|md|yaml|json|sh))`', text)
                files_touched.update(paths)

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
            
        if documents_received:
            parts.append("**Documents Received:**")
            for doc in documents_received:
                parts.append(f"- {doc}")
            parts.append("")

        # Get timestamped requests using the new extractor
        timestamped_requests = self._extract_requests_with_timestamps(raw)
        
        if timestamped_requests:
            parts.append("**Requests:**")
            for r in timestamped_requests[-8:]:  # Last 8
                ts = r.get('timestamp', '')
                # Format timestamp as HH:MM if available
                time_str = ''
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M")
                    except:
                        pass
                if time_str:
                    parts.append(f"- [{time_str}] {r['text'][:180]}")
                else:
                    parts.append(f"- {r['text'][:200]}")
            parts.append("")
        elif user_requests:
            # Fallback to old format if timestamped extraction fails
            parts.append("**Requests:**")
            for r in user_requests[-8:]:
                parts.append(f"- {r}")
            parts.append("")

        if completed_actions:
            parts.append("**Completed:**")
            for a in completed_actions[-6:]:  # Last 6
                parts.append(f"- {a}")
            parts.append("")
            
        if files_touched:
            parts.append("**Files:**")
            for f in sorted(files_touched)[:10]:  # Top 10
                parts.append(f"- `{f}`")
            parts.append("")

        if decisions:
            parts.append("**Decisions:**")
            for d in decisions[-5:]:  # Last 5
                parts.append(f"- {d}")
            parts.append("")

        # Add status note
        if completed_actions:
            parts.append("**Status:** Work completed and committed.")
        else:
            parts.append("**Status:** In progress.")
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
        """Return latest session summary + today's daily context.
        
        Uses clean headings and source paths so the model understands
        this is recalled memory context, not new user input.
        """
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
                content = latest.read_text(encoding="utf-8").strip()
                if content:
                    result_parts.append(f"## Recalled Memory: Latest Clean Session Summary")
                    result_parts.append(f"Source: {latest}")
                    result_parts.append("")
                    result_parts.append(content[:500])
                    result_parts.append("")
        except OSError:
            pass

        # 2. Load today's daily note — ONLY if it has a clean # Daily Summary section
        if self._daily_dir:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily = self._daily_dir / f"{today}.md"
            if daily.exists():
                try:
                    content = daily.read_text(encoding="utf-8")
                    # Find the last # Daily Summary section
                    summary_marker = "# Daily Summary"
                    last_idx = content.rfind(summary_marker)
                    if last_idx != -1:
                        # Extract from the marker to end
                        summary_section = content[last_idx:].strip()
                        if summary_section:
                            result_parts.append(f"## Recalled Memory: Latest Daily Summary")
                            result_parts.append(f"Source: {daily}")
                            result_parts.append("")
                            # Take first ~300 chars of the summary section
                            result_parts.append(summary_section[:300])
                            result_parts.append("")
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
                current = vault_file.read_text(encoding="utf-8") if vault_file.exists() else ""
                entry = f"{content}\n"
                vault_file.write_text((current + entry).strip() + "\n", encoding="utf-8")

            elif action == "replace":
                old_text = (metadata or {}).get("old_text", "")
                current = vault_file.read_text(encoding="utf-8") if vault_file.exists() else ""
                if old_text and old_text in current:
                    vault_file.write_text(current.replace(old_text, content), encoding="utf-8")
                else:
                    vault_file.write_text((current + f"\n{content}").strip() + "\n", encoding="utf-8")

            elif action == "remove":
                old_text = (metadata or {}).get("old_text", "")
                current = vault_file.read_text(encoding="utf-8") if vault_file.exists() else ""
                if old_text and old_text in current:
                    vault_file.write_text(current.replace(old_text, "").strip() + "\n", encoding="utf-8")

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
        summary_path = self.summarize_session()
        
        # If summarize_session produced a clean summary, use that
        if summary_path and summary_path.exists():
            try:
                clean_summary = summary_path.read_text(encoding="utf-8").strip()
                if clean_summary:
                    # Extract just the key sections from the clean summary
                    entry_parts = [f"\n## Session {self._session_id[:8]} @ {datetime.now(timezone.utc).strftime('%H:%M')}"]
                    
                    # Parse the clean summary for requests, completed, decisions
                    in_requests = False
                    in_completed = False
                    in_decisions = False
                    
                    for line in clean_summary.splitlines():
                        stripped = line.strip()
                        
                        if stripped.startswith("**Requests:**"):
                            in_requests = True
                            in_completed = False
                            in_decisions = False
                            entry_parts.append("\n**Requests:**")
                            continue
                        elif stripped.startswith("**Completed:**"):
                            in_requests = False
                            in_completed = True
                            in_decisions = False
                            entry_parts.append("\n**Completed:**")
                            continue
                        elif stripped.startswith("**Decisions:**"):
                            in_requests = False
                            in_completed = False
                            in_decisions = True
                            entry_parts.append("\n**Decisions:**")
                            continue
                        elif stripped.startswith("**"):
                            # End of relevant sections
                            in_requests = False
                            in_completed = False
                            in_decisions = False
                            continue
                        
                        if stripped.startswith("- ") and (in_requests or in_completed or in_decisions):
                            entry_parts.append(stripped)
                    
                    if len(entry_parts) > 1:  # More than just the header
                        entry = "\n".join(entry_parts) + "\n"
                        with open(daily, "a", encoding="utf-8") as f:
                            f.write(entry)
                        return  # Done — clean summary written
            except OSError:
                pass

        # Fallback: only use the LAST user message (not full history) to avoid bloat
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                content = m["content"].strip()
                if len(content) > 10 and content.lower() not in ("ok", "hello", "hi", "thanks", "ty", "yes", "no"):
                    last_user_msg = content[:120]
                    break

        if last_user_msg:
            now = datetime.now(timezone.utc)
            time_str = now.strftime("%H:%M")
            entry = f"\n## Session {self._session_id[:8]} @ {time_str}\n\n**Requests:**\n- [{time_str}] {last_user_msg}\n"
            try:
                with open(daily, "a", encoding="utf-8") as f:
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
                content = s.read_text(encoding="utf-8").strip()
                if content:
                    # Extract key lines with timestamps
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("**"):
                            lines.append(line)
                        elif line.startswith("- "):
                            # Preserve timestamp if present
                            lines.append(line)
                    lines.append("")

            # Append to daily note
            with open(daily, "a", encoding="utf-8") as f:
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

    def write_agent_memory(self, agent_name: str, task: str, result: str, *,
                           child_session_id: str = "") -> Optional[Path]:
        """Write subagent completion memory under Memory/agents/{date}."""
        if not self._agents_dir:
            return None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        agent_dir = self._agents_dir / today
        agent_dir.mkdir(parents=True, exist_ok=True)

        safe_agent = re.sub(r"[^A-Za-z0-9_.-]+", "-", agent_name or "unknown").strip("-") or "unknown"
        log_file = agent_dir / f"{safe_agent}.md"
        timestamp = datetime.now(timezone.utc).isoformat()
        session_line = f"**Child session:** {child_session_id}\n" if child_session_id else ""

        entry = f"""
## {timestamp}
**Task:** {(task or '')[:200]}
{session_line}**Result:** {(result or '')[:1000]}
---
"""
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.debug("Agent log written: %s", log_file.name)
            return log_file
        except OSError as e:
            logger.debug("Agent log failed: %s", e)
            return None

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Log subagent work to Memory/agents/{date}/{agent_name}.md"""
        agent_name = kwargs.get("agent_name", "unknown")
        self.write_agent_memory(agent_name, task, result, child_session_id=child_session_id)

    def on_agent_complete(self, task: str, result: str, *,
                          child_session_id: str = "", **kwargs) -> None:
        """Compatibility hook for agent completion events."""
        self.on_delegation(task, result, child_session_id=child_session_id, **kwargs)

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
                    content = f.read_text(encoding="utf-8").lower()
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
