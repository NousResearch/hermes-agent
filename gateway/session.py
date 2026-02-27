"""
Session management for the gateway.

Handles:
- Session context tracking (where messages come from)
- Session storage (conversations persisted to disk)
- Reset policy evaluation (when to start fresh)
- Dynamic system prompt injection (agent knows its context)
"""

import logging
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

from .config import (
    Platform,
    GatewayConfig,
    SessionResetPolicy,
    HomeChannel,
)


@dataclass
class SessionSource:
    """
    Describes where a message originated from.
    
    This information is used to:
    1. Route responses back to the right place
    2. Inject context into the system prompt
    3. Track origin for cron job delivery
    """
    platform: Platform
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"  # "dm", "group", "channel", "thread"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None  # For forum topics, Discord threads, etc.
    
    @property
    def description(self) -> str:
        """Human-readable description of the source."""
        if self.platform == Platform.LOCAL:
            return "CLI terminal"
        
        parts = []
        if self.chat_type == "dm":
            parts.append(f"DM with {self.user_name or self.user_id or 'user'}")
        elif self.chat_type == "group":
            parts.append(f"group: {self.chat_name or self.chat_id}")
        elif self.chat_type == "channel":
            parts.append(f"channel: {self.chat_name or self.chat_id}")
        else:
            parts.append(self.chat_name or self.chat_id)
        
        if self.thread_id:
            parts.append(f"thread: {self.thread_id}")
        
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_id": self.thread_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSource":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            chat_name=data.get("chat_name"),
            chat_type=data.get("chat_type", "dm"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            thread_id=data.get("thread_id"),
        )
    
    @classmethod
    def local_cli(cls) -> "SessionSource":
        """Create a source representing the local CLI."""
        return cls(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI terminal",
            chat_type="dm",
        )


@dataclass
class SessionContext:
    """
    Full context for a session, used for dynamic system prompt injection.
    
    The agent receives this information to understand:
    - Where messages are coming from
    - What platforms are available
    - Where it can deliver scheduled task outputs
    """
    source: SessionSource
    connected_platforms: List[Platform]
    home_channels: Dict[Platform, HomeChannel]
    
    # Session metadata
    session_key: str = ""
    session_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "connected_platforms": [p.value for p in self.connected_platforms],
            "home_channels": {
                p.value: hc.to_dict() for p, hc in self.home_channels.items()
            },
            "session_key": self.session_key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def build_session_context_prompt(context: SessionContext) -> str:
    """
    Build the dynamic system prompt section that tells the agent about its context.
    
    This is injected into the system prompt so the agent knows:
    - Where messages are coming from
    - What platforms are connected
    - Where it can deliver scheduled task outputs
    """
    lines = [
        "## Current Session Context",
        "",
    ]
    
    # Source info
    platform_name = context.source.platform.value.title()
    if context.source.platform == Platform.LOCAL:
        lines.append(f"**Source:** {platform_name} (the machine running this agent)")
    else:
        lines.append(f"**Source:** {platform_name} ({context.source.description})")

    # User identity (especially useful for WhatsApp where multiple people DM)
    if context.source.user_name:
        lines.append(f"**User:** {context.source.user_name}")
    elif context.source.user_id:
        lines.append(f"**User ID:** {context.source.user_id}")
    
    # Connected platforms
    platforms_list = ["local (files on this machine)"]
    for p in context.connected_platforms:
        if p != Platform.LOCAL:
            platforms_list.append(f"{p.value}: Connected ✓")
    
    lines.append(f"**Connected Platforms:** {', '.join(platforms_list)}")
    
    # Home channels
    if context.home_channels:
        lines.append("")
        lines.append("**Home Channels (default destinations):**")
        for platform, home in context.home_channels.items():
            lines.append(f"  - {platform.value}: {home.name} (ID: {home.chat_id})")
    
    # Delivery options for scheduled tasks
    lines.append("")
    lines.append("**Delivery options for scheduled tasks:**")
    
    # Origin delivery
    if context.source.platform == Platform.LOCAL:
        lines.append("- `\"origin\"` → Local output (saved to files)")
    else:
        lines.append(f"- `\"origin\"` → Back to this chat ({context.source.chat_name or context.source.chat_id})")
    
    # Local always available
    lines.append("- `\"local\"` → Save to local files only (~/.hermes/cron/output/)")
    
    # Platform home channels
    for platform, home in context.home_channels.items():
        lines.append(f"- `\"{platform.value}\"` → Home channel ({home.name})")
    
    # Note about explicit targeting
    lines.append("")
    lines.append("*For explicit targeting, use `\"platform:chat_id\"` format if the user provides a specific chat ID.*")
    
    return "\n".join(lines)


@dataclass
class SessionEntry:
    """
    Entry in the session store.
    
    Maps a session key to its current session ID and metadata.
    """
    session_key: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    
    # Origin metadata for delivery routing
    origin: Optional[SessionSource] = None
    
    # Display metadata
    display_name: Optional[str] = None
    platform: Optional[Platform] = None
    chat_type: str = "dm"
    
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Set when a session was created because the previous one expired;
    # consumed once by the message handler to inject a notice into context
    was_auto_reset: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_key": self.session_key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "display_name": self.display_name,
            "platform": self.platform.value if self.platform else None,
            "chat_type": self.chat_type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.origin:
            result["origin"] = self.origin.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionEntry":
        origin = None
        if "origin" in data and data["origin"]:
            origin = SessionSource.from_dict(data["origin"])
        
        platform = None
        if data.get("platform"):
            try:
                platform = Platform(data["platform"])
            except ValueError:
                pass
        
        return cls(
            session_key=data["session_key"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            origin=origin,
            display_name=data.get("display_name"),
            platform=platform,
            chat_type=data.get("chat_type", "dm"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


class SessionStore:
    """
    Manages session storage and retrieval.
    
    Uses SQLite (via SessionDB) for session metadata and message transcripts.
    Falls back to legacy JSONL files if SQLite is unavailable.
    """
    
    def __init__(self, sessions_dir: Path, config: GatewayConfig,
                 has_active_processes_fn=None,
                 on_auto_reset=None):
        self.sessions_dir = sessions_dir
        self.config = config
        self._entries: Dict[str, SessionEntry] = {}
        self._loaded = False
        self._has_active_processes_fn = has_active_processes_fn
        self._on_auto_reset = on_auto_reset  # callback(old_entry) before auto-reset
        
        # Initialize SQLite session database
        self._db = None
        try:
            from hermes_state import SessionDB
            self._db = SessionDB()
        except Exception as e:
            print(f"[gateway] Warning: SQLite session store unavailable, falling back to JSONL: {e}")
    
    def _ensure_loaded(self) -> None:
        """Load sessions index from disk if not already loaded."""
        if self._loaded:
            return
        
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_file = self.sessions_dir / "sessions.json"
        
        if sessions_file.exists():
            try:
                with open(sessions_file, "r") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._entries[key] = SessionEntry.from_dict(entry_data)
            except Exception as e:
                print(f"[gateway] Warning: Failed to load sessions: {e}")
        
        self._loaded = True
    
    def _save(self) -> None:
        """Save sessions index to disk (kept for session key -> ID mapping)."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_file = self.sessions_dir / "sessions.json"
        
        data = {key: entry.to_dict() for key, entry in self._entries.items()}
        with open(sessions_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_session_key(self, source: SessionSource) -> str:
        """Generate a session key from a source."""
        platform = source.platform.value

        if source.chat_type == "dm":
            # WhatsApp DMs come from different people, each needs its own session.
            # Other platforms (Telegram, Discord) have a single DM with the bot owner.
            if platform == "whatsapp" and source.chat_id:
                return f"agent:main:{platform}:dm:{source.chat_id}"
            return f"agent:main:{platform}:dm"
        else:
            return f"agent:main:{platform}:{source.chat_type}:{source.chat_id}"
    
    def _should_reset(self, entry: SessionEntry, source: SessionSource) -> bool:
        """
        Check if a session should be reset based on policy.
        
        Sessions with active background processes are never reset.
        """
        if self._has_active_processes_fn:
            session_key = self._generate_session_key(source)
            if self._has_active_processes_fn(session_key):
                return False

        policy = self.config.get_reset_policy(
            platform=source.platform,
            session_type=source.chat_type
        )
        
        if policy.mode == "none":
            return False
        
        now = datetime.now()
        
        if policy.mode in ("idle", "both"):
            idle_deadline = entry.updated_at + timedelta(minutes=policy.idle_minutes)
            if now > idle_deadline:
                return True
        
        if policy.mode in ("daily", "both"):
            today_reset = now.replace(
                hour=policy.at_hour, 
                minute=0, 
                second=0, 
                microsecond=0
            )
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)
            
            if entry.updated_at < today_reset:
                return True
        
        return False
    
    def has_any_sessions(self) -> bool:
        """Check if any sessions have ever been created (across all platforms)."""
        self._ensure_loaded()
        return len(self._entries) > 1  # >1 because the current new session is already in _entries
    
    def get_or_create_session(
        self, 
        source: SessionSource,
        force_new: bool = False
    ) -> SessionEntry:
        """
        Get an existing session or create a new one.
        
        Evaluates reset policy to determine if the existing session is stale.
        Creates a session record in SQLite when a new session starts.
        """
        self._ensure_loaded()
        
        session_key = self._generate_session_key(source)
        now = datetime.now()
        
        if session_key in self._entries and not force_new:
            entry = self._entries[session_key]
            
            if not self._should_reset(entry, source):
                entry.updated_at = now
                self._save()
                return entry
            else:
                # Session is being auto-reset — flush memories before destroying
                was_auto_reset = True
                if self._on_auto_reset:
                    try:
                        self._on_auto_reset(entry)
                    except Exception as e:
                        logger.debug("Auto-reset callback failed: %s", e)
                if self._db:
                    try:
                        self._db.end_session(entry.session_id, "session_reset")
                    except Exception as e:
                        logger.debug("Session DB operation failed: %s", e)
        else:
            was_auto_reset = False
        
        # Create new session
        session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        entry = SessionEntry(
            session_key=session_key,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            origin=source,
            display_name=source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
            was_auto_reset=was_auto_reset,
        )
        
        self._entries[session_key] = entry
        self._save()
        
        # Create session in SQLite
        if self._db:
            try:
                self._db.create_session(
                    session_id=session_id,
                    source=source.platform.value,
                    user_id=source.user_id,
                )
            except Exception as e:
                print(f"[gateway] Warning: Failed to create SQLite session: {e}")
        
        return entry
    
    def update_session(
        self, 
        session_key: str,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """Update a session's metadata after an interaction."""
        self._ensure_loaded()
        
        if session_key in self._entries:
            entry = self._entries[session_key]
            entry.updated_at = datetime.now()
            entry.input_tokens += input_tokens
            entry.output_tokens += output_tokens
            entry.total_tokens = entry.input_tokens + entry.output_tokens
            self._save()
            
            if self._db:
                try:
                    self._db.update_token_counts(
                        entry.session_id, input_tokens, output_tokens
                    )
                except Exception as e:
                    logger.debug("Session DB operation failed: %s", e)
    
    def reset_session(self, session_key: str) -> Optional[SessionEntry]:
        """Force reset a session, creating a new session ID."""
        self._ensure_loaded()
        
        if session_key not in self._entries:
            return None
        
        old_entry = self._entries[session_key]
        
        # End old session in SQLite
        if self._db:
            try:
                self._db.end_session(old_entry.session_id, "session_reset")
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)
        
        now = datetime.now()
        session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        new_entry = SessionEntry(
            session_key=session_key,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            origin=old_entry.origin,
            display_name=old_entry.display_name,
            platform=old_entry.platform,
            chat_type=old_entry.chat_type,
        )
        
        self._entries[session_key] = new_entry
        self._save()
        
        # Create new session in SQLite
        if self._db:
            try:
                self._db.create_session(
                    session_id=session_id,
                    source=old_entry.platform.value if old_entry.platform else "unknown",
                    user_id=old_entry.origin.user_id if old_entry.origin else None,
                )
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)
        
        return new_entry
    
    def list_sessions(self, active_minutes: Optional[int] = None) -> List[SessionEntry]:
        """List all sessions, optionally filtered by activity."""
        self._ensure_loaded()
        
        entries = list(self._entries.values())
        
        if active_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=active_minutes)
            entries = [e for e in entries if e.updated_at >= cutoff]
        
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        
        return entries
    
    def get_transcript_path(self, session_id: str) -> Path:
        """Get the path to a session's legacy transcript file."""
        return self.sessions_dir / f"{session_id}.jsonl"
    
    def append_to_transcript(self, session_id: str, message: Dict[str, Any]) -> None:
        """Append a message to a session's transcript (SQLite + legacy JSONL)."""
        # Write to SQLite
        if self._db:
            try:
                self._db.append_message(
                    session_id=session_id,
                    role=message.get("role", "unknown"),
                    content=message.get("content"),
                    tool_name=message.get("tool_name"),
                    tool_calls=message.get("tool_calls"),
                    tool_call_id=message.get("tool_call_id"),
                )
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)
        
        # Also write legacy JSONL (keeps existing tooling working during transition)
        transcript_path = self.get_transcript_path(session_id)
        with open(transcript_path, "a") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    def load_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages from a session's transcript."""
        # Try SQLite first
        if self._db:
            try:
                messages = self._db.get_messages_as_conversation(session_id)
                if messages:
                    return messages
            except Exception as e:
                logger.debug("Could not load messages from DB: %s", e)
        
        # Fall back to legacy JSONL
        transcript_path = self.get_transcript_path(session_id)
        
        if not transcript_path.exists():
            return []
        
        messages = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        
        return messages


def build_session_context(
    source: SessionSource,
    config: GatewayConfig,
    session_entry: Optional[SessionEntry] = None
) -> SessionContext:
    """
    Build a full session context from a source and config.
    
    This is used to inject context into the agent's system prompt.
    """
    connected = config.get_connected_platforms()
    
    home_channels = {}
    for platform in connected:
        home = config.get_home_channel(platform)
        if home:
            home_channels[platform] = home
    
    context = SessionContext(
        source=source,
        connected_platforms=connected,
        home_channels=home_channels,
    )
    
    if session_entry:
        context.session_key = session_entry.session_key
        context.session_id = session_entry.session_id
        context.created_at = session_entry.created_at
        context.updated_at = session_entry.updated_at
    
    return context
