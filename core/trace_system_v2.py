"""
Trace System V2 - Enhanced call chain tracing for Hermes Agent

Features:
- Three-level indexing: session_id -> trace_id -> tool_call_id
- Intelligent compression: Filter tool_result to keep important content
- Real-time streaming + async storage: Stream during execution, store asynchronously
- Session boundary detection: Detect boundaries during context compression

Storage:
- Primary: SQLite database (~/.hermes/trace/trace_v2.db)
- Fallback: JSON Lines file (~/.hermes/trace/traces_v2.jsonl)
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Callable, AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Constants
TRACE_DIR = Path.home() / ".hermes" / "trace"
TRACE_DB_PATH = TRACE_DIR / "trace_v2.db"
TRACE_JSONL_PATH = TRACE_DIR / "traces_v2.jsonl"

# Configuration cache
_config_cache: Optional[Dict[str, Any]] = None
_config_cache_time: float = 0
CONFIG_CACHE_TTL = 5.0  # seconds


class EventType(str, Enum):
    """Trace event types."""
    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_BOUNDARY = "session_boundary"
    
    # LLM events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    
    # Tool events
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    
    # Compression events
    COMPRESSION_START = "compression_start"
    COMPRESSION_COMPLETE = "compression_complete"
    COMPRESSION_BOUNDARY = "compression_boundary"
    
    # Gateway events
    GATEWAY_MESSAGE_RECEIVED = "gateway:message_received"
    GATEWAY_RESPONSE_SENT = "gateway:response_sent"
    
    # Custom events
    CUSTOM = "custom"


class EventPriority(str, Enum):
    """Event priority levels for filtering."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TraceEvent:
    """Lightweight trace event with minimal fields."""
    # Core identifiers (three-level index)
    session_id: str  # Level 1: Session identifier
    trace_id: str    # Level 2: Trace/request identifier
    tool_call_id: Optional[str] = None  # Level 3: Tool call identifier
    
    # Event metadata
    event_type: str = EventType.CUSTOM
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))
    priority: str = EventPriority.NORMAL
    
    # Timing
    duration_ms: Optional[float] = None
    
    # Content (intelligently compressed)
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None  # Serialized, truncated
    tool_result: Optional[str] = None  # Compressed
    error: Optional[str] = None
    
    # Context
    model: Optional[str] = None
    message_count: Optional[int] = None
    response_preview: Optional[str] = None
    
    # Extra metadata (flexible)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Short IDs for display
    trace_id_short: str = field(init=False)
    _tool_call_id_short: Optional[str] = field(init=False, repr=False)
    
    def __post_init__(self):
        """Generate short IDs after initialization."""
        self.trace_id_short = self.trace_id[:8] if self.trace_id else ""
        self._tool_call_id_short = None
    
    @property
    def tool_call_id_short(self) -> Optional[str]:
        """Get short tool call ID, generating if needed."""
        if self.tool_call_id and not self._tool_call_id_short:
            self._tool_call_id_short = self.tool_call_id[:8]
        return self._tool_call_id_short
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Remove None values to save space
        return {k: v for k, v in data.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceEvent':
        """Create from dictionary."""
        # Filter out unknown fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TraceSession:
    """Represents a trace session with metadata."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    trace_ids: Set[str] = field(default_factory=set)
    event_count: int = 0
    tool_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[Dict[str, str]] = field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentCompressor:
    """Intelligently compress tool results to keep important content."""
    
    # Patterns that indicate important content
    IMPORTANT_PATTERNS = [
        r'error|Error|ERROR',
        r'warning|Warning|WARNING',
        r'failed|Failed|FAILED',
        r'success|Success|SUCCESS',
        r'created|Created|CREATED',
        r'deleted|Deleted|DELETED',
        r'updated|Updated|UPDATED',
        r'http[s]?://',  # URLs
        r'/[\w/.-]+',  # File paths
        r'\d+\.\d+\.\d+\.\d+',  # IP addresses
        r'[a-f0-9]{32,}',  # Hashes
    ]
    
    # Patterns that indicate less important content
    NOISE_PATTERNS = [
        r'^\s*$',  # Empty lines
        r'^[\s\-_=]+$',  # Separator lines
        r'^\d{2}:\d{2}:\d{2}',  # Timestamps
        r'Debug:|DEBUG:',  # Debug messages
        r'Trace:|TRACE:',  # Trace messages
    ]
    
    @classmethod
    def compress_tool_result(
        cls,
        result: str,
        max_length: int = 2048,
        preserve_important: bool = True
    ) -> str:
        """
        Compress tool result while preserving important content.
        
        Args:
            result: Original tool result
            max_length: Maximum length after compression
            preserve_important: Whether to prioritize important patterns
            
        Returns:
            Compressed result
        """
        if not result:
            return ""
        
        if len(result) <= max_length:
            return result
        
        lines = result.split('\n')
        
        if preserve_important:
            # Separate important and regular lines
            important_lines = []
            regular_lines = []
            
            import re
            for line in lines:
                is_important = any(
                    re.search(pattern, line, re.IGNORECASE)
                    for pattern in cls.IMPORTANT_PATTERNS
                )
                is_noise = any(
                    re.search(pattern, line, re.IGNORECASE)
                    for pattern in cls.NOISE_PATTERNS
                )
                
                if is_important:
                    important_lines.append(line)
                elif not is_noise:
                    regular_lines.append(line)
            
            # Start with important lines
            compressed_lines = important_lines.copy()
            current_length = sum(len(line) for line in compressed_lines)
            
            # Add regular lines until we hit the limit
            for line in regular_lines:
                if current_length + len(line) + 1 > max_length - 100:  # Leave room for truncation marker
                    break
                compressed_lines.append(line)
                current_length += len(line) + 1
        else:
            # Simple truncation
            compressed_lines = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > max_length - 100:
                    break
                compressed_lines.append(line)
                current_length += len(line) + 1
        
        result = '\n'.join(compressed_lines)
        
        if len(result) < len('\n'.join(lines)):
            result += "\n\n[... truncated ...]"
        
        return result[:max_length]
    
    @classmethod
    def compress_tool_args(cls, args: Any, max_length: int = 1000) -> str:
        """Compress tool arguments."""
        if args is None:
            return ""
        
        if isinstance(args, str):
            serialized = args
        else:
            try:
                serialized = json.dumps(args, ensure_ascii=False)
            except Exception:
                serialized = str(args)
        
        if len(serialized) <= max_length:
            return serialized
        
        return serialized[:max_length - 20] + "...[truncated]"


class SessionBoundaryDetector:
    """Detect session boundaries during context compression."""
    
    def __init__(self):
        self._last_compression_time: Optional[float] = None
        self._message_count_at_last_boundary: int = 0
        self._current_message_count: int = 0
        
    def update_message_count(self, count: int):
        """Update current message count."""
        self._current_message_count = count
    
    def detect_boundary(
        self,
        event_type: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Detect if this event represents a session boundary.
        
        Boundaries are detected at:
        1. Compression events
        2. Significant time gaps (> 30 minutes)
        3. Large message count changes (> 50 messages since last boundary)
        """
        extra = extra or {}
        
        # Compression boundary
        if event_type in (EventType.COMPRESSION_START, EventType.COMPRESSION_COMPLETE):
            self._last_compression_time = time.time()
            self._message_count_at_last_boundary = self._current_message_count
            return True
        
        # Time gap boundary
        if self._last_compression_time:
            time_gap = time.time() - self._last_compression_time
            if time_gap > 1800:  # 30 minutes
                self._last_compression_time = time.time()
                self._message_count_at_last_boundary = self._current_message_count
                return True
        
        # Message count boundary
        message_delta = self._current_message_count - self._message_count_at_last_boundary
        if message_delta > 50:
            self._message_count_at_last_boundary = self._current_message_count
            return True
        
        return False


class TraceStreamContext:
    """Real-time streaming context for trace events."""
    
    def __init__(self, session_id: str, trace_id: str):
        self.session_id = session_id
        self.trace_id = trace_id
        self._events: asyncio.Queue[TraceEvent] = asyncio.Queue()
        self._subscribers: List[Callable[[TraceEvent], None]] = []
        self._closed = False
    
    async def emit(self, event: TraceEvent):
        """Emit an event to the stream."""
        if self._closed:
            return
        
        # Set identifiers if not already set
        if not event.session_id:
            event.session_id = self.session_id
        if not event.trace_id:
            event.trace_id = self.trace_id
        
        await self._events.put(event)
        
        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Error in trace subscriber callback: {e}")
    
    def subscribe(self, callback: Callable[[TraceEvent], None]):
        """Subscribe to events."""
        self._subscribers.append(callback)
    
    async def stream(self) -> AsyncIterator[TraceEvent]:
        """Stream events as they arrive."""
        while not self._closed or not self._events.empty():
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
    
    def close(self):
        """Close the stream."""
        self._closed = True


class AsyncStorageManager:
    """Manages asynchronous storage of trace events."""
    
    def __init__(self, max_batch_size: int = 100, flush_interval: float = 5.0):
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval
        self._batch: List[TraceEvent] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._flush_timer: Optional[threading.Timer] = None
        self._running = False
        
        # Try to import database manager
        self._db_manager = None
        try:
            from .trace_manager_v2 import TraceManagerV2
            self._db_manager = TraceManagerV2()
        except ImportError:
            logger.debug("Database manager not available, using JSONL fallback")
    
    def start(self):
        """Start the async storage manager."""
        self._running = True
        self._schedule_flush()
    
    def stop(self):
        """Stop the async storage manager."""
        self._running = False
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush()
    
    def add_event(self, event: TraceEvent):
        """Add an event for async storage."""
        with self._lock:
            self._batch.append(event)
            
            if len(self._batch) >= self.max_batch_size:
                self._executor.submit(self._flush)
    
    def _schedule_flush(self):
        """Schedule periodic flush."""
        if not self._running:
            return
        
        self._flush_timer = threading.Timer(self.flush_interval, self._scheduled_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def _scheduled_flush(self):
        """Called by timer to flush batch."""
        self._flush()
        self._schedule_flush()
    
    def _flush(self):
        """Flush batch to storage."""
        with self._lock:
            if not self._batch:
                return
            
            batch = self._batch.copy()
            self._batch.clear()
        
        try:
            if self._db_manager:
                self._db_manager.store_events_batch(batch)
            else:
                self._store_jsonl_batch(batch)
        except Exception as e:
            logger.error(f"Failed to store trace batch: {e}")
            # Try to store to JSONL as fallback
            try:
                self._store_jsonl_batch(batch)
            except Exception as e2:
                logger.error(f"Fallback JSONL storage also failed: {e2}")
    
    def _store_jsonl_batch(self, batch: List[TraceEvent]):
        """Store batch to JSONL file."""
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(TRACE_JSONL_PATH, "a", encoding="utf-8") as f:
            for event in batch:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


class TraceSystemV2:
    """
    Enhanced Trace System V2 for Hermes Agent.
    
    Features:
    - Three-level indexing: session_id -> trace_id -> tool_call_id
    - Intelligent compression of tool results
    - Real-time streaming with async storage
    - Session boundary detection
    """
    
    _instance: Optional['TraceSystemV2'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the trace system."""
        if self._initialized:
            return
        
        self._initialized = True
        self._config = self._load_config()
        self._enabled = self._config.get("enabled", True)
        self._sample_rate = self._config.get("sample_rate", 1.0)
        
        # Components
        self._compressor = IntelligentCompressor()
        self._boundary_detector = SessionBoundaryDetector()
        self._storage_manager = AsyncStorageManager()
        
        # Active sessions and streams
        self._sessions: Dict[str, TraceSession] = {}
        self._streams: Dict[str, TraceStreamContext] = {}
        
        # Thread safety
        self._session_lock = threading.Lock()
        
        # Start storage manager
        if self._enabled:
            self._storage_manager.start()
        
        logger.info("TraceSystemV2 initialized (enabled=%s, sample_rate=%.2f)", 
                    self._enabled, self._sample_rate)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load trace configuration."""
        global _config_cache, _config_cache_time
        
        now = time.time()
        if _config_cache is not None and (now - _config_cache_time) < CONFIG_CACHE_TTL:
            return _config_cache
        
        config_path = Path.home() / ".hermes" / "config.yaml"
        config = {"enabled": True, "sample_rate": 1.0}
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                trace_cfg = cfg.get("trace", {}) or {}
                config["enabled"] = trace_cfg.get("enabled", True)
                config["sample_rate"] = float(trace_cfg.get("sample_rate", 1.0))
            except Exception as e:
                logger.debug("Failed to load trace config: %s", e)
        
        _config_cache = config
        _config_cache_time = now
        return config
    
    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        if not self._enabled:
            return False
        
        # Sampling check
        if self._sample_rate < 1.0:
            return random.random() < self._sample_rate
        
        return True
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new trace session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        if not self.enabled:
            return session_id or uuid.uuid4().hex[:12]
        
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]
        
        with self._session_lock:
            session = TraceSession(
                session_id=session_id,
                started_at=datetime.now().isoformat(timespec="milliseconds"),
                metadata=metadata or {}
            )
            self._sessions[session_id] = session
        
        # Emit session start event
        self.record_event(TraceEvent(
            session_id=session_id,
            trace_id=session_id,
            event_type=EventType.SESSION_START,
            extra={"metadata": metadata} if metadata else {}
        ))
        
        return session_id
    
    def end_session(self, session_id: str, status: str = "completed"):
        """End a trace session."""
        if not self.enabled:
            return
        
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.ended_at = datetime.now().isoformat(timespec="milliseconds")
                session.status = status
        
        # Emit session end event
        self.record_event(TraceEvent(
            session_id=session_id,
            trace_id=session_id,
            event_type=EventType.SESSION_END,
            extra={"status": status}
        ))
    
    def get_session(self, session_id: str) -> Optional[TraceSession]:
        """Get a trace session."""
        with self._session_lock:
            return self._sessions.get(session_id)
    
    def create_stream(
        self,
        session_id: str,
        trace_id: str
    ) -> TraceStreamContext:
        """Create a streaming context for real-time events."""
        stream_key = f"{session_id}:{trace_id}"
        
        with self._session_lock:
            if stream_key not in self._streams:
                self._streams[stream_key] = TraceStreamContext(session_id, trace_id)
            
            return self._streams[stream_key]
    
    def get_stream(
        self,
        session_id: str,
        trace_id: str
    ) -> Optional[TraceStreamContext]:
        """Get an existing streaming context."""
        stream_key = f"{session_id}:{trace_id}"
        return self._streams.get(stream_key)
    
    def record_event(self, event: TraceEvent) -> None:
        """
        Record a trace event.
        
        The event is:
        1. Stored in the session
        2. Emitted to any active streams
        3. Added to async storage queue
        """
        if not self.enabled:
            return
        
        # Update session
        with self._session_lock:
            session = self._sessions.get(event.session_id)
            if session:
                session.event_count += 1
                session.trace_ids.add(event.trace_id)
                
                if event.tool_name:
                    session.tool_calls[event.tool_name] += 1
                
                if event.error:
                    session.errors.append({
                        "timestamp": event.timestamp,
                        "tool_name": event.tool_name or "unknown",
                        "error": event.error
                    })
        
        # Detect session boundary
        if self._boundary_detector.detect_boundary(event.event_type, event.extra):
            event.extra["_boundary_detected"] = True
        
        # Emit to stream if exists
        stream_key = f"{event.session_id}:{event.trace_id}"
        stream = self._streams.get(stream_key)
        if stream:
            # Schedule async emission
            asyncio.create_task(stream.emit(event))
        
        # Add to async storage
        self._storage_manager.add_event(event)
    
    def record_tool_start(
        self,
        session_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Any
    ) -> None:
        """Record tool start event."""
        compressed_args = self._compressor.compress_tool_args(tool_args)
        
        event = TraceEvent(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            event_type=EventType.TOOL_START,
            tool_name=tool_name,
            tool_args=compressed_args
        )
        
        self.record_event(event)
    
    def record_tool_complete(
        self,
        session_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Any,
        tool_result: str,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Record tool complete event with compressed result."""
        compressed_args = self._compressor.compress_tool_args(tool_args)
        compressed_result = self._compressor.compress_tool_result(tool_result)
        
        event = TraceEvent(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            event_type=EventType.TOOL_COMPLETE if not error else EventType.TOOL_ERROR,
            tool_name=tool_name,
            tool_args=compressed_args,
            tool_result=compressed_result,
            duration_ms=duration_ms,
            error=error
        )
        
        self.record_event(event)
    
    def record_llm_request(
        self,
        session_id: str,
        trace_id: str,
        model: str,
        message_count: int
    ) -> None:
        """Record LLM request event."""
        event = TraceEvent(
            session_id=session_id,
            trace_id=trace_id,
            event_type=EventType.LLM_REQUEST,
            model=model,
            message_count=message_count
        )
        
        self.record_event(event)
    
    def record_llm_response(
        self,
        session_id: str,
        trace_id: str,
        model: str,
        response_preview: str,
        duration_ms: float
    ) -> None:
        """Record LLM response event."""
        event = TraceEvent(
            session_id=session_id,
            trace_id=trace_id,
            event_type=EventType.LLM_RESPONSE,
            model=model,
            response_preview=response_preview[:500] if response_preview else "",
            duration_ms=duration_ms
        )
        
        self.record_event(event)
    
    def record_compression(
        self,
        session_id: str,
        trace_id: str,
        original_size: int,
        compressed_size: int,
        duration_ms: float
    ) -> None:
        """Record compression event."""
        event = TraceEvent(
            session_id=session_id,
            trace_id=trace_id,
            event_type=EventType.COMPRESSION_COMPLETE,
            duration_ms=duration_ms,
            extra={
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0
            }
        )
        
        self.record_event(event)
    
    def update_message_count(self, count: int):
        """Update message count for boundary detection."""
        self._boundary_detector.update_message_count(count)
    
    def list_sessions(
        self,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[TraceSession]:
        """
        List trace sessions.
        
        Args:
            limit: Maximum number of sessions to return
            status: Filter by status
            
        Returns:
            List of sessions
        """
        with self._session_lock:
            sessions = list(self._sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions[:limit]
    
    def get_session_events(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TraceEvent]:
        """
        Get events for a session with optional filtering.
        
        Args:
            session_id: Session ID
            trace_id: Optional trace ID filter
            tool_call_id: Optional tool call ID filter
            limit: Maximum events to return
            
        Returns:
            List of events
        """
        # Try database first
        if self._storage_manager._db_manager:
            try:
                return self._storage_manager._db_manager.get_events(
                    session_id=session_id,
                    trace_id=trace_id,
                    tool_call_id=tool_call_id,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Database query failed, falling back to JSONL: {e}")
        
        # Fallback to JSONL
        return self._get_events_from_jsonl(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            limit=limit
        )
    
    def _get_events_from_jsonl(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TraceEvent]:
        """Get events from JSONL file."""
        if not TRACE_JSONL_PATH.exists():
            return []
        
        events = []
        
        with open(TRACE_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = TraceEvent.from_dict(data)
                    
                    # Apply filters
                    if session_id and event.session_id != session_id:
                        continue
                    if trace_id and event.trace_id != trace_id:
                        continue
                    if tool_call_id and event.tool_call_id != tool_call_id:
                        continue
                    
                    events.append(event)
                    
                    if len(events) >= limit:
                        break
                except (json.JSONDecodeError, Exception):
                    continue
        
        return events
    
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a trace session.
        
        Returns:
            Analysis with timing, errors, tool calls
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        
        events = self.get_session_events(session_id, limit=1000)
        
        # Calculate timing
        total_duration_ms = 0
        if events:
            try:
                first_ts = datetime.fromisoformat(events[0].timestamp)
                last_ts = datetime.fromisoformat(events[-1].timestamp)
                total_duration_ms = (last_ts - first_ts).total_seconds() * 1000
            except Exception:
                pass
        
        # Analyze tool calls
        tool_analysis = {}
        for tool_name, count in session.tool_calls.items():
            tool_events = [e for e in events if e.tool_name == tool_name]
            durations = [e.duration_ms for e in tool_events if e.duration_ms is not None]
            
            tool_analysis[tool_name] = {
                "count": count,
                "avg_duration_ms": sum(durations) / len(durations) if durations else None,
                "max_duration_ms": max(durations) if durations else None,
                "errors": len([e for e in tool_events if e.error])
            }
        
        return {
            "session_id": session_id,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "status": session.status,
            "total_duration_ms": round(total_duration_ms, 2),
            "event_count": session.event_count,
            "trace_count": len(session.trace_ids),
            "tool_analysis": tool_analysis,
            "errors": session.errors,
            "metadata": session.metadata
        }
    
    def shutdown(self):
        """Shutdown the trace system."""
        self._storage_manager.stop()
        
        # End all active sessions
        with self._session_lock:
            for session_id in list(self._sessions.keys()):
                self.end_session(session_id, status="shutdown")
        
        # Close all streams
        for stream in self._streams.values():
            stream.close()


# Convenience functions for backward compatibility
_global_system: Optional[TraceSystemV2] = None


def get_trace_system() -> TraceSystemV2:
    """Get the global trace system instance."""
    global _global_system
    if _global_system is None:
        _global_system = TraceSystemV2()
    return _global_system


def record_event(
    session_id: str,
    trace_id: str,
    event_type: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[Any] = None,
    tool_result: Optional[str] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a trace event (backward compatible interface).
    
    This function provides compatibility with the original trace system.
    """
    system = get_trace_system()
    
    event = TraceEvent(
        session_id=session_id,
        trace_id=trace_id,
        event_type=event_type,
        tool_name=tool_name,
        tool_args=system._compressor.compress_tool_args(tool_args) if tool_args else None,
        tool_result=system._compressor.compress_tool_result(tool_result) if tool_result else None,
        duration_ms=duration_ms,
        error=error,
        extra=extra or {}
    )
    
    system.record_event(event)


def create_trace_callbacks(
    session_id: str,
    trace_id: str
) -> tuple:
    """
    Create trace callbacks for tool execution.
    
    Returns:
        (tool_start_callback, tool_complete_callback)
    """
    system = get_trace_system()
    
    def start_cb(tool_call_id: str, name: str, args: Any) -> None:
        system.record_tool_start(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_args=args
        )
    
    def complete_cb(tool_call_id: str, name: str, args: Any, result: str) -> None:
        system.record_tool_complete(
            session_id=session_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_args=args,
            tool_result=result
        )
    
    return start_cb, complete_cb


# Cleanup on module unload
import atexit
atexit.register(lambda: get_trace_system().shutdown() if _global_system else None)