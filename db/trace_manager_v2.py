"""
Trace Manager V2 - Database operations for enhanced trace system.

Provides efficient batch operations and three-level indexing for trace data.
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator
from collections import defaultdict

logger = logging.getLogger(__name__)

# Database path
TRACE_DB_PATH = Path.home() / ".hermes" / "trace" / "trace_v2.db"
SCHEMA_PATH = Path(__file__).parent / "trace_schema_v2.sql"


class TraceManagerV2:
    """
    Database manager for trace system V2.
    
    Features:
    - Three-level indexing: session_id -> trace_id -> tool_call_id
    - Batch operations for performance
    - Connection pooling with thread safety
    - Automatic schema initialization
    """
    
    _instance: Optional['TraceManagerV2'] = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[Path] = None):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the trace manager."""
        if self._initialized:
            return
        
        self._initialized = True
        self._db_path = db_path or TRACE_DB_PATH
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized_db = False
        
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
            
            # Initialize schema if needed
            if not self._initialized_db:
                with self._init_lock:
                    if not self._initialized_db:
                        self._init_schema()
                        self._initialized_db = True
        
        return self._local.connection
    
    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_schema(self):
        """Initialize database schema."""
        if not SCHEMA_PATH.exists():
            logger.warning(f"Schema file not found: {SCHEMA_PATH}")
            return
        
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        conn = self._get_connection()
        conn.executescript(schema_sql)
        logger.info("Trace database schema initialized")
    
    # =========================================================================
    # Session Operations
    # =========================================================================
    
    def create_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new trace session."""
        with self._transaction() as cursor:
            cursor.execute("""
                INSERT OR IGNORE INTO trace_sessions (session_id, started_at, metadata)
                VALUES (?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(timespec="milliseconds"),
                json.dumps(metadata) if metadata else None
            ))
        return session_id
    
    def end_session(self, session_id: str, status: str = "completed"):
        """End a trace session."""
        with self._transaction() as cursor:
            cursor.execute("""
                UPDATE trace_sessions 
                SET status = ?, ended_at = ?, updated_at = ?
                WHERE session_id = ?
            """, (
                status,
                datetime.now().isoformat(timespec="milliseconds"),
                datetime.now().isoformat(timespec="milliseconds"),
                session_id
            ))
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM trace_sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def list_sessions(
        self,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List sessions."""
        conn = self._get_connection()
        
        if status:
            cursor = conn.execute("""
                SELECT * FROM trace_sessions 
                WHERE status = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (status, limit))
        else:
            cursor = conn.execute("""
                SELECT * FROM trace_sessions 
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Trace Operations
    # =========================================================================
    
    def create_trace(
        self,
        session_id: str,
        trace_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new trace within a session."""
        with self._transaction() as cursor:
            # Ensure session exists
            cursor.execute("""
                INSERT OR IGNORE INTO trace_sessions (session_id, started_at)
                VALUES (?, ?)
            """, (session_id, datetime.now().isoformat(timespec="milliseconds")))
            
            # Create trace
            cursor.execute("""
                INSERT OR IGNORE INTO trace_traces (trace_id, session_id, started_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                trace_id,
                session_id,
                datetime.now().isoformat(timespec="milliseconds"),
                json.dumps(metadata) if metadata else None
            ))
            
            # Update session trace count
            cursor.execute("""
                UPDATE trace_sessions 
                SET trace_count = trace_count + 1, updated_at = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(timespec="milliseconds"), session_id))
        
        return trace_id
    
    def end_trace(
        self,
        session_id: str,
        trace_id: str,
        status: str = "completed"
    ):
        """End a trace."""
        with self._transaction() as cursor:
            cursor.execute("""
                UPDATE trace_traces 
                SET status = ?, ended_at = ?, updated_at = ?
                WHERE session_id = ? AND trace_id = ?
            """, (
                status,
                datetime.now().isoformat(timespec="milliseconds"),
                datetime.now().isoformat(timespec="milliseconds"),
                session_id,
                trace_id
            ))
    
    def get_trace(
        self,
        session_id: str,
        trace_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM trace_traces 
            WHERE session_id = ? AND trace_id = ?
        """, (session_id, trace_id))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def list_traces(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List traces in a session."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM trace_traces 
            WHERE session_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (session_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Tool Call Operations
    # =========================================================================
    
    def create_tool_call(
        self,
        session_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Optional[str] = None
    ) -> str:
        """Create a new tool call record."""
        with self._transaction() as cursor:
            # Ensure trace exists
            cursor.execute("""
                INSERT OR IGNORE INTO trace_traces (trace_id, session_id, started_at)
                VALUES (?, ?, ?)
            """, (trace_id, session_id, datetime.now().isoformat(timespec="milliseconds")))
            
            # Create tool call
            cursor.execute("""
                INSERT OR REPLACE INTO trace_tool_calls 
                (tool_call_id, trace_id, session_id, tool_name, tool_args, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'running')
            """, (
                tool_call_id,
                trace_id,
                session_id,
                tool_name,
                tool_args,
                datetime.now().isoformat(timespec="milliseconds")
            ))
        
        return tool_call_id
    
    def complete_tool_call(
        self,
        session_id: str,
        trace_id: str,
        tool_call_id: str,
        tool_result: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Complete a tool call."""
        status = "error" if error else "completed"
        completed_at = datetime.now().isoformat(timespec="milliseconds")
        
        with self._transaction() as cursor:
            cursor.execute("""
                UPDATE trace_tool_calls 
                SET tool_result = ?, duration_ms = ?, error = ?, 
                    completed_at = ?, status = ?
                WHERE session_id = ? AND trace_id = ? AND tool_call_id = ?
            """, (
                tool_result,
                duration_ms,
                error,
                completed_at,
                status,
                session_id,
                trace_id,
                tool_call_id
            ))
    
    def get_tool_call(
        self,
        session_id: str,
        trace_id: str,
        tool_call_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get tool call by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM trace_tool_calls 
            WHERE session_id = ? AND trace_id = ? AND tool_call_id = ?
        """, (session_id, trace_id, tool_call_id))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def list_tool_calls(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List tool calls with optional filters."""
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_tool_calls WHERE session_id = ?"
        params: List[Any] = [session_id]
        
        if trace_id:
            query += " AND trace_id = ?"
            params.append(trace_id)
        
        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)
        
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Event Operations
    # =========================================================================
    
    def store_event(self, event: Dict[str, Any]) -> int:
        """Store a single trace event."""
        with self._transaction() as cursor:
            cursor.execute("""
                INSERT INTO trace_events 
                (session_id, trace_id, tool_call_id, event_type, timestamp, priority,
                 duration_ms, tool_name, tool_args, tool_result, error, model,
                 message_count, response_preview, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get("session_id"),
                event.get("trace_id"),
                event.get("tool_call_id"),
                event.get("event_type", "custom"),
                event.get("timestamp", datetime.now().isoformat(timespec="milliseconds")),
                event.get("priority", "normal"),
                event.get("duration_ms"),
                event.get("tool_name"),
                event.get("tool_args"),
                event.get("tool_result"),
                event.get("error"),
                event.get("model"),
                event.get("message_count"),
                event.get("response_preview"),
                json.dumps(event.get("extra")) if event.get("extra") else None
            ))
            return cursor.lastrowid
    
    def store_events_batch(self, events: List[Dict[str, Any]]) -> List[int]:
        """Store multiple events in a batch."""
        event_ids = []
        
        with self._transaction() as cursor:
            for event in events:
                cursor.execute("""
                    INSERT INTO trace_events 
                    (session_id, trace_id, tool_call_id, event_type, timestamp, priority,
                     duration_ms, tool_name, tool_args, tool_result, error, model,
                     message_count, response_preview, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.get("session_id"),
                    event.get("trace_id"),
                    event.get("tool_call_id"),
                    event.get("event_type", "custom"),
                    event.get("timestamp", datetime.now().isoformat(timespec="milliseconds")),
                    event.get("priority", "normal"),
                    event.get("duration_ms"),
                    event.get("tool_name"),
                    event.get("tool_args"),
                    event.get("tool_result"),
                    event.get("error"),
                    event.get("model"),
                    event.get("message_count"),
                    event.get("response_preview"),
                    json.dumps(event.get("extra")) if event.get("extra") else None
                ))
                event_ids.append(cursor.lastrowid)
        
        return event_ids
    
    def get_events(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get events with three-level indexing.
        
        Args:
            session_id: Level 1 filter
            trace_id: Level 2 filter (requires session_id)
            tool_call_id: Level 3 filter (requires session_id)
            event_type: Event type filter
            limit: Maximum events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_events WHERE 1=1"
        params: List[Any] = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if trace_id:
            query += " AND trace_id = ?"
            params.append(trace_id)
        
        if tool_call_id:
            query += " AND tool_call_id = ?"
            params.append(tool_call_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_events_by_time_range(
        self,
        start_time: str,
        end_time: str,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get events within a time range."""
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_events WHERE timestamp BETWEEN ? AND ?"
        params: List[Any] = [start_time, end_time]
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Error Pattern Operations
    # =========================================================================
    
    def record_error_pattern(
        self,
        error_message: str,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record or update an error pattern."""
        import hashlib
        error_hash = hashlib.md5(f"{tool_name}:{error_message}".encode()).hexdigest()
        
        with self._transaction() as cursor:
            cursor.execute("""
                INSERT INTO trace_error_patterns 
                (error_hash, error_message, tool_name, first_seen, last_seen, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(error_hash) DO UPDATE SET
                    last_seen = ?,
                    occurrence_count = occurrence_count + 1,
                    metadata = ?
            """, (
                error_hash,
                error_message,
                tool_name,
                datetime.now().isoformat(timespec="milliseconds"),
                datetime.now().isoformat(timespec="milliseconds"),
                json.dumps(metadata) if metadata else None,
                datetime.now().isoformat(timespec="milliseconds"),
                json.dumps(metadata) if metadata else None
            ))
    
    def get_error_patterns(
        self,
        tool_name: Optional[str] = None,
        min_occurrences: int = 1,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get error patterns."""
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_error_patterns WHERE occurrence_count >= ?"
        params: List[Any] = [min_occurrences]
        
        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)
        
        query += " ORDER BY occurrence_count DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Feedback Operations
    # =========================================================================
    
    def record_feedback(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        feedback_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Record user feedback."""
        with self._transaction() as cursor:
            cursor.execute("""
                INSERT INTO trace_feedbacks 
                (session_id, trace_id, tool_call_id, rating, feedback_text, 
                 feedback_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                trace_id,
                tool_call_id,
                rating,
                feedback_text,
                feedback_type,
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid
    
    def get_feedbacks(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        feedback_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get feedbacks with optional filters."""
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_feedbacks WHERE 1=1"
        params: List[Any] = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if trace_id:
            query += " AND trace_id = ?"
            params.append(trace_id)
        
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Statistics Operations
    # =========================================================================
    
    def get_tool_stats(
        self,
        tool_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tool statistics."""
        conn = self._get_connection()
        
        query = "SELECT * FROM trace_tool_stats WHERE 1=1"
        params: List[Any] = []
        
        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)
        
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        
        query += " ORDER BY date DESC, call_count DESC"
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        conn = self._get_connection()
        
        # Get session
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        
        # Get trace count
        cursor = conn.execute("""
            SELECT COUNT(*) as trace_count FROM trace_traces WHERE session_id = ?
        """, (session_id,))
        trace_count = cursor.fetchone()["trace_count"]
        
        # Get tool call stats
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms
            FROM trace_tool_calls 
            WHERE session_id = ?
        """, (session_id,))
        tool_stats = dict(cursor.fetchone())
        
        # Get tool breakdown
        cursor = conn.execute("""
            SELECT tool_name, COUNT(*) as count, AVG(duration_ms) as avg_duration
            FROM trace_tool_calls 
            WHERE session_id = ?
            GROUP BY tool_name
            ORDER BY count DESC
        """, (session_id,))
        tool_breakdown = [dict(row) for row in cursor.fetchall()]
        
        # Get errors
        cursor = conn.execute("""
            SELECT tool_name, error, timestamp
            FROM trace_events 
            WHERE session_id = ? AND error IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """, (session_id,))
        recent_errors = [dict(row) for row in cursor.fetchall()]
        
        return {
            "session": session,
            "trace_count": trace_count,
            "tool_stats": tool_stats,
            "tool_breakdown": tool_breakdown,
            "recent_errors": recent_errors
        }
    
    # =========================================================================
    # Cleanup Operations
    # =========================================================================
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up sessions older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._transaction() as cursor:
            # Get session IDs to delete
            cursor.execute("""
                SELECT session_id FROM trace_sessions 
                WHERE started_at < ? AND status != 'active'
            """, (cutoff_date,))
            session_ids = [row[0] for row in cursor.fetchall()]
            
            if not session_ids:
                return 0
            
            # Delete sessions (cascades to traces, tool_calls, events)
            placeholders = ','.join(['?'] * len(session_ids))
            cursor.execute(f"""
                DELETE FROM trace_sessions 
                WHERE session_id IN ({placeholders})
            """, session_ids)
            
            return len(session_ids)
    
    def vacuum(self):
        """Optimize database."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        conn.execute("PRAGMA optimize")
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Convenience functions
_manager: Optional[TraceManagerV2] = None


def get_trace_manager() -> TraceManagerV2:
    """Get the global trace manager instance."""
    global _manager
    if _manager is None:
        _manager = TraceManagerV2()
    return _manager