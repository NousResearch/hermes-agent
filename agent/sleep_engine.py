#!/usr/bin/env python3
"""
Sleep Engine - Memory consolidation and cleaning system for Hermes Agent.

Implements a sleep mode that filters redundant memories based on session usage patterns.
Inspired by human sleep-memory consolidation mechanisms.

Key concepts:
1. Session importance scoring: Based on duration, message count, recency
2. Adaptive vocabulary learning: Extracts important vs unimportant words from sessions
3. Memory scoring: Evaluates memories based on learned vocabulary
4. Cleaning: Removes low-scoring memories, consolidates important ones
"""

import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Statistics for a single session."""
    session_id: str
    started_at: float
    ended_at: Optional[float]
    message_count: int
    tool_call_count: int
    title: Optional[str]
    source: str
    duration_hours: float = 0.0
    importance_score: float = 0.0


class SleepEngine:
    """Core engine for memory consolidation during sleep."""
    
    def __init__(self, memory_store: MemoryStore, db_path: Optional[Path] = None):
        """
        Initialize the sleep engine.
        
        Args:
            memory_store: The MemoryStore instance to clean
            db_path: Path to state.db (default: ~/.hermes/state.db)
        """
        self.memory_store = memory_store
        self.db_path = db_path or (get_hermes_home() / "state.db")
        
        # Vocabulary caches
        self.important_words: Dict[str, float] = {}  # word -> positive weight
        self.unimportant_words: Dict[str, float] = {}  # word -> negative weight
        
        # Configuration
        self.importance_threshold = 0.5  # Sessions with score > 0.5 are important
        self.memory_delete_threshold = 0.4  # Memories with score < 0.4 are deleted
        self.min_word_length = 2  # Minimum word length to consider
        self.max_words_to_keep = 100  # Max number of words to track
        
        # For progress reporting
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set a callback for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, stage: str, progress: float, message: str = ""):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
    
    # -------------------------------------------------------------------------
    # Session Analysis
    # -------------------------------------------------------------------------
    
    def _get_all_sessions(self) -> List[SessionStats]:
        """Retrieve all sessions from the database."""
        if not self.db_path.exists():
            logger.warning(f"Database not found at {self.db_path}")
            return []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all sessions with basic stats
            cursor.execute("""
                SELECT 
                    id, source, started_at, ended_at, message_count, 
                    tool_call_count, title
                FROM sessions 
                WHERE ended_at IS NOT NULL 
                ORDER BY started_at DESC
            """)
            
            sessions = []
            for row in cursor.fetchall():
                # Calculate duration in hours
                duration = 0.0
                if row['ended_at']:
                    duration = (row['ended_at'] - row['started_at']) / 3600.0  # hours
                
                session = SessionStats(
                    session_id=row['id'],
                    started_at=row['started_at'],
                    ended_at=row['ended_at'],
                    message_count=row['message_count'],
                    tool_call_count=row['tool_call_count'],
                    title=row['title'],
                    source=row['source'],
                    duration_hours=max(0.0, duration)
                )
                sessions.append(session)
            
            conn.close()
            logger.info(f"Retrieved {len(sessions)} sessions from database")
            return sessions
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return []
    
    def _calculate_session_importance(self, session: SessionStats) -> float:
        """
        Calculate importance score for a session (0-1).
        
        Factors:
        1. Duration: Longer sessions are more important
        2. Message density: More messages per hour = more engagement
        3. Tool usage: Tool calls indicate serious work
        4. Recency: Recent sessions are more relevant
        """
        score = 0.0
        
        # 1. Duration factor (0-0.3)
        # Sessions longer than 1 hour get max points, shorter get proportional
        duration_score = min(session.duration_hours / 1.0, 1.0) * 0.3
        score += duration_score
        
        # 2. Message density factor (0-0.3)
        if session.duration_hours > 0.1:  # At least 6 minutes
            messages_per_hour = session.message_count / session.duration_hours
            density_score = min(messages_per_hour / 20.0, 1.0) * 0.3  # 20 msgs/hour = max
            score += density_score
        
        # 3. Tool usage factor (0-0.2)
        if session.tool_call_count > 0:
            tool_score = min(session.tool_call_count / 10.0, 1.0) * 0.2  # 10 tools = max
            score += tool_score
        
        # 4. Recency factor (0-0.2)
        # Sessions from last 7 days get full points, older decay
        days_ago = (time.time() - session.started_at) / (24 * 3600)
        recency_score = max(0.0, 1.0 - (days_ago / 7.0)) * 0.2
        score += recency_score
        
        return min(1.0, max(0.0, score))
    
    def _extract_words_from_session(self, session_id: str) -> List[str]:
        """Extract words from session messages."""
        if not self.db_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get all messages for this session
            cursor.execute("""
                SELECT content FROM messages 
                WHERE session_id = ? AND content IS NOT NULL
            """, (session_id,))
            
            all_text = " ".join(row[0] for row in cursor.fetchall() if row[0])
            conn.close()
            
            # Simple word extraction: split by non-alphanumeric, filter short words
            words = re.findall(r'\b\w+\b', all_text.lower())
            words = [w for w in words if len(w) >= self.min_word_length]
            
            return words
            
        except sqlite3.Error as e:
            logger.error(f"Error extracting words from session {session_id}: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Vocabulary Learning
    # -------------------------------------------------------------------------
    
    def learn_vocabulary(self, sessions: List[SessionStats]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Learn important and unimportant words from sessions.
        
        Important words come from important sessions.
        Unimportant words come from unimportant sessions.
        """
        self._update_progress("vocabulary", 0.1, "Analyzing session importance...")
        
        # Calculate importance for all sessions
        for i, session in enumerate(sessions):
            session.importance_score = self._calculate_session_importance(session)
            if i % 10 == 0:
                progress = 0.1 + (i / len(sessions)) * 0.3 if sessions else 0.4
                self._update_progress("vocabulary", progress, f"Scoring sessions...")
        
        # Separate sessions by importance
        important_sessions = [s for s in sessions if s.importance_score > self.importance_threshold]
        unimportant_sessions = [s for s in sessions if s.importance_score < 0.3]
        
        logger.info(f"Found {len(important_sessions)} important sessions, {len(unimportant_sessions)} unimportant sessions")
        
        # Extract words from important sessions
        self._update_progress("vocabulary", 0.5, "Extracting important words...")
        important_word_counts = Counter()
        for i, session in enumerate(important_sessions):
            words = self._extract_words_from_session(session.session_id)
            important_word_counts.update(words)
            
            if i % 5 == 0 and important_sessions:
                progress = 0.5 + (i / len(important_sessions)) * 0.2
                self._update_progress("vocabulary", progress, f"Processing important sessions...")
        
        # Extract words from unimportant sessions
        self._update_progress("vocabulary", 0.7, "Extracting unimportant words...")
        unimportant_word_counts = Counter()
        for i, session in enumerate(unimportant_sessions):
            words = self._extract_words_from_session(session.session_id)
            unimportant_word_counts.update(words)
            
            if i % 5 == 0 and unimportant_sessions:
                progress = 0.7 + (i / len(unimportant_sessions)) * 0.2
                self._update_progress("vocabulary", progress, f"Processing unimportant sessions...")
        
        # Calculate word weights
        self._update_progress("vocabulary", 0.9, "Calculating word weights...")
        important_words = {}
        unimportant_words = {}
        
        # Common stopwords to ignore (English and Chinese)
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'have', 'from', 'what',
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '一些', '可以', '可能', '应该', '需要', '现在', '今天', '明天',
            'weather', 'time', 'today', 'now', 'query', 'search', 'find', 'look'
        }
        
        # Important words: weight = log(frequency) * session_importance_avg
        for word, count in important_word_counts.most_common(self.max_words_to_keep):
            if word in stopwords or len(word) < self.min_word_length:
                continue
            # Weight based on frequency (log scale to reduce dominance of very common words)
            weight = (1.0 + 0.5 * (count ** 0.5))  # 1.0 to ~6.0 for very frequent words
            important_words[word] = weight
        
        # Unimportant words: negative weight
        for word, count in unimportant_word_counts.most_common(self.max_words_to_keep):
            if word in stopwords or len(word) < self.min_word_length:
                continue
            # Negative weight for unimportant words
            weight = -0.5 * (1.0 + 0.3 * (count ** 0.5))  # -0.5 to ~-3.0
            unimportant_words[word] = weight
        
        self._update_progress("vocabulary", 1.0, "Vocabulary learned!")
        logger.info(f"Learned {len(important_words)} important words, {len(unimportant_words)} unimportant words")
        
        return important_words, unimportant_words
    
    # -------------------------------------------------------------------------
    # Memory Scoring
    # -------------------------------------------------------------------------
    
    def _score_memory(self, memory_text: str) -> Tuple[float, Dict[str, float]]:
        """
        Score a memory based on learned vocabulary.
        
        Returns:
            Tuple of (total_score, word_contributions)
        """
        text_lower = memory_text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 1.0  # Base score
        contributions = {}
        
        # Check for important words
        for word, weight in self.important_words.items():
            if word in text_lower:
                score += weight * 0.1  # Scale down contribution
                contributions[word] = weight * 0.1
        
        # Check for unimportant words
        for word, weight in self.unimportant_words.items():
            if word in text_lower:
                score += weight * 0.15  # Slightly stronger negative impact
                contributions[word] = weight * 0.15
        
        # Length factor: very short memories might be fragments
        if len(memory_text) < 20:
            score -= 0.3
            contributions["length_short"] = -0.3
        elif len(memory_text) > 100:
            score += 0.1  # Longer memories often more substantial
            contributions["length_long"] = 0.1
        
        return max(0.0, score), contributions
    
    # -------------------------------------------------------------------------
    # Main Sleep Function
    # -------------------------------------------------------------------------
    
    def sleep(self, mode: str = "quick", apply_changes: bool = False) -> Dict[str, Any]:
        """
        Perform memory consolidation (sleep mode).
        
        Args:
            mode: "quick" for fast filtering, "deep" for thorough analysis
            apply_changes: When True, persist deletions to MEMORY.md. Otherwise preview only.
        
        Returns:
            Dictionary with results and statistics
        """
        if mode not in {"quick", "deep"}:
            return {
                "success": False,
                "error": f"Unsupported sleep mode: {mode}. Expected 'quick' or 'deep'.",
            }

        start_time = time.time()
        logger.info(f"Starting sleep mode: {mode}")
        
        # Get all sessions
        self._update_progress("sleep", 0.1, "Loading sessions...")
        sessions = self._get_all_sessions()
        
        if not sessions:
            return {
                "success": False,
                "error": "No sessions found in database. Sleep requires historical session data.",
                "stats": {"sessions_analyzed": 0}
            }
        
        # Learn vocabulary from sessions
        self._update_progress("sleep", 0.2, "Learning vocabulary from sessions...")
        self.important_words, self.unimportant_words = self.learn_vocabulary(sessions)
        
        # Load current memories
        self._update_progress("sleep", 0.6, "Loading memories...")
        memory_entries = self.memory_store.memory_entries.copy()

        if not memory_entries:
            return {
                "success": True,
                "mode": mode,
                "applied": False,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "stats": {
                    "sessions_analyzed": len(sessions),
                    "important_sessions": len(
                        [s for s in sessions if s.importance_score > self.importance_threshold]
                    ),
                    "avg_session_duration_hours": round(
                        sum(s.duration_hours for s in sessions) / len(sessions), 2
                    ) if sessions else 0,
                    "memories_before": 0,
                    "memories_after": 0,
                    "memories_deleted": 0,
                    "chars_before": 0,
                    "chars_after": 0,
                    "chars_saved": 0,
                    "char_reduction_pct": 0,
                },
                "vocabulary": {
                    "important_words_count": len(self.important_words),
                    "unimportant_words_count": len(self.unimportant_words),
                    "top_important_words": [],
                    "top_unimportant_words": [],
                },
                "deleted_memories_preview": [],
                "top_kept_memories": [],
            }
        
        # Score all memories
        self._update_progress("sleep", 0.7, "Scoring memories...")
        scored_memories = []
        deleted_memories = []
        kept_memories = []
        
        for i, memory in enumerate(memory_entries):
            score, contributions = self._score_memory(memory)
            scored_memories.append({
                "text": memory,
                "score": score,
                "contributions": contributions
            })
            
            if i % 10 == 0 and memory_entries:
                progress = 0.7 + (i / len(memory_entries)) * 0.2
                self._update_progress("sleep", progress, f"Scoring memory {i+1}/{len(memory_entries)}...")
        
        # Sort by score
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        
        # Determine which memories to delete
        delete_threshold = self.memory_delete_threshold
        if mode == "deep":
            delete_threshold = 0.6  # More aggressive in deep mode
        
        for mem in scored_memories:
            if mem["score"] < delete_threshold:
                deleted_memories.append(mem)
            else:
                kept_memories.append(mem)
        
        applied = False

        # Apply changes if there are memories to delete
        if deleted_memories and apply_changes:
            self._update_progress("sleep", 0.9, "Applying changes...")
            # Create new memory list with only kept memories
            new_memories = [mem["text"] for mem in kept_memories]
            
            # Update memory store
            self.memory_store.memory_entries = new_memories
            self.memory_store.save_to_disk("memory")
            applied = True
            
            logger.info(f"Deleted {len(deleted_memories)} memories, kept {len(kept_memories)}")
        elif deleted_memories:
            logger.info("Sleep preview found %d memories below threshold; no changes applied", len(deleted_memories))
        else:
            logger.info("No memories met deletion threshold")
        
        # Generate report
        elapsed = time.time() - start_time
        report = self._generate_report(
            sessions, scored_memories, deleted_memories, kept_memories, 
            mode, elapsed, applied
        )
        
        self._update_progress("sleep", 1.0, "Sleep complete!")
        return report
    
    def _generate_report(self, sessions: List[SessionStats], 
                        all_memories: List[Dict], 
                        deleted: List[Dict], 
                        kept: List[Dict],
                        mode: str,
                        elapsed_seconds: float,
                        applied: bool) -> Dict[str, Any]:
        """Generate a detailed sleep report."""
        
        # Calculate session statistics
        important_sessions = [s for s in sessions if s.importance_score > self.importance_threshold]
        avg_session_duration = sum(s.duration_hours for s in sessions) / len(sessions) if sessions else 0
        
        # Memory statistics
        total_chars_before = sum(len(m["text"]) for m in all_memories)
        total_chars_after = sum(len(m["text"]) for m in kept)
        char_savings = total_chars_before - total_chars_after
        
        # Top important words
        top_important = sorted(self.important_words.items(), key=lambda x: x[1], reverse=True)[:10]
        top_unimportant = sorted(self.unimportant_words.items(), key=lambda x: x[1])[:10]
        
        return {
            "success": True,
            "mode": mode,
            "applied": applied,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "stats": {
                "sessions_analyzed": len(sessions),
                "important_sessions": len(important_sessions),
                "avg_session_duration_hours": round(avg_session_duration, 2),
                "memories_before": len(all_memories),
                "memories_after": len(kept),
                "memories_deleted": len(deleted),
                "chars_before": total_chars_before,
                "chars_after": total_chars_after,
                "chars_saved": char_savings,
                "char_reduction_pct": round((char_savings / total_chars_before * 100) if total_chars_before > 0 else 0, 1)
            },
            "vocabulary": {
                "important_words_count": len(self.important_words),
                "unimportant_words_count": len(self.unimportant_words),
                "top_important_words": [{"word": w, "weight": round(v, 2)} for w, v in top_important],
                "top_unimportant_words": [{"word": w, "weight": round(v, 2)} for w, v in top_unimportant]
            },
            "deleted_memories_preview": [
                {
                    "text": m["text"][:100] + ("..." if len(m["text"]) > 100 else ""),
                    "score": round(m["score"], 2)
                }
                for m in deleted[:5]  # Preview first 5
            ],
            "top_kept_memories": [
                {
                    "text": m["text"][:100] + ("..." if len(m["text"]) > 100 else ""),
                    "score": round(m["score"], 2)
                }
                for m in kept[:5]  # Top 5 highest scoring
            ]
        }
    
# -----------------------------------------------------------------------------
# Helper functions for CLI integration
# -----------------------------------------------------------------------------

def format_sleep_report(report: Dict[str, Any]) -> str:
    """Format the sleep report for CLI display."""
    if not report.get("success"):
        return f"❌ Sleep failed: {report.get('error', 'Unknown error')}"
    
    stats = report["stats"]
    vocab = report["vocabulary"]
    
    lines = []
    lines.append("🌙 Hermes Sleep Mode Complete!")
    lines.append("═══════════════════════════════════════════")
    action = "applied" if report.get("applied") else "preview"
    lines.append(f"Mode: {report['mode']} | Result: {action} | Time: {report['elapsed_seconds']}s")
    lines.append("")
    
    # Session analysis
    lines.append("📊 Session Analysis:")
    lines.append(f"  • Sessions analyzed: {stats['sessions_analyzed']}")
    lines.append(f"  • Important sessions: {stats['important_sessions']}")
    lines.append(f"  • Avg duration: {stats['avg_session_duration_hours']} hours")
    lines.append("")
    
    # Memory consolidation
    lines.append("💾 Memory Consolidation:")
    lines.append(f"  • Memories before: {stats['memories_before']}")
    lines.append(f"  • Memories after: {stats['memories_after']}")
    lines.append(f"  • Memories deleted: {stats['memories_deleted']}")
    lines.append(f"  • Space saved: {stats['chars_saved']:,} chars ({stats['char_reduction_pct']}%)")
    if not report.get("applied"):
        lines.append("  • Preview only: rerun with --apply to persist these changes")
    lines.append("")
    
    # Vocabulary learned
    lines.append("📚 Vocabulary Learned:")
    lines.append(f"  • Important words: {vocab['important_words_count']}")
    lines.append(f"  • Unimportant words: {vocab['unimportant_words_count']}")
    
    if vocab['top_important_words']:
        lines.append("  • Top important words: " + ", ".join(
            f"{w['word']}({w['weight']})" for w in vocab['top_important_words'][:5]
        ))
    
    if vocab['top_unimportant_words']:
        lines.append("  • Top unimportant words: " + ", ".join(
            f"{w['word']}({w['weight']})" for w in vocab['top_unimportant_words'][:5]
        ))
    lines.append("")
    
    # Deleted memories preview
    if report.get('deleted_memories_preview'):
        lines.append("🗑️ Deleted Memories (preview):")
        for i, mem in enumerate(report['deleted_memories_preview']):
            lines.append(f"  {i+1}. [{mem['score']}] {mem['text']}")
        lines.append("")
    
    # Top kept memories
    if report.get('top_kept_memories'):
        lines.append("💾 Top Kept Memories:")
        for i, mem in enumerate(report['top_kept_memories']):
            lines.append(f"  {i+1}. [{mem['score']}] {mem['text']}")
        lines.append("")
    
    lines.append("🛌 Sleep complete! Memory has been consolidated.")
    return "\n".join(lines)
