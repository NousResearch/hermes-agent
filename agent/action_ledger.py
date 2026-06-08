"""
Action Ledger for tracking context management actions in Hermes.

This module implements an action ledger that records all context-related
operations for phone sessions, enabling better monitoring, debugging, and
preventing unwanted compaction behavior.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ContextAction:
    """Represents a single context management action."""
    timestamp: float
    action_type: str
    session_id: str
    message_id: Optional[str] = None
    context_tokens: Optional[int] = None
    compression_ratio: Optional[float] = None
    summary_tokens: Optional[int] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary."""
        return cls(**data)


class ActionLedger:
    """Ledger for tracking context management actions."""
    
    def __init__(self, ledger_file: Optional[str] = None):
        self.ledger_file = ledger_file or "~/.hermes/action_ledger.json"
        self.ledger_file = Path(self.ledger_file).expanduser()
        self.actions: List[ContextAction] = []
        self._load_ledger()
    
    def _load_ledger(self):
        """Load existing ledger from file."""
        try:
            if self.ledger_file.exists():
                with open(self.ledger_file, 'r') as f:
                    data = json.load(f)
                    self.actions = [ContextAction.from_dict(action) for action in data.get('actions', [])]
        except Exception as e:
            print(f"Warning: Failed to load action ledger: {e}")
    
    def _save_ledger(self):
        """Save ledger to file."""
        try:
            data = {
                'actions': [action.to_dict() for action in self.actions],
                'last_save': time.time()
            }
            self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save action ledger: {e}")
    
    def record_action(self, action_type: str, session_id: str, 
                        message_id: Optional[str] = None, 
                        context_tokens: Optional[int] = None,
                        compression_ratio: Optional[float] = None,
                        summary_tokens: Optional[int] = None,
                        reason: Optional[str] = None,
                        details: Optional[Dict[str, Any]] = None):
        """
        Record a context management action.
        
        Args:
            action_type: Type of action (e.g., "compaction", "compression", "overflow")
            session_id: Session identifier
            message_id: Message identifier (optional)
            context_tokens: Token count before action
            compression_ratio: Ratio of compressed/total tokens
            summary_tokens: Number of tokens in summary
            reason: Reason for the action
            details: Additional action details
        """
        action = ContextAction(
            timestamp=time.time(),
            action_type=action_type,
            session_id=session_id,
            message_id=message_id,
            context_tokens=context_tokens,
            compression_ratio=compression_ratio,
            summary_tokens=summary_tokens,
            reason=reason,
            details=details or {}
        )
        
        self.actions.append(action)
        self._save_ledger()
        
        return action
    
    def get_session_actions(self, session_id: str) -> List[ContextAction]:
        """Get all actions for a given session."""
        return [action for action in self.actions if action.session_id == session_id]
    
    def get_recent_actions(self, limit: int = 100) -> List[ContextAction]:
        """Get most recent actions."""
        return self.actions[-limit:]
    
    def get_compaction_actions(self, session_id: str) -> List[ContextAction]:
        """Get all compaction-related actions for a session."""
        return [action for action in self.actions 
                if action.session_id == session_id and 
                (action.action_type == "compaction" or action.action_type == "compression")]
    
    def get_compaction_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of compaction actions for a session."""
        compaction_actions = self.get_compaction_actions(session_id)
        if not compaction_actions:
            return {}
            
        total_actions = len(compaction_actions)
        total_tokens = sum(action.context_tokens or 0 for action in compaction_actions)
        avg_ratio = sum(action.compression_ratio or 0 for action in compaction_actions) / total_actions
        
        return {
            "session_id": session_id,
            "total_compactions": total_actions,
            "total_context_tokens": total_tokens,
            "average_compression_ratio": avg_ratio,
            "first_action": min(action.timestamp for action in compaction_actions),
            "last_action": max(action.timestamp for action in compaction_actions)
        }
    
    def clear_session_actions(self, session_id: str):
        """Clear all actions for a specific session."""
        self.actions = [action for action in self.actions if action.session_id != session_id]
        self._save_ledger()
    
    def clear_old_actions(self, days_old: int = 30):
        """Clear actions older than specified days."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        self.actions = [action for action in self.actions if action.timestamp >= cutoff_time]
        self._save_ledger()
    
    def record_pre_compression(self, n_messages: int, current_tokens: int, session_id: str = "default"):
        """Record pre-compression state."""
        self.record_action(
            action_type="pre_compression",
            session_id=session_id,
            context_tokens=current_tokens,
            reason=f"Pre-compression: {n_messages} messages, {current_tokens} tokens",
            details={
                "n_messages": n_messages,
                "current_tokens": current_tokens
            }
        )
    
    def record_post_compression(self, n_messages: int, new_tokens: int, saved_tokens: int, 
                                savings_pct: float, summary_used: bool, fallback_used: bool,
                                error: Optional[str] = None, session_id: str = "default"):
        """Record post-compression state."""
        self.record_action(
            action_type="post_compression",
            session_id=session_id,
            context_tokens=new_tokens,
            compression_ratio=1.0 - (savings_pct / 100.0) if savings_pct <= 100 else 0.0,
            reason=f"Post-compression: {n_messages} messages, {new_tokens} tokens, {savings_pct:.1f}% saved",
            details={
                "n_messages": n_messages,
                "new_tokens": new_tokens,
                "saved_tokens": saved_tokens,
                "savings_pct": savings_pct,
                "summary_used": summary_used,
                "fallback_used": fallback_used,
                "error": error
            }
        )