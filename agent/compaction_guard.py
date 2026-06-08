"""
Compaction Guard for Phone Sessions.

This module provides guards against unwanted context compaction for phone/gateway
sessions. Phone sessions have different constraints than desktop sessions:
- Limited screen real estate for displaying compressed summaries
- Higher risk of losing active task context
- Need for better continuity across compaction boundaries

The guard integrates with the ContextCompressor to:
1. Track compaction actions in an action ledger
2. Enforce minimum tail sizes for phone sessions
3. Prevent compaction during active conversations
4. Provide phone-specific compaction thresholds
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompactionGuardConfig:
    """Configuration for compaction guards."""
    # Minimum tail size for phone sessions (messages)
    phone_min_tail_messages: int = 30
    # Minimum tail size for phone sessions (tokens)
    phone_min_tail_tokens: int = 40000
    # Threshold override for phone sessions (higher to avoid premature compaction)
    phone_threshold_percent: float = 0.60
    # Guard against compaction within N seconds of the last compaction
    min_compaction_interval: float = 120.0
    # Maximum compressions per session before requiring manual intervention
    max_auto_compressions: int = 5
    # Phone session detection platforms
    phone_platforms: List[str] = field(default_factory=lambda: [
        "telegram", "whatsapp", "signal", "sms", "bluebubbles", "feishu",
        "wecom", "weixin", "qqbot", "dingtalk"
    ])


class CompactionGuard:
    """Guard against unwanted context compaction for phone/gateway sessions."""
    
    def __init__(self, config: Optional[CompactionGuardConfig] = None):
        self.config = config or CompactionGuardConfig()
        self._last_compaction_time: float = 0
        self._compaction_count: int = 0
        self._session_active: bool = True
    
    def is_phone_platform(self, platform: Optional[str]) -> bool:
        """Check if the given platform is considered a phone/messaging platform."""
        if not platform:
            return False
        return platform.lower() in [p.lower() for p in self.config.phone_platforms]
    
    def should_block_compaction(self, platform: Optional[str], 
                                 messages: List[Dict[str, Any]],
                                 current_tokens: int = 0) -> tuple[bool, str]:
        """
        Determine if compaction should be blocked for this request.
        
        Returns:
            (should_block, reason): Tuple of block flag and reason string
        """
        # Check minimum compaction interval
        now = time.time()
        if now - self._last_compaction_time < self.config.min_compaction_interval:
            remaining = self.config.min_compaction_interval - (now - self._last_compaction_time)
            return True, f"Within {remaining:.0f}s cooldown since last compaction"
        
        # Check max auto-compressions for phone sessions
        if self.is_phone_platform(platform):
            if self._compaction_count >= self.config.max_auto_compressions:
                return True, f"Max auto-compressions ({self.config.max_auto_compressions}) reached for phone session"
        
        # Check minimum tail size for phone sessions
        if self.is_phone_platform(platform):
            if len(messages) < self.config.phone_min_tail_messages + 3:
                return True, f"Too few messages ({len(messages)}) for phone session compaction"
        
        return False, ""
    
    def get_phone_threshold(self, context_length: int, platform: Optional[str]) -> int:
        """Get the compression threshold for phone sessions (higher than default)."""
        if self.is_phone_platform(platform):
            return int(context_length * self.config.phone_threshold_percent)
        return int(context_length * 0.50)  # default
    
    def get_phone_tail_budget(self, threshold_tokens: int, platform: Optional[str]) -> int:
        """Get the tail token budget for phone sessions (larger than default)."""
        if self.is_phone_platform(platform):
            # For phone sessions, use the larger of the calculated budget or the minimum
            calculated = int(threshold_tokens * 0.25)  # 25% of threshold for tail
            return max(calculated, self.config.phone_min_tail_tokens)
        return int(threshold_tokens * 0.20)  # default
    
    def record_compaction(self):
        """Record that a compaction just occurred."""
        self._last_compaction_time = time.time()
        self._compaction_count += 1
    
    def reset(self):
        """Reset guard state for a new session."""
        self._last_compaction_time = 0
        self._compaction_count = 0
        self._session_active = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current guard status for debugging/monitoring."""
        return {
            "last_compaction_time": self._last_compaction_time,
            "compaction_count": self._compaction_count,
            "session_active": self._session_active,
            "time_since_last_compaction": time.time() - self._last_compaction_time,
            "max_auto_compressions": self.config.max_auto_compressions,
            "min_compaction_interval": self.config.min_compaction_interval
        }


def create_phone_compaction_guard(config: Optional[CompactionGuardConfig] = None) -> CompactionGuard:
    """Factory function to create a CompactionGuard."""
    return CompactionGuard(config)