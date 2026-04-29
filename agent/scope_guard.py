"""Scope guard — detect and prevent runaway task loops.

Monitors agent execution for signs of unbounded tasks:
- Same tool called with identical args in consecutive iterations
- Iteration count exceeding expected complexity bounds
- No progress indicators (same error repeated)

When a runaway is detected, injects a mandatory checkpoint message
that forces the model to justify continuation or pivot strategy.
"""

import hashlib
import json
import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How many consecutive identical tool calls trigger a runaway warning
RUNAWAY_THRESHOLD = 3

# How many iterations before forcing a progress checkpoint
CHECKPOINT_INTERVAL = 25

# Maximum checkpoint messages per session to avoid spam
MAX_CHECKPOINTS = 4

# Window size for tracking recent tool calls
TRACKING_WINDOW = 10


class ScopeGuard:
    """Detect runaway loops and enforce progress checkpoints.

    Thread-safe — tool calls may execute concurrently.
    """

    def __init__(
        self,
        runaway_threshold: int = RUNAWAY_THRESHOLD,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        self._runaway_threshold = runaway_threshold
        self._checkpoint_interval = checkpoint_interval
        self._lock = threading.Lock()

        # Recent tool call signatures (rolling window)
        self._recent_calls: deque = deque(maxlen=TRACKING_WINDOW)

        # Counters
        self._iteration_count = 0
        self._checkpoints_emitted = 0

        # Track consecutive identical calls per tool
        self._consecutive: Dict[str, int] = defaultdict(int)
        self._last_signature: Dict[str, str] = {}

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        is_error: bool,
    ) -> Optional[str]:
        """Record a tool call and return an advisory message if warranted.

        Returns None when no intervention is needed.  Otherwise returns a
        structured message that should be injected into the tool result.
        """
        # Compute a stable signature for the call
        args_str = json.dumps(tool_args, sort_keys=True, default=str)
        sig = hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()[:12]

        with self._lock:
            self._iteration_count += 1
            self._recent_calls.append((tool_name, sig, is_error))

            # Check for runaway: same tool + same args consecutively
            last_sig = self._last_signature.get(tool_name)
            if last_sig == sig and is_error:
                self._consecutive[tool_name] += 1
            else:
                self._consecutive[tool_name] = 1 if is_error else 0
            self._last_signature[tool_name] = sig

            # Runaway detection
            if self._consecutive[tool_name] >= self._runaway_threshold:
                self._consecutive[tool_name] = 0  # Reset to avoid spam
                return self._build_runaway_warning(tool_name)

            # Periodic progress checkpoint
            if (self._iteration_count > 0
                    and self._iteration_count % self._checkpoint_interval == 0
                    and self._checkpoints_emitted < MAX_CHECKPOINTS):
                self._checkpoints_emitted += 1
                return self._build_progress_checkpoint()

        return None

    def _build_runaway_warning(self, tool_name: str) -> str:
        """Build a runaway loop warning message."""
        return (
            f"\n\n[SCOPE GUARD: '{tool_name}' has been called with identical "
            f"failing arguments {self._runaway_threshold}+ consecutive times. "
            f"This indicates a runaway loop. You MUST:\n"
            f"1. STOP retrying the same approach\n"
            f"2. Analyze WHY the tool is failing\n"
            f"3. Try a fundamentally different tool or strategy\n"
            f"4. If stuck, inform the user and ask for guidance\n"
            f"Continuing to retry the same failing call is not acceptable.]"
        )

    def _build_progress_checkpoint(self) -> str:
        """Build a progress checkpoint message."""
        return (
            f"\n\n[PROGRESS CHECKPOINT (iteration {self._iteration_count}): "
            f"You have been working for {self._iteration_count} iterations. "
            f"Briefly assess: (1) Are you making progress toward the goal? "
            f"(2) Is the current approach working? "
            f"(3) Should you adjust strategy or decompose the task? "
            f"If the task is complete, summarize and stop. "
            f"If stuck, pivot or ask the user for clarification.]"
        )

    def get_stats(self) -> Dict:
        """Return current tracking statistics."""
        with self._lock:
            return {
                "iteration_count": self._iteration_count,
                "checkpoints_emitted": self._checkpoints_emitted,
                "consecutive_failures": dict(self._consecutive),
                "recent_calls_count": len(self._recent_calls),
            }

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._recent_calls.clear()
            self._consecutive.clear()
            self._last_signature.clear()
            self._iteration_count = 0
            self._checkpoints_emitted = 0
