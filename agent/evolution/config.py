"""Evolution Engine configuration — mirrors the MemoryProvider config pattern.

Evolution settings live under ``evolution`` in ``config.yaml``:

.. code-block:: yaml

    evolution:
      enabled: false            # Master on/off switch
      mode: "on_failure"       # "on_failure" | "continuous" | "manual"
      max_iterations: 5        # Max improvement attempts per task
      auxiliary_provider: auto # Model for evolution analysis (default: main)
      auxiliary_model: auto
      store_path: null         # Override default trace store path
      regression_gate:
        enabled: true
        max_regression_tasks: 20  # Max prior tasks to check for regression
      safety:
        require_approval_for: ["tool_create", "tool_modify", "prompt_modify"]
        auto_approve: ["skill_create", "skill_patch"]
      trace_retention_days: 90
      max_trace_size_bytes: 10485760  # 10MB per trace
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ENABLED = False
DEFAULT_MODE = "on_failure"  # "on_failure" | "continuous" | "manual"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_REGRESSION_ENABLED = True
DEFAULT_MAX_REGRESSION_TASKS = 20
DEFAULT_TRACE_RETENTION_DAYS = 90
DEFAULT_MAX_TRACE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Which improvement actions require human approval
DEFAULT_REQUIRE_APPROVAL = ["tool_create", "tool_modify", "prompt_modify"]
DEFAULT_AUTO_APPROVE = ["skill_create", "skill_patch"]

# Valid evolution modes
VALID_MODES = {"on_failure", "continuous", "manual"}

# Valid improvement action types
VALID_ACTION_TYPES = {
    "skill_create",
    "skill_patch",
    "tool_create",
    "tool_modify",
    "prompt_modify",
    "memory_update",
    "strategy_change",
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Immutable configuration for the Evolution Engine.

    Construct via :meth:`from_config` (reads config.yaml) or directly
    with the defaults below.
    """

    enabled: bool = DEFAULT_ENABLED
    mode: str = DEFAULT_MODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    auxiliary_provider: Optional[str] = None
    auxiliary_model: Optional[str] = None
    store_path: Optional[Path] = None
    regression_enabled: bool = DEFAULT_REGRESSION_ENABLED
    max_regression_tasks: int = DEFAULT_MAX_REGRESSION_TASKS
    require_approval_for: List[str] = field(default_factory=lambda: list(DEFAULT_REQUIRE_APPROVAL))
    auto_approve: List[str] = field(default_factory=lambda: list(DEFAULT_AUTO_APPROVE))
    trace_retention_days: int = DEFAULT_TRACE_RETENTION_DAYS
    max_trace_size_bytes: int = DEFAULT_MAX_TRACE_SIZE_BYTES

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "EvolutionConfig":
        """Build EvolutionConfig from the evolution section of config.yaml.

        Args:
            config: The full config dict (or None for defaults).
        """
        if config is None:
            try:
                from hermes_cli.config import load_config
                config = load_config()
            except Exception:
                config = {}

        evo = config.get("evolution", {}) if isinstance(config, dict) else {}
        if not isinstance(evo, dict):
            evo = {}

        regression_gate = evo.get("regression_gate", {}) or {}
        safety = evo.get("safety", {}) or {}

        store_path = evo.get("store_path")
        if store_path:
            store_path = Path(store_path)
        else:
            store_path = get_hermes_home() / "evolution"

        return cls(
            enabled=bool(evo.get("enabled", DEFAULT_ENABLED)),
            mode=str(evo.get("mode", DEFAULT_MODE)),
            max_iterations=int(evo.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
            auxiliary_provider=evo.get("auxiliary_provider") or None,
            auxiliary_model=evo.get("auxiliary_model") or None,
            store_path=store_path,
            regression_enabled=bool(regression_gate.get("enabled", DEFAULT_REGRESSION_ENABLED)),
            max_regression_tasks=int(regression_gate.get("max_regression_tasks", DEFAULT_MAX_REGRESSION_TASKS)),
            require_approval_for=list(safety.get("require_approval_for", DEFAULT_REQUIRE_APPROVAL)),
            auto_approve=list(safety.get("auto_approve", DEFAULT_AUTO_APPROVE)),
            trace_retention_days=int(evo.get("trace_retention_days", DEFAULT_TRACE_RETENTION_DAYS)),
            max_trace_size_bytes=int(evo.get("max_trace_size_bytes", DEFAULT_MAX_TRACE_SIZE_BYTES)),
        )

    def validate(self) -> List[str]:
        """Return list of config errors (empty = valid)."""
        errors = []
        if self.mode not in VALID_MODES:
            errors.append(f"mode must be one of {VALID_MODES}, got '{self.mode}'")
        if self.max_iterations < 1:
            errors.append(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.max_regression_tasks < 0:
            errors.append(f"max_regression_tasks must be >= 0, got {self.max_regression_tasks}")
        if self.trace_retention_days < 1:
            errors.append(f"trace_retention_days must be >= 1, got {self.trace_retention_days}")
        # Validate action types
        for action in self.require_approval_for:
            if action not in VALID_ACTION_TYPES:
                errors.append(f"Unknown action type in require_approval_for: '{action}'")
        for action in self.auto_approve:
            if action not in VALID_ACTION_TYPES:
                errors.append(f"Unknown action type in auto_approve: '{action}'")
        return errors

    def needs_approval(self, action_type: str) -> bool:
        """Check whether *action_type* requires human approval."""
        if action_type in self.auto_approve:
            return False
        if action_type in self.require_approval_for:
            return True
        # Default: anything not explicitly auto-approved needs approval
        return True
