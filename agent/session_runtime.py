"""SessionRuntime — per-session ephemeral workspace with output promotion.

Plan 002-B: Runtime Session Isolation.

Every AIAgent session gets an isolated sandbox under:
    runtime/sessions/{session_id}/
        workspace/          ← default cwd for the terminal tool
        workspace/outputs/  ← files here are promoted to artifacts on close
        subagents/          ← container for subagent workspaces

On session close:
  1. workspace/outputs/ is copied to users/{id}/artifacts/sessions/{date}-{id}/outputs/
  2. runtime/sessions/{id}/ is deleted

Design constraints:
  - Never raises on close() — sandbox cleanup must not crash the agent
  - Idempotent _setup() — safe to call from concurrent constructors
  - Does NOT credential-inject (that's Phase 002-C / CredentialResolver)
  - Does NOT block on large output directories — shutil operations are
    synchronous but are bounded by outputs the agent explicitly placed there
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_runtime_root, get_user_home

logger = logging.getLogger(__name__)

# Only files under this sub-directory of workspace are promoted on close.
# This is intentional: agents must explicitly move outputs here to persist them.
# Temp scratch work stays in workspace/ and is silently discarded.
_OUTPUTS_SUBDIR = "outputs"


class SessionRuntime:
    """Manages an ephemeral sandbox for one AIAgent session.

    Lifecycle::

        runtime = SessionRuntime(session_id="20250518_abc123", user_id="blake")
        # ... session runs ...
        runtime.close()   # promotes outputs, destroys sandbox

    Directory layout created on __init__::

        runtime/sessions/{session_id}/
            workspace/
            workspace/outputs/
            subagents/

    Args:
        session_id: Stable identifier for this session (used as directory name
                    and as part of the promoted artifact path).
        user_id: User namespace to promote artifacts into.
                 Reads HERMES_USER_ID from env if not provided.
                 If neither is set, output promotion is skipped on close.
    """

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id or os.environ.get("HERMES_USER_ID", "").strip()

        self.root = get_runtime_root() / "sessions" / session_id
        self.workspace = self.root / "workspace"
        self.outputs = self.workspace / _OUTPUTS_SUBDIR
        self.subagents_dir = self.root / "subagents"

        self._setup()

    def _setup(self) -> None:
        """Create sandbox directory tree.

        Uses exist_ok=True so concurrent constructors with the same session_id
        (unlikely but possible in test scenarios) don't race.
        """
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.subagents_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("SessionRuntime: created sandbox at %s", self.root)

    def subagent_workspace(self, sub_id: str) -> Path:
        """Return (and create) an isolated workspace for a subagent.

        The path is deterministic given sub_id, so this can be called by
        both the parent (to determine where to tell the subagent to write)
        and the subagent itself (to verify its own workspace root).

        Args:
            sub_id: Unique identifier for this subagent (e.g. "sa-0-abc12345").

        Returns:
            Path: ``subagents/{sub_id}/workspace/``, guaranteed to exist.
        """
        path = self.subagents_dir / sub_id / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        """Promote outputs to artifacts and destroy sandbox.

        Always succeeds — errors during promotion or cleanup are logged as
        warnings, not raised.  The session must not crash because output
        promotion failed (e.g. disk full, permission error).
        """
        if not self.root.exists():
            logger.debug(
                "SessionRuntime.close: sandbox %s already gone, nothing to do",
                self.root,
            )
            return

        # Only promote when we know which user's artifacts directory to write to.
        if self.user_id and self._has_outputs():
            self._promote_outputs()

        self._destroy_sandbox()

    def _has_outputs(self) -> bool:
        """Return True if outputs/ contains at least one file."""
        try:
            if not self.outputs.exists():
                return False
            return any(True for _ in self.outputs.iterdir())
        except Exception:
            return False

    def _promote_outputs(self) -> None:
        """Copy workspace/outputs/ → users/{id}/artifacts/sessions/{date}-{id}/outputs/.

        Uses shutil.copytree so the source is not modified and the sandbox
        cleanup step can unconditionally rmtree the whole sandbox root.

        Assumption: outputs/ is not gigabytes of data.  It's intended for
        concise deliverables (reports, code files, summaries) — not raw data.
        If this becomes a bottleneck, a future phase can make this async.
        """
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dest = (
            get_user_home(self.user_id)
            / "artifacts"
            / "sessions"
            / f"{date_str}-{self.session_id}"
            / "outputs"
        )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(self.outputs), str(dest))
            logger.info(
                "SessionRuntime: promoted outputs → %s (%d item(s))",
                dest,
                sum(1 for _ in dest.rglob("*") if _.is_file()),
            )
        except Exception as exc:
            logger.warning(
                "SessionRuntime: output promotion failed for session %s: %s",
                self.session_id,
                exc,
            )

    def _destroy_sandbox(self) -> None:
        """Delete the entire sandbox directory tree."""
        try:
            shutil.rmtree(str(self.root))
            logger.debug("SessionRuntime: destroyed sandbox at %s", self.root)
        except Exception as exc:
            logger.warning(
                "SessionRuntime: failed to destroy sandbox %s: %s",
                self.root,
                exc,
            )
