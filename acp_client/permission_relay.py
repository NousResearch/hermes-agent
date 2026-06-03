"""Inbound ``request_permission`` policy for the ACP client (deny-default).

Reversed mirror of ``acp_adapter/permissions.py``.  On the server side Hermes
*asks* an editor for permission; here Hermes is the **parent / policy
boundary** and an *external* agent (claude, codex, …) asks Hermes for
permission to act.  Kanban/delegate contexts are non-interactive, so decisions
are **policy-driven**, not a UI round-trip (design §2.4):

1. Workspace allowlist  → ``allow_once`` for a small set of safe tool kinds
   whose locations stay inside the bound workspace.
2. Credential-path denylist → ``deny`` (structured reason) for anything that
   touches auth/state/env files, regardless of workspace.
3. Deny-default          → ``deny`` for everything else.

``allow_always`` is **never** honoured: any persistent-allow option offered by
the external agent is downgraded to ``allow_once`` so the boundary stays
per-prompt.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Tool kinds we are willing to auto-allow when they stay inside the workspace.
# Deliberately excludes "execute" (arbitrary shell) — that is deny-default.
_WORKSPACE_SAFE_KINDS: frozenset[str] = frozenset({"read", "edit"})

# Substrings that mark a credential / state path.  A request touching any of
# these is denied even if it is technically inside the workspace, because these
# files should never be read or written by an external agent.
_CREDENTIAL_PATH_MARKERS: tuple[str, ...] = (
    "auth.json",
    "auth.",            # auth.<provider>.json
    ".env",
    "credentials",
    "secrets",
    "state.db",
    "kanban.db",
    ".ssh",
    "id_rsa",
    ".netrc",
    "token",
)

# ACP outcome literals.
_ALLOW = "allow"
_DENY = "deny"


@dataclass
class PermissionDecision:
    """Result of evaluating one inbound permission request."""

    outcome: str  # "allow" | "deny"
    reason: str
    option_id: Optional[str] = None  # chosen allow option id, when allowing


@dataclass
class PermissionRelay:
    """Deny-default permission policy for an external ACP agent.

    Args:
        workspace_path: Absolute path the external agent is bound to.  Only
            read/edit requests whose locations resolve inside this directory
            are eligible for auto-allow.
        allow_workspace_edits: When ``False`` even in-workspace edits are denied
            (read-only posture).  Defaults to ``True``.
        audit_log: Optional callable invoked with each :class:`PermissionDecision`
            for EMA observability (a ``progress.jsonl`` writer in Phase 2).
    """

    workspace_path: str
    allow_workspace_edits: bool = True
    audit_log: Any = None
    _denied: int = field(default=0, init=False, repr=False)
    _allowed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace_path = os.path.abspath(os.path.expanduser(self.workspace_path))

    # ---- policy ------------------------------------------------------------

    def _is_inside_workspace(self, path: str) -> bool:
        try:
            resolved = os.path.abspath(os.path.expanduser(path))
        except Exception:
            return False
        # ``commonpath`` raises on different drives / relative mix — treat as
        # outside on any failure (deny-default).
        try:
            return os.path.commonpath([resolved, self.workspace_path]) == self.workspace_path
        except (ValueError, Exception):
            return False

    @staticmethod
    def _touches_credential_path(text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in _CREDENTIAL_PATH_MARKERS)

    def evaluate(
        self,
        *,
        kind: Optional[str],
        locations: Optional[Sequence[str]] = None,
        raw_input: Any = None,
        title: str = "",
    ) -> PermissionDecision:
        """Return the policy decision for a tool-call permission request.

        Args:
            kind:       ACP tool kind ("read"/"edit"/"execute"/…).
            locations:  File paths the tool intends to touch.
            raw_input:  The tool's raw input payload (inspected for cred paths).
            title:      Human title (inspected for cred paths).
        """
        locations = list(locations or [])

        # 1. Credential-path denylist — highest priority, beats workspace allow.
        haystack = " ".join(
            [title or "", " ".join(locations), self._stringify(raw_input)]
        )
        if self._touches_credential_path(haystack):
            return self._record(
                PermissionDecision(
                    _DENY,
                    "denied: request touches a credential/state path "
                    "(auth/env/state/secrets)",
                )
            )

        # 2. Workspace allowlist for safe kinds.
        if kind in _WORKSPACE_SAFE_KINDS:
            if kind == "edit" and not self.allow_workspace_edits:
                return self._record(
                    PermissionDecision(_DENY, "denied: edits disabled for this session")
                )
            if locations and all(self._is_inside_workspace(loc) for loc in locations):
                return self._record(
                    PermissionDecision(
                        _ALLOW,
                        f"allowed: {kind} inside workspace",
                        option_id="allow_once",
                    )
                )
            if not locations and kind == "read":
                # A read with no declared location can't be proven in-workspace.
                return self._record(
                    PermissionDecision(
                        _DENY, "denied: read with no declared location (cannot verify workspace)"
                    )
                )
            return self._record(
                PermissionDecision(_DENY, f"denied: {kind} target outside workspace")
            )

        # 3. Deny-default for everything else (execute, delete, fetch, unknown).
        return self._record(
            PermissionDecision(_DENY, f"denied (default): kind={kind!r} not in allowlist")
        )

    def select_option(
        self, decision: PermissionDecision, options: Sequence[Any]
    ) -> Optional[str]:
        """Map a decision onto one of the ACP-offered ``option_id`` values.

        On allow, prefer an ``allow_once`` option; never select an
        ``allow_always``/``allow_session`` (persistent) option — those are
        downgraded by refusing to pick them.  On deny, prefer an explicit
        reject option; ``None`` signals the caller to send a ``DeniedOutcome``.
        """
        ids = {self._option_id(o): o for o in options}
        if decision.outcome == _ALLOW:
            for candidate in ("allow_once", "allow"):
                if candidate in ids:
                    return candidate
            # No once-scoped option offered → refuse to escalate to persistent.
            logger.warning(
                "External agent offered no allow_once option; denying to avoid "
                "honouring a persistent allow"
            )
            return None
        # deny
        for candidate in ("reject_once", "deny", "reject"):
            if candidate in ids:
                return candidate
        return None

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _option_id(option: Any) -> str:
        return str(getattr(option, "option_id", getattr(option, "kind", "")) or "")

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            import json

            return json.dumps(value, default=str)
        except Exception:
            return str(value)

    def _record(self, decision: PermissionDecision) -> PermissionDecision:
        if decision.outcome == _ALLOW:
            self._allowed += 1
        else:
            self._denied += 1
        logger.info(
            "ACP client permission %s: %s", decision.outcome.upper(), decision.reason
        )
        if callable(self.audit_log):
            try:
                self.audit_log(decision)
            except Exception:
                logger.debug("permission audit_log hook failed", exc_info=True)
        return decision

    @property
    def stats(self) -> dict:
        return {"allowed": self._allowed, "denied": self._denied}
