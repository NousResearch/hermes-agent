"""
hermes_identity.py — Immutable identity carrier for a single Hermes agent turn.

Rules:
- Created ONCE at the gateway boundary from raw platform-event fields.
- Never mutated. Never constructed by agent business logic.
- Exposes a three-layer scope chain used for skill/memory resolution:
    personal (most specific) → team → global (fallback).

Assumptions surfaced explicitly:
- `team_id` always maps to a workspace/organisation concept.  For platforms
  that lack a workspace (e.g. Telegram DMs) callers SHOULD pass the channel_id
  as team_id so the scoping model degrades gracefully (per Q-0.1 in phase spec).
- All string fields are non-empty after stripping.  The Slack gateway validates
  this at construction time; other gateways must do the same.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HermesIdentity:
    """
    Immutable identity context for one agent turn.

    Fields
    ------
    platform : str
        Lowercase platform name, e.g. "slack", "telegram", "discord".
    team_id : str
        Workspace/organisation identifier.  On Slack this is the workspace ID
        (T-prefixed).  On platforms without a workspace concept callers should
        use the channel_id as a stable fallback.
    user_id : str
        Platform-specific user identifier (U-prefixed on Slack).
    channel_id : str
        Channel or conversation identifier.
    thread_id : str | None
        Thread or topic anchor.  None for top-level messages.
    """

    platform: str
    team_id: str
    user_id: str
    channel_id: str
    thread_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    @property
    def tenant_slug(self) -> str:
        """Deterministic, URL-safe tenant identifier: ``{platform}_{team_id}``.

        This is the canonical tenant key segment used by the S3 skill layout
        shared with hermes-skills-service:
        ``hermes-skills/{tenant_slug}/{scope}/{skill}/SKILL.md``.
        """
        return f"{self.platform}_{self.team_id}"

    @property
    def personal_scope(self) -> str:
        """Most-specific scope — private to this user in this workspace."""
        return f"personal/{self.platform}/{self.team_id}/{self.user_id}"

    @property
    def team_scope(self) -> str:
        """Workspace-level scope — shared across all users in the team."""
        return f"team/{self.platform}/{self.team_id}"

    @property
    def global_scope(self) -> str:
        """Platform-global scope — read-only defaults for all agents."""
        return "global"

    @property
    def scope_chain(self) -> list[str]:
        """
        Resolution order for skills and memory: most-specific first.

        Returns [personal_scope, team_scope, "global"].  Callers iterate
        this list and stop at the first matching resource (CSS specificity model).
        """
        return [self.personal_scope, self.team_scope, self.global_scope]
