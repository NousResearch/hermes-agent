"""Discord collaborative-workspace identity, visibility & permission policy.

This module implements the config-driven half of the Discord collaborative
workspace policy: a mapping of Discord user IDs to display names and roles
(owner / collaborator / custom), plus helpers that surface

  * **speaker attribution** — a per-message prefix so a shared session shows
    who is speaking, with their role, id and channel visibility, and
  * **a permission policy** — prompt guidance that lets collaborators request
    low-risk work (research, drafts, tasks) but routes sensitive actions
    (external/paid/destructive/security/prod/customer/legal-finance/config/
    memory-policy) through owner approval.

Everything degrades to prior behavior when no Discord identity config is
present: :func:`resolve_discord_identity` returns ``None``,
:func:`discord_speaker_prefix` returns ``None`` (callers fall back to the
existing ``[name]`` multi-user prefix), and :func:`build_collab_policy_block`
returns ``""``.  So unconfigured / single-user deployments are unchanged.

Config lives under the Discord platform's ``extra`` block in ``config.yaml``::

    platforms:
      discord:
        enabled: true
        extra:
          identities:
            - id: "111111111111111111"   # Discord user ID (snowflake)
              name: Jayden
              role: owner
            - id: "222222222222222222"
              name: Peter
              role: collaborator

``id`` (alias ``user_id``), ``name`` (alias ``display_name``) and ``role``
are the recognised keys.  ``role`` defaults to ``collaborator`` and is
normalised to lowercase (stripped of surrounding whitespace).  Only ``owner``
gets owner privileges; any other value — including custom roles — is kept (in
lowercase form) and rendered verbatim for attribution, but treated as a
non-owner.

Roster + visibility are derived from config and channel type.  Callers should
only render the roster for trusted/configured senders; in shared channels with
unknown participants the policy block may be deliberately omitted to avoid
exposing configured collaborator details.  The variable, per-turn part (who is
speaking *this* message) rides on the user-message prefix, which lives in the
message tier and never changes the system-prompt block by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

OWNER_ROLE = "owner"
COLLABORATOR_ROLE = "collaborator"


@dataclass(frozen=True)
class DiscordIdentity:
    """A configured Discord participant."""

    user_id: str
    display_name: str
    role: str = COLLABORATOR_ROLE

    @property
    def is_owner(self) -> bool:
        return self.role == OWNER_ROLE


def _identity_entries(config: Any) -> List[Dict[str, Any]]:
    """Pull the raw identity list from ``platforms.discord.extra``.

    Returns an empty list (feature off) for any shape we don't recognise so a
    malformed config can never raise into prompt construction.
    """
    try:
        # Import locally to avoid a hard import cycle (config <-> session).
        from .config import Platform

        platforms = getattr(config, "platforms", None) or {}
        plat = platforms.get(Platform.DISCORD)
        if plat is None:
            return []
        extra = getattr(plat, "extra", None) or {}
        raw = extra.get("identities")
        if isinstance(raw, list):
            return [e for e in raw if isinstance(e, dict)]
    except Exception:
        pass
    return []


def load_discord_identities(config: Any) -> Dict[str, DiscordIdentity]:
    """Build a ``{user_id: DiscordIdentity}`` map from config.

    Owners sort first in :func:`identities_roster`; this map is keyed by id for
    O(1) resolution.  Entries without a usable id are skipped.
    """
    out: Dict[str, DiscordIdentity] = {}
    for entry in _identity_entries(config):
        uid = entry.get("id", entry.get("user_id"))
        if uid is None or str(uid).strip() == "":
            continue
        uid = str(uid)
        name = entry.get("name", entry.get("display_name")) or uid
        role = str(entry.get("role", COLLABORATOR_ROLE) or COLLABORATOR_ROLE).strip().lower()
        out[uid] = DiscordIdentity(user_id=uid, display_name=str(name), role=role)
    return out


def identities_roster(config: Any) -> List[DiscordIdentity]:
    """Configured identities, owners first then by display name — stable order.

    Deterministic ordering keeps the rendered roster identical across turns so
    it stays cache-stable in the system prompt.
    """
    identities = list(load_discord_identities(config).values())
    identities.sort(key=lambda i: (not i.is_owner, i.display_name.lower(), i.user_id))
    return identities


def resolve_discord_identity(config: Any, user_id: Optional[str]) -> Optional[DiscordIdentity]:
    """Resolve a Discord user id to its configured identity, or ``None``."""
    if not user_id:
        return None
    return load_discord_identities(config).get(str(user_id))


def visibility_label(source: Any) -> str:
    """Channel visibility for a source: ``"dm"`` (private) or ``"shared_channel"``.

    Anything that isn't a one-on-one DM is treated as shared, since other
    configured collaborators may read it.
    """
    return "dm" if getattr(source, "chat_type", None) == "dm" else "shared_channel"


def discord_speaker_prefix(source: Any, config: Any) -> Optional[str]:
    """Per-message speaker-attribution prefix for a Discord message.

    Returns ``[name · role · id:<id> · <visibility>]`` when the sender resolves
    to a configured identity, else ``None`` so the caller keeps its existing
    behavior (plain ``[name]`` prefix in shared sessions, nothing in DMs).
    """
    identity = resolve_discord_identity(config, getattr(source, "user_id", None))
    if identity is None:
        return None
    vis = visibility_label(source)
    return f"[{identity.display_name} · {identity.role} · id:{identity.user_id} · {vis}]"


def build_collab_policy_block(identities: List[DiscordIdentity], visibility: str) -> str:
    """Render the system-prompt guidance block for the shared workspace.

    Covers the participant roster, DM-vs-shared visibility isolation, speaker
    attribution, and the owner-approval permission policy.  Returns ``""`` when
    no identities are configured so callers can append unconditionally.
    """
    if not identities:
        return ""

    owners = [i for i in identities if i.is_owner]
    owner_names = ", ".join(i.display_name for i in owners) or "the owner"

    lines: List[str] = ["", "**Discord collaborative workspace:**"]
    for ident in identities:
        lines.append(
            f"  - {ident.display_name} — {ident.role} (id `{ident.user_id}`)"
        )

    lines.append(
        "Each user message is prefixed with "
        "`[name · role · id:<id> · visibility]` so you always know who is "
        "speaking and whether the message is private or shared."
    )

    if visibility == "dm":
        lines.append(
            "**Visibility — DM (private):** This conversation is a private DM "
            "and is NOT visible to other collaborators. Do not reveal another "
            "person's DM content here, and do not assume shared-channel context "
            "carries into this private thread."
        )
    else:
        lines.append(
            "**Visibility — shared channel:** Every configured collaborator can "
            "read this channel. Anything you say is visible to all of them, so "
            "do not surface content that was shared with you privately in a DM."
        )

    lines.append(
        f"**Permission policy:** Collaborators may directly request low-risk "
        f"work — research, drafts, summaries, and routine task tracking. "
        f"Actions that are external/outbound, paid, destructive, "
        f"security-sensitive, production-affecting, customer-facing, "
        f"legal/financial, or that change configuration or memory/policy "
        f"require {owner_names}'s (owner) explicit approval. When a "
        f"collaborator requests such an action, prepare it and ask for owner "
        f"approval before executing rather than refusing outright."
    )

    return "\n".join(lines)
