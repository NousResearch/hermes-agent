"""Which Slack channels a profile's own posts may go to (education design 5b).

Two lists under the profile's top-level ``slack:`` block in ``config.yaml``::

    slack:
      post_allowed_channels: [C0HOME, C0RELEASE]   # the only channels it posts to
      post_gated_channels: [C0RELEASE]            # no free text: row-gated only

- ``post_allowed_channels`` — when set, ``slack_post_thread`` and cron
  delivery reach only these channel ids (prompt-injection guard: read-and-act
  on a web page can only land where the team already reviews). When unset,
  posting is unrestricted as before, EXCEPT on a profile whose Notion write
  guard is enforcing: a guarded profile with no allowlist posts nowhere
  (fail closed — the guard marks it as a profile the boundary applies to).
- ``post_gated_channels`` — always applies, allowlist or not: these channels
  accept no free text. ``slack_post_thread`` takes a Release Assets row id
  for them (``tools/release_update_post.py``); cron delivery never posts there.

Channel ids are compared upper-cased. Platform settings come from the
active profile's config, read strictly through
``agent.boundary_config.read_boundary_config`` (resolves per profile under
multiplex; an unparseable config.yaml never reads as "no policy"), never
``os.getenv``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_CHANNEL_ID = re.compile(r"^[CGD][A-Z0-9]{6,}$")


@dataclass(frozen=True)
class PostPolicy:
    allowed: Optional[frozenset]  # None = unrestricted (legacy)
    gated: frozenset
    error: str = ""


def _channels(raw, key: str) -> Optional[frozenset]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError(f"slack.{key} must be a list of channel ids")
    return frozenset(str(x).strip().upper() for x in raw if str(x).strip())


def post_policy(cfg) -> PostPolicy:
    from agent.boundary_config import is_unreadable, unreadable_message
    if is_unreadable(cfg):
        return PostPolicy(frozenset(), frozenset(), unreadable_message(cfg))
    slack = cfg.get("slack") if isinstance(cfg, dict) and isinstance(cfg.get("slack"), dict) else {}
    try:
        allowed = _channels(slack.get("post_allowed_channels"), "post_allowed_channels")
        gated = _channels(slack.get("post_gated_channels"), "post_gated_channels") or frozenset()
    except ValueError as e:
        return PostPolicy(frozenset(), frozenset(), f"{e} — posting is refused until an admin fixes it.")
    if allowed is None:
        from tools.notion_write_guard import is_enforcing
        if is_enforcing(cfg):
            return PostPolicy(frozenset(), gated,
                              "slack.post_allowed_channels is not configured — this guarded profile "
                              "posts nowhere until an admin lists its channels.")
    return PostPolicy(allowed, gated)


def normalise(chat_id) -> str:
    return str(chat_id or "").strip().upper()


def is_gated(chat_id, cfg) -> bool:
    """True for a gated channel — and for EVERY channel when the config is
    unreadable (fail closed: no free text, no inbound answers)."""
    from agent.boundary_config import is_unreadable
    return is_unreadable(cfg) or normalise(chat_id) in post_policy(cfg).gated


def channel_denial(chat_id, cfg, *, gated_ok: bool = False) -> Optional[str]:
    """Why a post to ``chat_id`` is refused, or None. ``gated_ok`` is for the
    row-gated release-update path only."""
    policy = post_policy(cfg)
    if policy.error:
        return policy.error
    cid = normalise(chat_id)
    if (policy.gated or policy.allowed is not None) and not _CHANNEL_ID.match(cid):
        return (f"{cid or 'that target'} is not a Slack channel id; with a posting policy set, "
                "post only by channel id (names like #release-updates are refused).")
    if cid in policy.gated and not gated_ok:
        return (f"{cid} is a gated channel (slack.post_gated_channels): it accepts no free text. "
                "Post a release update by passing release_asset_row_id instead of message.")
    if policy.allowed is not None and cid not in policy.allowed:
        return (f"{cid} is not in slack.post_allowed_channels for this profile; posting there "
                "is refused.")
    return None
