"""Skill-declared reasoning effort (issue #108770).

A skill may declare ``metadata.hermes.reasoning_effort`` in its frontmatter to raise (or lower)
the turn's thinking level while it is active; the previous level is restored when the turn ends.
The feature is strictly opt-in: a skill without the key leaves ``agent.reasoning_config``
untouched, so existing behavior is unchanged.

Resolution is per-turn, not per-session: the invoked-skill scaffold that the slash-command
builders put at the head of the user message is the trigger, so nothing is cached on the agent
and no long-lived state can leak into the next turn. This mirrors the cron per-job pin
(``cron.scheduler._resolve_job_reasoning_config``) and reuses the same ladder parsing, so a
declared level is validated by ``hermes_constants.parse_reasoning_effort`` rather than by a
second vocabulary.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

#: ``metadata.hermes`` key a skill declares its reasoning tier under.
SKILL_REASONING_KEY = "reasoning_effort"

#: Turns a declared level may apply to. ``"turn"`` (default) scopes the raise to the invocation
#: turn and restores afterwards; ``"session"`` keeps it for the rest of the session.
VALID_SCOPES: tuple[str, ...] = ("turn", "session")


def _frontmatter_reasoning_effort(frontmatter: dict) -> Optional[object]:
    """Raw ``metadata.hermes.reasoning_effort`` value, or None when undeclared.

    Tolerant of the string-valued form the frontmatter linter accepts (``"high"``) and of an
    explicit mapping (``{effort: high, scope: turn}``), which is how a skill pins a scope.
    """
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    if not isinstance(hermes, dict):
        return None
    value = hermes.get(SKILL_REASONING_KEY)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("effort") or value.get("level") or None
    return value


def _frontmatter_reasoning_scope(frontmatter: dict) -> str:
    """Declared scope for the effort, defaulting to ``"turn"`` (unrecognized values default)."""
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    if not isinstance(hermes, dict):
        return "turn"
    value = hermes.get(SKILL_REASONING_KEY)
    scope = value.get("scope") if isinstance(value, dict) else None
    if scope is None:
        scope = hermes.get("reasoning_scope")
    scope = str(scope or "").strip().lower()
    return scope if scope in VALID_SCOPES else "turn"


def declared_reasoning_effort(frontmatter: dict) -> Optional[dict]:
    """``{"enabled": True, "effort": <level>}`` for a skill's declared tier, or None.

    None covers every "do not change anything" case: no declaration, an unparseable level, and
    an explicit disable that is not a level (``none`` parses to ``{"enabled": False}`` and IS
    returned, since a skill may legitimately ask for thinking off).
    """
    from hermes_constants import parse_reasoning_effort

    raw = _frontmatter_reasoning_effort(frontmatter)
    if raw is None:
        return None
    parsed = parse_reasoning_effort(raw)
    if parsed is None:
        logger.warning(
            "Skill declares an unknown metadata.hermes.reasoning_effort %r; ignoring it "
            "(valid: none, minimal, low, medium, high, xhigh, max, ultra).",
            raw,
        )
        return None
    return parsed


def _embedded_frontmatters(message: str):
    """Frontmatter dicts embedded in a skill scaffold, in the order they appear.

    The scaffold carries each skill's SKILL.md text verbatim — that is the text the model actually
    reads — so the declaration is resolved from the message itself. No second disk read, and no
    window where the file on disk disagrees with what was injected this turn.
    """
    from agent.skill_utils import parse_frontmatter

    for match in re.finditer(r"(?m)^---[ \t]*$", message):
        try:
            frontmatter, _ = parse_frontmatter(message[match.start():])
        except Exception:
            logger.debug("Could not parse embedded skill frontmatter", exc_info=True)
            continue
        if frontmatter:
            yield frontmatter


def declaration_for_message(message) -> Optional[dict]:
    """Resolve the skill-declared reasoning config for a user message, or None.

    None means "this turn does not opt in": the message is not a skill scaffold, no loaded skill
    declares anything, or every declaration is unparseable. The scaffold prefix is the gate, so an
    ordinary message that happens to quote frontmatter is never treated as opt-in.
    """
    if not isinstance(message, str):
        return None
    from agent import skill_commands as sc

    if not message.startswith(sc._SKILL_INVOCATION_PREFIX):
        return None
    for frontmatter in _embedded_frontmatters(message):
        config = declared_reasoning_effort(frontmatter)
        if config is None:
            continue
        return {"config": config, "scope": _frontmatter_reasoning_scope(frontmatter)}
    return None


def apply_skill_reasoning_override(agent, message) -> Optional[dict]:
    """Raise the turn's reasoning tier to a skill's declared level.

    Returns the snapshot to hand back to :func:`restore_skill_reasoning_override`, or None when
    the turn does not opt in (no declaration, or the feature is disabled by config). The previous
    ``agent.reasoning_config`` is preserved verbatim in the snapshot, so the restore is exact.
    """
    declaration = declaration_for_message(message)
    if declaration is None:
        return None
    config = declaration["config"]
    scope = declaration["scope"]
    previous = getattr(agent, "reasoning_config", None)
    agent.reasoning_config = dict(config)
    # A scope="session" declaration must survive the per-turn restore, so it is kept as the
    # restore target: the next turn's turn-end puts the same level back rather than the
    # pre-skill value. Turn scope restores the original immediately.
    restore_to = dict(config) if scope == "session" else (
        dict(previous) if isinstance(previous, dict) else previous
    )
    logger.info(
        "Skill-declared reasoning effort applied for this turn: %s (scope=%s, previous=%s)",
        config, scope, previous,
    )
    return {"previous": restore_to, "applied": config, "scope": scope}


def restore_skill_reasoning_override(agent, snapshot) -> None:
    """Put ``agent.reasoning_config`` back to the snapshot's ``previous`` value (no-op on None)."""
    if not snapshot:
        return
    agent.reasoning_config = snapshot.get("previous")
