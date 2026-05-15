"""
``@<agent> <message>`` inline routing for the multi-agent gateway.

A user types ``@coder fix this bug`` in any chat.  The gateway routes
that single turn to the ``coder`` agent (profile) without changing the
session's persistent ``active_agent``.  The user's *next* unprefixed
message reverts to whatever the session is bound to via ``/agent``.

The parser is deliberately strict:

* Only the literal ``@`` at message start counts — ``email@host.com``
  references mid-sentence don't trigger routing.
* The target must be a registered agent name (canonicalised via
  ``AgentRegistry``).  Unknown ``@foo`` mentions are left untouched —
  the message proceeds as a normal user message so the agent can
  decide how to interpret it.  This means users can still address
  external people / handles in chat without the gateway eating their
  message.

Wire-up: callers do
    parsed = parse_agent_mention(event.text, registry)
    if parsed.target_agent:
        event.text = parsed.stripped_text
        # run this turn with agent_home_scope(registry.get(...).home)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from gateway.agent_registry import AgentProfile, AgentRegistry


# `^@<name>\s+<rest>` — name limited to the on-disk profile id alphabet.
# Anchored at start so URLs / handles mid-message can't accidentally
# trigger routing.  Non-greedy match for whitespace separator.
_MENTION_RE = re.compile(
    r"^@([a-z0-9][a-z0-9_-]{0,63})\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ParsedMention:
    """Result of parsing a user message for an ``@<agent>`` prefix.

    ``target_agent`` is ``None`` when no mention was found OR when the
    mention referenced an unknown agent — both cases pass the original
    text through untouched.
    """

    target_agent: Optional[AgentProfile]
    stripped_text: str
    raw_mention: Optional[str] = None  # the literal token "@coder" if matched


def parse_agent_mention(text: str, registry: AgentRegistry) -> ParsedMention:
    """Parse ``text`` for a leading ``@<agent>`` route hint.

    Always returns a ``ParsedMention`` — never raises.  Callers can
    treat ``target_agent is None`` as "no routing applied".
    """
    if not isinstance(text, str) or not text:
        return ParsedMention(target_agent=None, stripped_text=text or "")

    # Cheap early-exit: must start with literal '@' after optional whitespace
    # is stripped — but only the leading whitespace, not the body.
    leading_ws_match = re.match(r"^(\s*)(@)", text)
    if not leading_ws_match:
        return ParsedMention(target_agent=None, stripped_text=text)

    leading_ws = leading_ws_match.group(1)
    body = text[len(leading_ws):]

    m = _MENTION_RE.match(body)
    if not m:
        return ParsedMention(target_agent=None, stripped_text=text)

    candidate = m.group(1)
    rest = m.group(2)

    agent = registry.get(candidate)
    if agent is None:
        # Mention syntax matched, but target unknown — pass through.
        # This is the "user wrote @alice in a normal sentence" case.
        return ParsedMention(target_agent=None, stripped_text=text)

    return ParsedMention(
        target_agent=agent,
        stripped_text=rest.strip(),
        raw_mention=f"@{candidate}",
    )
