"""Pasted Slack message permalinks: attach the linked thread, access-checked.

Opt-in (``slack.resolve_permalinks``). Only links explicitly present in an authorized
triggering message are read, bounded to three and to one time budget, with the event
workspace's bot token. Each link yields the thread root, the linked message and the latest
replies, or a short refusal note when the poster, or anyone in the destination conversation,
could not see the source themselves. Linked content is attacker-controlled: each message
collapses to one inert line and its square brackets go full-width, so it cannot forge a
heading or close the framing markers.

Modules: ``links`` (strict permalink parser, neutralizing helpers), ``access`` (who may see
the source), ``resolve`` (paged thread read, rendering, the adapter hook).
"""

from .access import access_refusal
from .links import PermalinkTarget, parse_permalinks
from .resolve import append_permalink_context, resolve_permalink_context

__all__ = [
    "PermalinkTarget",
    "access_refusal",
    "append_permalink_context",
    "parse_permalinks",
    "resolve_permalink_context",
]
