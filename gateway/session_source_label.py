"""The stored ``source`` label a local session's creator asked for (``session.create`` ``source_label``).

``source`` picks the session policy from a closed set of surfaces (``session_policy.SURFACES``). A script
that tags its runs for filtering (``hermes -z ... --source heartbeat``) needs a label outside that set
without changing the policy, so the label rides separately: the session runs as its ``source`` surface
(``oneshot`` for a finite run), and only the sessions-row ``source`` column carries the label, at creation
and on every per-turn peer refresh (``session_a2a.storage_source``). ``created_source`` keeps the surface.

It rides the frozen creation request (``LocalSessionPolicy.request_json``) as sent, like ``client`` and
``row_cwd``: no new column, receipt key or policy field, and it is part of the exact-retry identity.
Labels that another reader already gives a meaning to (a surface, a producer, a messaging platform) are
refused: a row labelled ``telegram`` would be recovered as a Telegram lane.
"""
from __future__ import annotations

import json
import re

_LABEL = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
# Producer and routing labels a reader keys behaviour on, beyond the surfaces and messaging platforms.
_RESERVED = frozenset({'local', 'unknown', 'desktop', 'cron', 'kanban', 'a2a', 'bot_room', 'api', 'api_server',
                       'subagent', 'batch', 'webhook'})


def source_label_problem(value) -> str | None:
    """Why *value* is not a creatable row label (``None`` when it is one)."""
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        return 'must be 1-64 characters of a-z, 0-9, ".", "_" or "-", starting with a letter or digit'
    from gateway.session_policy import SURFACES
    from gateway.session_route_recovery import _messaging_platform
    if value in SURFACES or value in _RESERVED or _messaging_platform(value) is not None:
        return f'{value!r} is a reserved source; pass it as source, or pick another label'
    return None


def requested_source_label(request_json) -> str | None:
    """The label a frozen creation request carries, or ``None`` (none sent, or not a valid label)."""
    try:
        value = json.loads(request_json or '{}').get('source_label')
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value is not None and source_label_problem(value) is None else None
