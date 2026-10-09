"""Per-board ``kanban.default_assignee`` resolution for the dispatcher.

Split out of ``kanban_db_dispatch`` (already over the file-size ratchet) so the
parent file can only shrink. ``resolve_default_assignee`` is the single seam
every dispatch entry point funnels through (gateway tick, ``hermes kanban
dispatch``, the standalone daemon, dashboard nudge): the board's
``kanban.boards.<slug>.default_assignee`` overrides the global the caller
passed, and an unset board key keeps that caller's value byte-for-byte.
"""

from __future__ import annotations

from typing import Optional

from hermes_cli.kanban_board_settings import effective_setting

# load_config failure modes we tolerate: a missing/unreadable file (OSError),
# a malformed YAML value (ValueError), an import-time problem (ImportError), a
# non-mapping root or missing key (TypeError/KeyError), or a stale attribute
# (AttributeError). A config read must never crash a dispatch tick.
_CONFIG_READ_ERRORS = (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError)


def resolve_default_assignee(
    default_assignee: Optional[str], *, board: Optional[str] = None,
) -> Optional[str]:
    """``kanban.default_assignee`` when it names a real profile this home may
    claim (``kanban.dispatch_profiles`` gated, same predicate as the spawn
    gate). Otherwise ``None`` so an unassigned shared-board card is never
    written to. When the profiles module isn't importable trust the
    operator's config: the downstream check still buckets a missing profile
    as nonspawnable.

    The board's ``kanban.boards.<slug>.default_assignee`` overrides the value
    the caller passed (its global default); a board with no override keeps the
    passed value, so existing callers are byte-for-byte unchanged.
    """
    # Late import: kanban_db_dispatch owns ``_kb`` / ``_profile_exists_fn`` and
    # imports this module at its foot, so a module-level import would be a cycle.
    from hermes_cli import kanban_db_dispatch as dispatch

    try:
        from hermes_cli.config import load_config
        kanban_cfg = (load_config() or {}).get("kanban", {})
    except _CONFIG_READ_ERRORS:
        kanban_cfg = {}
    if not isinstance(kanban_cfg, dict):
        kanban_cfg = {}
    resolved = effective_setting(
        kanban_cfg, "default_assignee", board or dispatch._kb.get_current_board(),
        fallback=default_assignee,
    )
    name = (resolved or "").strip() or None
    if name:
        profile_exists = dispatch._profile_exists_fn()
        if profile_exists is not None and not profile_exists(name):
            return None
    return name
