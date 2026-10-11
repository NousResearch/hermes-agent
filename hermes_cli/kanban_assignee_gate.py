"""Ghost-assignee gate for the agent ingress surfaces (Kanban).

``hermes_cli.kanban_db.create_task``/``assign_task`` only lowercase-normalize
their assignee, so any ingress could park a card on a profile nobody
installed; the dispatcher then refuses those cards every tick as
``skipped_nonspawnable`` and the card rusts (real case: 20.4h on a ghost
created via the ``kanban_create`` tool by an orchestration).

This gate extends the reviewer-existence contract (#106163) to card
creation and (re)assignment: the tool + CLI ingress surfaces call
:func:`validate_assignee_exists` so a ghost dies — with the installed
roster in the message — before any row is written.

Accepted: ``None`` (unassigned), ``default``, live on-disk profiles
(tombstoned dirs are refused), and ``bot_peers`` entries (cross-host
workers served by a remote gateway — ``ON DISK no`` locally, by design).
Operators can force a known-good ghost for seeding/CI by setting
``kanban.allow_any_assignee: true`` in config.yaml.
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger(__name__)


def validate_assignee_exists(assignee: Optional[str]) -> Optional[str]:
    """Canonicalize *assignee* and refuse profiles this host can never dispatch.

    Ingress-side guard only: the DB layer (``create_task``/``assign_task``)
    stays permissive for restore flows and graph builders, and the
    dispatcher-side ``profile_exists`` checks remain the last line of defense.
    """
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name, profile_exists

    canonical = normalize_profile_name(assignee)
    if canonical == "default" or profile_exists(canonical):
        return canonical
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except OSError as exc:
        _log.warning(
            "kanban: config unreadable (%s); bot_peers and allow_any_assignee "
            "unavailable — refusing ghost assignee %r", exc, canonical,
        )
        cfg = {}
    if isinstance(cfg, dict) and isinstance(cfg.get("kanban"), dict) \
            and cfg["kanban"].get("allow_any_assignee") is True:
        return canonical
    peers = cfg.get("bot_peers") if isinstance(cfg, dict) else None
    if isinstance(peers, dict) and canonical in peers:
        return canonical
    from hermes_cli.profiles import list_profile_names

    roster = ", ".join(list_profile_names())
    raise ValueError(
        f"assignee {assignee!r} is not a profile this host can dispatch "
        f"(not on disk and not a bot_peer). Installed profiles: {roster}. "
        "Assign an installed profile, register the worker as a bot_peer in "
        "config.yaml, or set kanban.allow_any_assignee: true (break-glass)."
    )