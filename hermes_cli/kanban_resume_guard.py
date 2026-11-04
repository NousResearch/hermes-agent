"""Centralized resume-time guard for Kanban-owned sessions (#68779) — the ONE place that
decides whether a Kanban worker transcript may resume as an ordinary session.

A Kanban dispatcher worker transcript must never continue as a write-capable ordinary
CLI / Desktop / TUI session: the resumed process carries none of the dispatcher-granted
ownership env (``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` / ``HERMES_KANBAN_BOARD`` /
``HERMES_KANBAN_CLAIM_LOCK``), so the board cannot observe or supervise it — and unblocking
the card can then dispatch a second writer into the same workspace. Every surface that makes
a resumed session WRITE-CAPABLE asks this guard before loading history:

- interactive CLI ``--resume`` / ``-c`` (``HermesCLI._preload_resumed_session``),
- mid-chat ``/resume`` / ``/sessions <id>`` (``_handle_resume_command``),
- quiet one-shot resume (``hermes_cli/oneshot.py``),
- gateway ``session.resume`` — Desktop session history and the TUI picker
  (``tui_gateway/methods_session.py``).

Detection is persisted provenance (``SessionDB.is_kanban_owned_session``: the session row's
``kanban`` source, checked across the compression lineage a resume would materialize) plus
dispatch-time env for the exemption: a process that itself holds ``HERMES_KANBAN_TASK`` is a
dispatcher-owned run — the one legitimate supervised continuation — and may proceed.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _dispatcher_owned_run() -> bool:
    """This process holds a dispatcher-granted Kanban task (``HERMES_KANBAN_TASK``, exactly
    what ``_default_spawn`` injects into worker children). That is the grant boundary: only a
    run the dispatcher owns — task, run id, claim, heartbeat — may continue a worker
    transcript, so it is exempt from the guard."""
    try:
        from gateway.session_context import get_session_env
    except ImportError:  # no gateway context (plain CLI child) — read the process env
        get_session_env = os.environ.get
    return bool(str(get_session_env("HERMES_KANBAN_TASK", "") or "").strip())


def kanban_resume_refusal(db, session_id: str):
    """Human-readable refusal reason when ``session_id`` is a Kanban-owned session that must
    NOT resume as an ordinary write-capable session, or ``None`` when the resume may proceed.

    Fails open ONLY for a store that cannot classify at all (a probe error must never block
    an ordinary session); a POSITIVE kanban-provenance match always refuses. Read failures
    are logged, not surfaced to the user as success.
    """
    if not session_id:
        return None
    if _dispatcher_owned_run():
        return None
    probe = getattr(db, "is_kanban_owned_session", None)
    if probe is None:  # lightweight adaptor DB without the method — cannot classify
        return None
    try:
        owned = probe(session_id)
    except Exception:
        # Deliberate fail-open boundary: an ordinary session must never be blocked by a
        # broken probe, so classify as unknown and log the full traceback for diagnosis.
        logger.warning("kanban resume probe failed for %s (proceeding unclassified)",
                       session_id, exc_info=True)
        return None
    # Strict True: SessionDB answers a real bool; anything else (a test double, an adaptor
    # quirk) is treated as unclassified, keeping the guard open for every non-worker store.
    if owned is not True:
        return None
    from agent.i18n import t
    return t("cli.resume.kanban_owned", session_id=session_id)
