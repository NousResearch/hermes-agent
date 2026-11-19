"""Refuse a claim whose assigned profile names no model before the dispatcher launches its worker."""
import logging

logger = logging.getLogger(__name__)


def refuse_unresolved_model(task, workspace: str, board: str | None) -> None:
    """Raise (the caller records ``spawn_failed``) when the owner would refuse the card for a missing
    model, with the reason in the task's worker log. Reads the assignee's config only: no credential
    or provider is probed, and any other policy problem is left to the owner as before."""
    from gateway.run import _load_gateway_config
    from gateway.session_kanban import kanban_launch_policy
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    from persistence.core.errors import AuthorityRefusal

    try:
        home = resolve_profile_env(normalize_profile_name(task.assignee))
    except FileNotFoundError:
        return  # no profile dir: the worker client resolves it (and refuses) itself
    try:
        with dispatch._worker_profile_scope(home):
            kanban_launch_policy(task, workspace, _load_gateway_config())
        return
    except AuthorityRefusal as exc:
        if exc.field != 'model':
            return
        message = exc.detail or exc.reason
    except Exception:  # health: allow BLE001 -- the owner builds and refuses this policy itself; logged
        logger.debug("kanban model preflight for %s skipped", task.id, exc_info=True)
        return
    with dispatch._open_worker_log(task, board) as log:
        log.write(f"Kanban worker refused: {message}\n".encode())
    raise RuntimeError(message)
