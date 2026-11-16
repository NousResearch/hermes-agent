"""A profile owner that is restarting: which refusals mean "ask the successor", and the wait for it.

Shared by every client that must outlive an owner restart instead of failing on it: the Kanban
worker (``kanban_worker_client``) and the classic CLI's session open (``cli_init_mixin``).
"""
from __future__ import annotations

import threading
import time

from hermes_cli.gateway_client import DISCONNECTED, NOT_SENT, GatewayClientError, GatewayRPCError, GatewayUnavailableError

# In-band owner refusals that leave the request intact: a draining owner admits nothing new, and a
# retired authority's successor holds the admission the next connection reads.
OWNER_RESTART_REASONS = frozenset({'runtime_draining', 'runtime_retired'})
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# Control-socket answers a restarting owner gives (an ``inaccessible`` owner's reason, from
# discovery or the ticket fetch): a reset or half-written answer, or a replaced owner refusing a
# stale instance. Unsafe nodes or peers, a refused authorization, a bad pointer or a failed start
# stay final.
_RESTART_CONTROL_REASONS = frozenset({'invalid_control_peer', 'incomplete_control_response',
                                      'invalid_control_response'})


def owner_restarting(exc) -> bool:
    """*exc* is a draining or retired owner's own answer: an RPC refusal with a restart reason, or
    no attachable owner because the one there is draining. Nothing was admitted."""
    if isinstance(exc, GatewayRPCError):
        return str(exc) in OWNER_RESTART_REASONS
    return isinstance(exc, GatewayUnavailableError) and exc.state == 'draining'


def restartable(exc) -> bool:
    """False when *exc* is the owner's own final answer; True for a restarting owner or a lost transport."""
    if isinstance(exc, GatewayRPCError):
        return str(exc) in OWNER_RESTART_REASONS
    if isinstance(exc, GatewayUnavailableError):
        return exc.state in _TRANSIENT_OWNER_STATES or exc.reason_code in _RESTART_CONTROL_REASONS
    return True


def _transport_lost(exc) -> bool:
    """A socket that dropped or never dialled (the old owner exiting, the successor not up yet), as
    opposed to an owner's final refusal carried in a plain ``GatewayClientError``. Only the
    connection's own failures count: a local error (a deleted launch directory's ``os.getcwd()``)
    is raised at once, never waited out as a restart."""
    if isinstance(exc, (GatewayRPCError, GatewayUnavailableError)):
        return restartable(exc)
    if type(exc) is not GatewayClientError:
        return False
    text = str(exc)
    return text in (DISCONNECTED, NOT_SENT) or text.startswith(("Gateway connection", "Gateway did not answer"))


class SuccessorWaitStopped(GatewayClientError):
    """The user stopped waiting for the restarted owner (Ctrl+C); nothing was admitted."""


def restart_wait_budget() -> float:
    """Seconds an owner restart may take: its after-turn wait, then its stop envelope (the CLI's
    ``hermes gateway restart`` waits the same)."""
    from hermes_cli.gateway import _get_restart_exit_wait_budget
    return _get_restart_exit_wait_budget()


def wait_for_successor(attempt, *, announce, wake=None, budget=None, clock=time.monotonic):
    """``attempt(remaining)``, retried while its owner restarts (``remaining``: seconds left in the
    wait, ``None`` before it starts). The first restart refusal calls ``announce()`` once; from then
    on any restartable failure (the old owner exiting, the successor starting) is retried with
    backoff until ``budget`` seconds (default :func:`restart_wait_budget`) pass, then a
    ``GatewayClientError`` saying the owner did not come back is raised (never the bare reason code).
    ``wake`` (a ``threading.Event`` Ctrl+C sets) stops the wait with :class:`SuccessorWaitStopped`,
    checked before every attempt and polled by the attempt's own waits (:class:`CallStopped`).
    Any other failure is raised at once."""
    from agent.i18n import t
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_connection import CallStopped
    wake = wake if wake is not None else threading.Event()
    deadline = None
    failures = 0
    while True:
        if wake.is_set():
            raise SuccessorWaitStopped(t("cli.gateway_restart.stopped"))
        try:
            return attempt(None if deadline is None else max(deadline - clock(), 0.0))
        except CallStopped:
            raise SuccessorWaitStopped(t("cli.gateway_restart.stopped")) from None
        except Exception as exc:  # health: allow BLE001 -- classified below; anything not a restart is re-raised
            if deadline is None:
                if not owner_restarting(exc):
                    raise
                announce()
                deadline = clock() + (restart_wait_budget() if budget is None else budget)
            elif not _transport_lost(exc):
                raise
            if clock() >= deadline:
                raise GatewayClientError(t("cli.gateway_restart.timed_out")) from exc
        failures += 1
        if wake.wait(min(jittered_backoff(failures, base_delay=0.5, max_delay=5.0, jitter_ratio=0.5),
                         max(deadline - clock(), 0.0))):
            raise SuccessorWaitStopped(t("cli.gateway_restart.stopped"))


def _fresh_connection(store):
    """A connection whose first dial failed never dials again (``GatewayConnection.final``): the
    retry after a refused dial (the old owner exiting) needs a new one, keeping this CLI's launch key."""
    from hermes_cli.gateway_connection import GatewayConnection
    old = store.connection
    if not isinstance(old, GatewayConnection) or not old.final:
        return
    new = GatewayConnection.shared() if old is GatewayConnection._shared else GatewayConnection(connect=old._connect)
    new.launch_api_key, new.launch_api_key_typed = old.launch_api_key, old.launch_api_key_typed
    store.connection = new


def _announce(text):
    """One line, before the prompt_toolkit app runs (stderr: ``-q`` keeps stdout for the answer) or
    through its renderer once it does."""
    import sys
    try:
        from prompt_toolkit.application import get_app_or_none
        running = getattr(get_app_or_none(), "_is_running", False)
    except Exception:  # health: allow BLE001 -- no prompt_toolkit app: plain stderr
        running = False
    if running:
        from cli import _cprint
        _cprint(text)
    else:
        print(text, file=sys.stderr, flush=True)


def through_owner_restart(cli, store, open_):
    """``open_()`` (a session create/resume/branch on *store*), waiting out an owner restart: one plain
    line, then retried on the successor. ``store.retry`` holds one dict kept across every attempt, so
    a retry re-sends the first attempt's exact request (its ``request_id``) and the owner answers with
    what that attempt committed. Ctrl+C (the CLI's ``_gateway_command_wait`` closer, or a
    KeyboardInterrupt at startup) stops the wait: the dial runs in short slices and every answer wait
    polls it, so nothing is opened or switched to after it."""
    from agent.i18n import t
    wake = threading.Event()
    announced = []
    prepared: dict = {}

    def attempt(remaining):
        _fresh_connection(store)
        start = getattr(store.connection, "start", None)
        if remaining is not None and callable(start):
            # A sliced dial: its timeout is a transport loss the loop retries after checking ``wake``.
            start(timeout=min(remaining, 2.0))
        return open_()

    def announce():
        announced.append(True)
        cli._gateway_command_wait = wake  # Ctrl+C in the running app sets it (cli_tui_mixin)
        _announce(t("cli.gateway_restart.waiting"))
    store.retry = (prepared, wake.is_set)
    try:
        return wait_for_successor(attempt, announce=announce, wake=wake)
    except KeyboardInterrupt:  # Ctrl+C before the app runs (startup): the same clean stop
        if not announced:
            raise
        raise SuccessorWaitStopped(t("cli.gateway_restart.stopped")) from None
    finally:
        store.retry = None
        if announced:
            cli._gateway_command_wait = None


def exit_on_stopped_wait(cli, exc) -> None:
    """A one-shot (``-q``) run whose user stopped waiting for the restarted owner exits 130 with the
    plain stopped line: a cancel, not the agent-init failure (exit 1) its caller would report.
    Interactive startup handles the same stop itself (``_tui_print_startup``)."""
    if isinstance(exc, SuccessorWaitStopped) and getattr(cli, "_single_query_mode", False):
        import sys
        from hermes_cli.quiet_single_query import exit_single_query
        print(str(exc), file=sys.stderr, flush=True)
        exit_single_query(130)
