"""``/handoff``: transfer this CLI session to a gateway platform; a base of ``CLICommandsMixin``.

``cli.py``-internal symbols are imported at call time via ``from cli import ...`` (cli.py imports
``cli_commands_mixin``, which imports this module).
"""

from __future__ import annotations

import time
from contextlib import suppress

from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _t


def _db_unavailable_line() -> str:
    from storage_user_copy import format_session_db_unavailable
    return f"  {format_session_db_unavailable(details=True)}"


def _local_handoff_store():
    """A writer handle on the current home's store, or None when it cannot open."""
    with suppress(Exception):
        from persistence_assembly import open_persistence, profile_scope
        return open_persistence(profile_scope(), eager=True)
    return None


class CLIHandoffMixin:
    """``/handoff <platform>`` for ``HermesCLI``."""

    _HANDOFF_PENDING_TIMEOUT = 60.0
    _HANDOFF_RUNNING_TIMEOUT = 900.0  # full synthetic agent turn + delivery
    _HANDOFF_HEARTBEAT_EVERY = 30.0

    @staticmethod
    def _handoff_keep(*lines: str) -> bool:
        """Print ``lines`` and keep the CLI session (the True verdict of /handoff)."""
        _cp(*lines)
        return True

    def _handle_handoff_command(self, cmd_original: str) -> bool:
        """Handle ``/handoff <platform>`` — transfer this CLI session to a gateway platform.

        Validate target → prepare session row → mark pending → block-poll (see ``_handoff_wait``).
        Returns False only on ``completed`` (caller exits like /quit); True keeps the session."""
        platform_name = _command_arg(cmd_original).lower()
        if not platform_name:
            return self._handoff_keep(
                f"  {_t('handoff.usage')}", f"  {_t('handoff.usage_detail_1')}",
                f"  {_t('handoff.usage_detail_2')}")
        home = self._handoff_validate_target(platform_name)
        if home is None:
            return True
        # Without the gateway session store the handoff runs on a local writer handle.
        store = None if self._session_db else _local_handoff_store()
        try:
            session_title = self._handoff_prepare_session(store)
            if session_title is None:
                return True
            requested = (store.coordination.request_handoff(session_id=self.session_id, platform=platform_name)
                         if store is not None else self._session_db.request_handoff(self.session_id, platform_name))
            if not requested:
                return self._handoff_keep(f"  {_t('handoff.already_in_flight')}")
            _cp(f"  {_t('handoff.queued', title=session_title, platform=platform_name, home=home.name)}",
                f"  {_t('handoff.waiting')}")
            return self._handoff_wait(platform_name, session_title, store)
        finally:
            if store is not None:
                store.close()

    def _handoff_validate_target(self, platform_name: str):
        """Resolve the destination home channel via the live gateway config; None (after printing
        the reason) when the platform is unknown, disabled, or has no home channel. UX pre-check
        only: the gateway watcher re-checks before dispatch."""
        from gateway.handoff_target import HandoffTargetError, resolve_handoff_home
        try:
            return resolve_handoff_home(platform_name)
        except HandoffTargetError as exc:
            if exc.reason == 'no_home_channel':
                return _cp(f"  {_t('handoff.no_home_channel', platform=platform_name)}",
                           f"  {_t('handoff.set_home_hint')}")
            return _cp(f"  {_t('handoff.' + exc.reason, platform=platform_name)}")
        except Exception as exc:
            return _cp(f"  {_t('handoff.config_load_failed', error=exc)}")

    def _handoff_prepare_session(self, store=None):
        """Refuse mid-turn, offer a session with no row the handoff title on ``store`` (a local writer
        handle) or, without one, on ``self._session_db``, and return the display title (None after
        printing why the handoff cannot start)."""
        # An in-flight agent run would race the gateway's switch_session and the synthetic turn.
        if getattr(self, "_agent_running", False):
            return _cp(f"  {_t('shared.agent_busy', command='/handoff')}")
        if store is None and not self._session_db:
            return _cp(_db_unavailable_line())
        # The gateway switches onto the session's row; a session without one is offered the handoff
        # title (setting a title writes nothing when no row exists).
        title = f"handoff-{self.session_id[:8]}"
        try:
            if store is not None:
                if store.sessions.get(self.session_id) is None:
                    store.sessions.set_title(self.session_id, title)
            elif not self._session_db.get_session(self.session_id):
                self._session_db.set_session_title(self.session_id, title)
        except Exception as exc:  # health: allow BLE001 -- the failure is printed and the handoff does not start
            return _cp(f"  {_t('handoff.session_row_failed', error=exc)}")
        session_title = ""
        with suppress(Exception):
            session_title = ((store.sessions.get_title(self.session_id) if store is not None
                              else (self._session_db.get_session(self.session_id) or {}).get("title")) or "")
        return session_title or self.session_id[:8]

    def _handoff_wait(self, platform_name: str, session_title: str, store=None, read_state=None,
                      fail_pending=None) -> bool:
        """Two-phase 0.5s poll. PENDING (unclaimed): 60s, then CAS-fail the row so the user can
        retry (a claim racing this instant wins). RUNNING (claimed): the gateway replays the
        transcript via a synthetic turn (routinely >60s) — wait 15 min with heartbeats and on
        timeout do NOT touch the row; failing it here was the split-brain bug.

        ``read_state`` / ``fail_pending`` read and CAS-fail the row; by default ``store`` (a local
        writer handle) or, without one, this process's own store, the shared gateway's
        ``handoff.state`` / ``handoff.fail`` when it owns the session."""
        if read_state is None:
            from hermes_cli.gateway_session_store import SurfaceLabelReader
            read_state = ((lambda: SurfaceLabelReader(store).get_handoff_state(self.session_id)) if store is not None
                          else (lambda: self._session_db.get_handoff_state(self.session_id)))
        fail_pending = fail_pending or (lambda reason: self._handoff_fail_pending(reason, store))
        pending_deadline = time.time() + self._HANDOFF_PENDING_TIMEOUT
        running_deadline = None
        next_heartbeat = None
        last_state = "pending"
        recovered = None  # the row read right after a pending timeout's CAS-fail
        while True:
            try:
                state_row, recovered = (recovered or read_state()), None
            except Exception:  # health: allow BLE001 -- an unreadable handoff state polls again as pending
                state_row = None
            current = (state_row or {}).get("state") or "pending"
            if current != last_state:
                if current == "running":
                    _cp(f"  {_t('handoff.picked_up')}")
                    running_deadline = time.time() + self._HANDOFF_RUNNING_TIMEOUT
                    next_heartbeat = time.time() + self._HANDOFF_HEARTBEAT_EVERY
                last_state = current
            if current == "completed":
                _cp("", f"  {_t('handoff.complete', platform=platform_name)}",
                    f"  {_t('handoff.resume_hint', title=session_title)}", "")
                # _run_cleanup must NOT finalize the row on exit: the gateway owns it now, and an
                # end_reason set under it would drop the handoff leg from session history/search.
                # See #88234.
                from cli import _handed_off_session_ids
                _handed_off_session_ids.add(self.session_id)
                self._should_exit = True  # same exit semantics as /quit
                return False
            if current == "failed":
                err = (state_row or {}).get("error") or _t("handoff.unknown_error")
                return self._handoff_keep(
                    f"  {_t('handoff.failed', error=err)}", f"  {_t('handoff.session_intact_retry')}")
            now = time.time()
            if current == "pending":
                if now >= pending_deadline:
                    with suppress(Exception):  # unclaimed: CAS-clear so the user can retry
                        fail_pending("timed out waiting for gateway")
                    # A claim that won the race goes through the branches above (running starts
                    # its own deadline); our own failed row, or an unreadable one, is the timeout.
                    with suppress(Exception):
                        recovered = read_state()
                    if ((recovered or {}).get("state") or "") in ("running", "completed"):
                        continue
                    break
            else:  # running
                if next_heartbeat is not None and now >= next_heartbeat:
                    _cp(f"  {_t('handoff.still_transferring')}")
                    next_heartbeat = now + self._HANDOFF_HEARTBEAT_EVERY
                if running_deadline is not None and now >= running_deadline:
                    # Do NOT fail the row: the gateway owns it (split-brain bug otherwise).
                    return self._handoff_keep(
                        f"  {_t('handoff.taking_long')}",
                        f"  {_t('handoff.check_platform', platform=platform_name)}",
                        f"  {_t('handoff.no_longer_waiting')}", f"  {_t('handoff.retry_when_settled')}")
            time.sleep(0.5)
        return self._handoff_keep(f"  {_t('handoff.timed_out')}", f"  {_t('handoff.session_intact')}")

    def _handoff_fail_pending(self, reason: str, store=None) -> None:
        if store is not None:
            from persistence.authority.coordination import HandoffStatus
            store.coordination.fail_handoff(session_id=self.session_id, error=reason,
                                            only_states=(HandoffStatus.PENDING,))
            return
        try:
            self._session_db.fail_handoff(self.session_id, reason, only_states=("pending",))
        except TypeError:
            # Older SessionDB without only_states (mixed installs): legacy unconditional fail.
            self._session_db.fail_handoff(self.session_id, reason)
