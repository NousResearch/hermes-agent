"""The canonical conversation and user receipt for restored pending input."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gateway.input_owner import gateway_input_owner
from gateway.platforms.event import MessageEvent
from gateway.session_identity import identity_of
from gateway.session_transcript import TranscriptReadError

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


@dataclass(frozen=True)
class PendingExecutionOwner:
    home: Path
    session_key: str
    session_id: str
    owner: str

    def current(
        self, runner: GatewayRunner, event: MessageEvent, session_key: str
    ) -> bool:
        from hermes_constants import get_hermes_home

        try:
            identity = identity_of(event.source)
            entry = runner.session_store.lookup_by_session_key(session_key)
            if (
                session_key != self.session_key
                or get_hermes_home().resolve() != self.home
                or identity is None
                or identity.runtime_home.resolve() != self.home
                or entry is None
                or entry.suspended
                or gateway_input_owner(event, event.source) != self.owner
            ):
                return False
            adapter = identity.adapter()
            if (
                adapter is None
                or not adapter.is_connected
                or runner._adapters_for_profile(identity.transport_profile).get(
                    event.source.platform
                )
                is not adapter
            ):
                return False
            db = runner.session_store._db_for_session_id(self.session_id)
            return (
                Path(db.db_path).resolve().parent == self.home
                and (db.get_compression_tip(self.session_id) or self.session_id)
                == entry.session_id
                and not runner.session_store.has_input_owner(
                    self.session_id, self.owner
                )
            )
        except (OSError, ValueError, KeyError, TranscriptReadError):
            return False


def pending_execution_current(
    runner: GatewayRunner, event: MessageEvent, session_key: str
) -> bool:
    owner = getattr(event, "_pending_execution_owner", None)
    return (
        owner is None
        or isinstance(owner, PendingExecutionOwner)
        and owner.current(runner, event, session_key)
    )


def consume_pending_execution(runner: GatewayRunner, event: MessageEvent) -> None:
    owner = getattr(event, "_pending_execution_owner", None)
    if not isinstance(owner, PendingExecutionOwner):
        return
    from gateway.run import _profile_runtime_scope
    from gateway.shutdown_recovery import consume_executed_pending

    try:
        with _profile_runtime_scope(owner.home):
            consume_executed_pending(runner.session_store)
    except (OSError, ValueError, KeyError, TranscriptReadError):
        logger.warning(
            "Could not inspect pending execution in %s; preserving its records",
            owner.home,
            exc_info=True,
        )
