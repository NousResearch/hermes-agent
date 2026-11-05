"""The legacy store's refusals raised as the contract's error family."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from hermes_state_errors import (
    CompressionSessionBusyError, CompressionSessionClosedError, HeldRowRewrittenError, PruneRowStaleError, PruneRowUnresolvedError, SessionActiveWriteGuardError, SessionCompressionInProgressError, SessionTurnLeaseLostError,
)
from persistence import (
    AuthorityRefusal, CompressionBusy, CompressionClosed, CompressionInProgress, ExportTooLarge, HeldRowRewritten,
    PruneRowStale, PruneRowUnresolved, SessionWriteGuarded, TurnLeaseLost, UnknownExecution,
)
from persistence.conversation.messages import ResumeTooLarge, RewindTargetUnavailable

from .._backend import translated
from .._holder import StoreHolder
from ._store.compression import get_compression_tip
from ._store.core import SessionExportTooLargeError, SessionResumeTooLargeError
from ._store.rewind import RewindTargetUnavailableError

__all__ = ["refusals"]


@contextmanager
def refusals(holder: Optional[StoreHolder] = None, session_id: Optional[str] = None) -> Iterator[None]:
    """Raise the store's failures (``translated``) and its typed refusals as the contract's classes, chaining the
    original. An authority refusal already is the contract's class and passes unchanged, except that a bare
    ``unknown_execution`` becomes ``UnknownExecution``. ``session_id`` names the session a session-level refusal
    reports, and ``holder`` resolves the live continuation a compression-closed refusal carries."""
    sid = session_id or ""
    try:
        with translated():
            yield
    except AuthorityRefusal as exc:
        if type(exc) is AuthorityRefusal and exc.reason == "unknown_execution":
            raise UnknownExecution("session", sid) from exc
        raise
    except SessionCompressionInProgressError as exc:
        raise CompressionInProgress(sid, str(exc)) from exc
    except CompressionSessionBusyError as exc:
        raise CompressionBusy(sid, str(exc)) from exc
    except CompressionSessionClosedError as exc:
        tip = get_compression_tip(holder, exc.session_id) if holder is not None else exc.session_id
        raise CompressionClosed(exc.session_id, tip) from exc
    except SessionTurnLeaseLostError as exc:
        raise TurnLeaseLost(sid, str(exc)) from exc
    except SessionActiveWriteGuardError as exc:
        raise SessionWriteGuarded(sid, str(exc)) from exc
    except HeldRowRewrittenError as exc:
        raise HeldRowRewritten(sid, str(exc)) from exc
    except PruneRowStaleError as exc:
        raise PruneRowStale(sid, str(exc)) from exc
    except PruneRowUnresolvedError as exc:
        raise PruneRowUnresolved(sid, str(exc)) from exc
    except SessionResumeTooLargeError as exc:
        raise ResumeTooLarge(exc.message_count, exc.limit, exc.scope) from exc
    except SessionExportTooLargeError as exc:
        raise ExportTooLarge(exc.session_id, exc.message_count, exc.limit) from exc
    except RewindTargetUnavailableError as exc:
        raise RewindTargetUnavailable(str(exc)) from exc
