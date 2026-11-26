"""The stored-instant read convention (``persistence.core.conventions.stored_instant``).

Both backends are bound to it: each decodes its stored row instants through it, pinned below by planting a corrupt cell in the reference's row dict and reading it back through ``SessionRepository.get`` as an epoch float. The trust window and the unset spellings are those of ``hermes_cli.timefmt.coerce_epoch``; the parity case pins the two against each other so they cannot drift apart.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone, UTC

import pytest

from hermes_cli.timefmt import coerce_epoch
from persistence import SessionId
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant
from persistence.conversation.sessions import NewSession

_CELLS = (
    None, "", 0, 0.0, 1_700_000_000, 1_700_000_000.25, "1700000000.5", 4_200_000_000.0, 4_200_000_001.0, -1.0,
    8.4e252, 1e30, math.inf, -math.inf, math.nan, "not-a-timestamp", b"\x00", [1],
    datetime(2026, 9, 23, 12, tzinfo=UTC), datetime(2026, 9, 23, 14, tzinfo=timezone(timedelta(hours=2))),
)


@pytest.mark.parametrize("cell", _CELLS, ids=repr)
def test_stored_instant_trusts_exactly_what_the_legacy_reader_trusts(cell):
    epoch = coerce_epoch(cell)
    instant = stored_instant(cell)
    if epoch is None:
        assert instant is None
    else:
        assert instant is not None and instant.tzinfo is UTC and instant.timestamp() == pytest.approx(epoch, abs=1e-6)


def test_untrusted_cell_warns_with_the_session_unset_cell_stays_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="persistence.core.conventions"):
        assert stored_instant(None, session_id="quiet") is None
        assert stored_instant("", session_id="quiet") is None
        assert stored_instant("garbage", session_id="loud", field="started_at") is None
    messages = [rec.getMessage() for rec in caplog.records]
    assert len(messages) == 1 and "started_at" in messages[0] and "on session loud" in messages[0]


def test_untrusted_instant_is_the_trust_window_floor():
    # The floor is the one instant the CLI renderers treat as unset (relative_time: falsy epoch reads "?").
    assert UNTRUSTED_INSTANT == datetime(1970, 1, 1, tzinfo=UTC) and UNTRUSTED_INSTANT.timestamp() == 0.0


def _plant(backend, scope, session_id: str, **cells) -> None:
    """Write raw cells into the stored row, bypassing every typed write path (how a damaged page or a foreign writer lands them)."""
    backend.scope_store(scope)["sessions"][session_id].update(cells)


def test_a_corrupt_stored_instant_reads_degraded_on_every_backend(backend, scope_a, caplog):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=SessionId("si-bad"), source="cli"))
    store_a.sessions.create(NewSession(session_id=SessionId("si-good"), source="cli"))
    store_a.sessions.end(SessionId("si-good"), "idle")
    _plant(backend, scope_a, "si-bad", started_at="not-a-timestamp", ended_at=8.4e252)
    with caplog.at_level(logging.WARNING, logger="persistence.core.conventions"):
        bad = store_a.sessions.get(SessionId("si-bad"))
        good = store_a.sessions.get(SessionId("si-good"))
    assert bad is not None and bad.started_at == UNTRUSTED_INSTANT.timestamp() and bad.ended_at is None
    assert good is not None and good.started_at > UNTRUSTED_INSTANT.timestamp() and good.ended_at is not None
    warned = [rec.getMessage() for rec in caplog.records]
    assert len(warned) == 2 and all("on session si-bad" in message for message in warned)
