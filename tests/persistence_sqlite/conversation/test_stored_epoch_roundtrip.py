"""Stored session clocks on SQLite retain the epoch their writers supplied."""

import copy
import math
from unittest.mock import patch

import pytest

from persistence.conversation.sessions import NewSession
from persistence.sqlite.conversation._store.compression import (
    get_compression_recovery_deadline, set_compression_recovery_deadline,
)


@pytest.mark.parametrize("epoch", [0.0, -0.0, 0.0000001, 1.2345678, 1700000000.1234567, 1790639758.6100938])
def test_session_epoch_survives_read_and_copy(backend, scope_a, epoch):
    sessions = backend.open(scope_a).sessions
    with patch("time.time", return_value=epoch):
        sessions.create(NewSession("clock", "cli"))
    record = sessions.get("clock")
    assert record.started_at == epoch
    assert copy.deepcopy(record).started_at == epoch
    assert math.copysign(1, record.started_at) == 1


@pytest.mark.parametrize("epoch", [0.0000001, 1.2345678, 1700000000.1234567, 1790639758.6100938])
def test_recovery_deadline_read_can_be_written_without_rounding(backend, scope_a, epoch):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("clock", "cli"))
    set_compression_recovery_deadline(store.holder(), "clock", epoch)
    deadline = store.lineage.compression_recovery_deadline("clock")
    assert deadline == epoch
    store.lineage.set_compression_recovery_deadline("clock", deadline)
    assert get_compression_recovery_deadline(store.holder(), "clock") == epoch


def test_recovery_deadline_of_a_missing_session_reads_as_unset(backend, scope_a):
    assert backend.open(scope_a).lineage.compression_recovery_deadline("missing") == 0.0
