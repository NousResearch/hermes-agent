"""Delivery gap guards must not excuse failures owned by a strict acceptance run."""

import pytest

from tests.e2e.core._pending_fixes import STRICT_ACCEPTANCE_ENV
from tests.e2e.core.delivery._pending_fixes import known_failure


@pytest.mark.parametrize(
    ("strict", "reason", "refuse_gap"),
    [
        (None, "upd-txn: tracked gap", False),
        ("", "upd-txn: tracked gap", False),
        ("0", "upd-txn: tracked gap", False),
        ("other-batch", "upd-txn: tracked gap", False),
        ("upd-tx", "upd-txn: tracked gap", False),
        ("upd-txn", "upd-txn-other: tracked gap", False),
        ("upd-txn", "other-batch: tracked gap", False),
        ("upd-txn", "upd-txn: tracked gap", True),
        (" other-batch, 0, upd-txn ", "upd-txn: tracked gap", True),
        ("1", "other-batch: tracked gap", True),
    ],
)
def test_delivery_gap_respects_strict_owner_scope(monkeypatch, strict, reason, refuse_gap):
    if strict is None:
        monkeypatch.delenv(STRICT_ACCEPTANCE_ENV, raising=False)
    else:
        monkeypatch.setenv(STRICT_ACCEPTANCE_ENV, strict)
    callbacks = []
    gap = AssertionError("marker survived the update")
    expected = AssertionError if refuse_gap else pytest.xfail.Exception

    # Catch both outcomes so an unexpected xfail cannot excuse this regression test.
    with pytest.raises((AssertionError, pytest.xfail.Exception)) as caught:
        with known_failure("marker survived", reason, on_xfail=lambda: callbacks.append("xfail")):
            raise gap

    assert isinstance(caught.value, expected)
    if refuse_gap:
        assert caught.value is gap
        assert callbacks == []
    else:
        assert reason in str(caught.value)
        assert callbacks == ["xfail"]


@pytest.mark.parametrize("strict", ["", "upd-txn", "1"])
@pytest.mark.parametrize("failure", [None, AssertionError("unrelated failure"), ValueError("marker survived")])
def test_delivery_guard_preserves_clean_passes_and_unrelated_errors(monkeypatch, strict, failure):
    monkeypatch.setenv(STRICT_ACCEPTANCE_ENV, strict)
    callbacks = []

    def run_cell():
        with known_failure("marker survived", "upd-txn: tracked gap", on_xfail=lambda: callbacks.append("xfail")):
            if failure is not None:
                raise failure
        return "passed"

    try:
        result = run_cell()
    except (AssertionError, ValueError, pytest.xfail.Exception) as caught:
        assert caught is failure
    else:
        assert failure is None
        assert result == "passed"
    assert callbacks == []
