"""Task 29 advisory inspection constants invariants."""

from htr.advisory_inspection_constants import (
    MAX_CONTROL_JSON_BYTES,
    MAX_RAW_READ_BYTES,
    SUPPLEMENTAL_FINDING_TOKENS,
)


def test_supplemental_finding_registry_count():
    assert len(SUPPLEMENTAL_FINDING_TOKENS) == 23


def test_raw_read_budget_alignment():
    assert MAX_RAW_READ_BYTES == MAX_CONTROL_JSON_BYTES + 2
