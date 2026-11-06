"""Regression tests for background-review compressor detachment warnings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    ("error", "has_traceback"),
    [
        (NotImplementedError("provider-specific refusal detail"), False),
        (RuntimeError("unexpected detachment detail"), True),
    ],
    ids=["unsupported", "unexpected-error"],
)
def test_detachment_warning_classifies_expected_refusal(
    error: Exception, has_traceback: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Expected unsupported detachment is concise; unexpected failures retain traceback."""
    import agent.background_review as br

    compressor = MagicMock()
    compressor.bind_session_state.side_effect = error
    review_agent = SimpleNamespace(
        context_compressor=compressor,
        compression_in_place=False,
        compression_enabled=True,
    )

    with caplog.at_level("WARNING", logger=br.logger.name):
        br._detach_fork_compression(review_agent)

    [record] = [
        record
        for record in caplog.records
        if "background-review compressor detachment" in record.getMessage()
    ]
    if has_traceback:
        assert record.exc_info is not None
    else:
        assert record.exc_info is None
    assert str(error) not in record.getMessage()
    assert review_agent.compression_in_place is True
    assert review_agent.compression_enabled is False
