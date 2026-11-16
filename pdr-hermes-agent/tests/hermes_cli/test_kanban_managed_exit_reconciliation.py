"""Join authority-owned worker receipts with main's exit classification."""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.mark.parametrize(
    "receipt, reaped, logged, event, rate_limited, terminal_provider",
    [
        (0, ("unknown", None), None, "protocol_violation", False, False),
        (kb.KANBAN_RATE_LIMIT_EXIT_CODE, ("nonzero_exit", 1), None,
         "rate_limited", True, False),
        (kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE, ("unknown", None), None,
         "crashed", False, True),
        (3, ("unknown", None), None, "crashed", False, False),
        (None, ("nonzero_exit", 3), None, "crashed", False, False),
        (None, ("unknown", None), kb.KANBAN_RATE_LIMIT_EXIT_CODE,
         "rate_limited", True, False),
    ],
)
def test_dead_worker_classification_accepts_managed_and_legacy_results(
    monkeypatch, receipt, reaped, logged, event, rate_limited, terminal_provider,
):
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _pid: reaped)
    monkeypatch.setattr(dispatch, "_worker_log_exit_code", lambda *_a, **_k: logged)
    monkeypatch.setattr(dispatch, "_worker_final_output", lambda *_a, **_k: "worker detail")

    result = dispatch._classify_dead_worker(
        900001, "private-host:owner", receipt, task_id="private-task", board="private-board",
    )

    assert result.event_kind == event
    assert result.rate_limited is rate_limited
    assert result.terminal_provider is terminal_provider
    expected_code = receipt if receipt is not None else reaped[1] if reaped[1] is not None else logged
    assert result.event_payload["exit_code"] == expected_code
    if rate_limited:
        assert "worker_output" not in result.event_payload
    else:
        assert result.event_payload["worker_output"] == "worker detail"
