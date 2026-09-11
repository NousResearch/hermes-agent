"""Main's queued-delivery metadata preserves the receipt privacy boundary."""

import pytest


def test_cli_queued_warning_does_not_expose_receipt_details():
    from hermes_cli.cron import _job_warnings

    lines = _job_warnings({"last_delivery_queued": {
        "bot-chat:private-profile": {"status": "queued", "delivery_id": "private-receipt"},
    }})
    warning = "\n".join(lines)
    assert "queued" in warning
    assert "do not resend" in warning
    assert "private-profile" not in warning
    assert "private-receipt" not in warning


@pytest.mark.parametrize("outcome", ["private provider diagnostic", {"private": "details"}])
def test_execution_rejects_unbounded_delivery_outcome(monkeypatch, tmp_path, outcome):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    execution = executions.create_execution("safe-job", source="direct")
    with pytest.raises(ValueError, match="delivery_outcome is invalid"):
        executions.finish_execution(execution["id"], success=True, delivery_outcome=outcome)
    assert executions.get_execution(execution["id"])["status"] == "claimed"
