"""Regression tests for #98022 — interrupted receipt must not re-fire restarts.

`plan.runtimes[].code_sha` is captured *before* the pull, so an interrupted
receipt's plan always looks stale. When the interrupted run already pulled
the code matching the current checkout (receipt["post_update"]["sha"]),
the fleet is current and no catchup restart is owed.
"""
import hermes_cli.update_cmd_fleet as fleet


def _receipt(**over):
    base = {
        "stop_reason": "interrupted",
        "plan": {"runtimes": [{"code_sha": "pre-pull-sha"}]},
    }
    base.update(over)
    return base


def test_matching_post_update_sha_reports_current(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.update_receipt.read_latest_receipt",
        lambda: _receipt(post_update={"sha": "abc123"}),
    )
    monkeypatch.setattr(
        "hermes_cli.update_cmd_fleet._current_checkout_sha", lambda: "abc123"
    )
    assert fleet._receipt_reports_stale_runtime() is False


def test_mismatched_post_update_sha_still_reports_stale(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.update_receipt.read_latest_receipt",
        lambda: _receipt(post_update={"sha": "newer-sha"}),
    )
    monkeypatch.setattr(
        "hermes_cli.update_cmd_fleet._current_checkout_sha", lambda: "abc123"
    )
    assert fleet._receipt_reports_stale_runtime() is True


def test_missing_post_update_preserves_old_behavior(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.update_receipt.read_latest_receipt",
        lambda: _receipt(),
    )
    monkeypatch.setattr(
        "hermes_cli.update_cmd_fleet._current_checkout_sha", lambda: "abc123"
    )
    assert fleet._receipt_reports_stale_runtime() is True
