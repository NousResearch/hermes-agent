"""Mechanical deduplication for typed terminal-status/final mirrors."""

from gateway.config import Platform
from gateway.run import _final_mirror_statuses_to_deliver


def test_exact_visible_terminal_status_is_suppressed():
    message = "Provider authentication failed."

    assert (
        _final_mirror_statuses_to_deliver(
            Platform.DISCORD,
            [message],
            message,
        )
        == ()
    )


def test_equality_is_checked_after_normal_secret_redaction(monkeypatch):
    monkeypatch.setattr(
        "agent.redact.redact_sensitive_text",
        lambda _text, *, force: "Rejected [REDACTED]",
    )

    assert (
        _final_mirror_statuses_to_deliver(
            Platform.DISCORD,
            ["Rejected first-secret"],
            "Rejected second-secret",
        )
        == ()
    )


def test_distinct_terminal_status_is_preserved():
    assert _final_mirror_statuses_to_deliver(
        Platform.DISCORD,
        ["Provider failed before final delivery."],
        "Authentication failed; credentials need refresh.",
    ) == ("Provider failed before final delivery.",)


def test_empty_final_does_not_discard_a_terminal_status():
    assert _final_mirror_statuses_to_deliver(
        Platform.DISCORD,
        ["Provider failed before final delivery."],
        "",
    ) == ("Provider failed before final delivery.",)
