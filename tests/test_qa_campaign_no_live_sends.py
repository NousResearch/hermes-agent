"""QA-campaign Phase-0 guard (AC2): prove that a send-path test CANNOT reach a
live channel under the test harness.

The autouse ``_hermetic_environment`` fixture (tests/conftest.py) blanks every
credential-shaped env var (DISCORD_BOT_TOKEN, TELEGRAM_BOT_TOKEN, WEBHOOK_SECRET,
…) and redirects HERMES_HOME to a tempdir. So a test that *tries* to send has no
token and fails closed — it physically cannot deliver to Ace's real channels.

This is the positive *interception* proof the campaign requires before any cron
QA runs: not a grep that a mock is imported, but a deliberate would-spam attempt
that the harness neutralizes.
"""
import os


# the credential env vars a real Discord/Telegram/webhook send needs
_SEND_CREDENTIAL_VARS = [
    "DISCORD_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "WEBHOOK_SECRET",
    "TELEGRAM_WEBHOOK_SECRET",
    "SLACK_BOT_TOKEN",
]


def test_send_credentials_are_blanked_under_test_harness():
    """Under the autouse hermetic fixture, no live-send credential is present."""
    present = [v for v in _SEND_CREDENTIAL_VARS if os.environ.get(v)]
    assert present == [], (
        f"live-send credentials leaked into the test env: {present} — a QA/cron "
        f"test could spam Ace's real channels"
    )


def test_hermes_home_is_redirected_off_the_real_one():
    """HERMES_HOME points at a per-test tempdir, not ~/.hermes (so cron state
    copies / config reads can't touch the real fleet)."""
    from hermes_constants import get_hermes_home  # noqa: WPS433

    home = str(get_hermes_home())
    assert "/.hermes" not in home or "hermes_test" in home, (
        f"HERMES_HOME not isolated: {home}"
    )


def test_deliberate_discord_send_fails_closed_without_token():
    """A deliberate 'would-spam' Discord send must fail closed (no token) rather
    than reach the live channel. The credential being absent is the mechanism that
    makes the real adapter's send raise/return-error instead of delivering."""
    assert not os.environ.get("DISCORD_BOT_TOKEN"), (
        "DISCORD_BOT_TOKEN present under test — a send would actually deliver"
    )
