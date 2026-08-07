"""`hermes status` must agree with how the gateway actually reads config.

Two failure modes this guards:

1. Platforms the gateway supports were missing from the status output
   entirely (Matrix, Mattermost, Home Assistant), so a user debugging a
   silent platform had no signal at all.
2. Rows were decided by mere presence of an env var. The gateway parses the
   flag-style ones through ``is_truthy_value()``, so ``WHATSAPP_ENABLED=false``
   is *disabled* while a presence check reports it configured — the status
   command was stating the opposite of the truth.
"""

from __future__ import annotations

import pytest

from hermes_cli import status as status_mod


PLATFORM_ENV_VARS = {
    "Telegram": "TELEGRAM_BOT_TOKEN",
    "Discord": "DISCORD_BOT_TOKEN",
    "Slack": "SLACK_BOT_TOKEN",
    "Matrix": "MATRIX_ACCESS_TOKEN",
    "Mattermost": "MATTERMOST_TOKEN",
    "Home Assistant": "HASS_TOKEN",
}


@pytest.fixture(autouse=True)
def clean_platform_env(monkeypatch):
    for var in (
        "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN",
        "MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD", "MATRIX_HOME_CHANNEL",
        "MATTERMOST_TOKEN", "MATTERMOST_HOME_CHANNEL", "HASS_TOKEN",
        "WHATSAPP_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _platform_rows(capsys):
    """Run the status command and return its Messaging Platforms lines."""
    from types import SimpleNamespace

    try:
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    return out


class TestMissingPlatformsAreListed:
    @pytest.mark.parametrize("label", ["Matrix", "Mattermost", "Home Assistant"])
    def test_platform_has_a_row(self, capsys, label):
        out = _platform_rows(capsys)
        assert label in out, f"{label} has no row in `hermes status`"


class TestEnvVarNamesMatchTheGateway:
    """The names must come from the gateway's own ingestion, not a guess.

    The first cut of this feature used MATRIX_HOMESERVER_URL and other
    invented names, so the rows were always "not configured" no matter how
    the user had set things up.
    """

    def test_token_env_names_match_the_canonical_map(self):
        from gateway.config import PLATFORM_TOKEN_ENV_NAMES, Platform

        canonical = {
            Platform.MATRIX: "MATRIX_ACCESS_TOKEN",
            Platform.MATTERMOST: "MATTERMOST_TOKEN",
        }
        for platform, expected in canonical.items():
            assert PLATFORM_TOKEN_ENV_NAMES[platform] == expected, (
                "the gateway's canonical map moved; the status rows must follow"
            )

    @pytest.mark.parametrize("label,var", sorted(PLATFORM_ENV_VARS.items()))
    def test_row_reflects_the_documented_var(self, capsys, monkeypatch, label, var):
        monkeypatch.setenv(var, "x" * 20)
        out = _platform_rows(capsys)
        line = next(l for l in out.splitlines() if l.strip().startswith(label))
        assert "not configured" not in line, (
            f"{label} stayed 'not configured' with {var} set: {line!r}"
        )


class TestMatrixAcceptsEitherCredential:
    def test_password_alone_counts_as_configured(self, capsys, monkeypatch):
        """gateway/config.py gates on MATRIX_ACCESS_TOKEN *or* MATRIX_PASSWORD."""
        monkeypatch.setenv("MATRIX_PASSWORD", "hunter2")
        out = _platform_rows(capsys)
        line = next(l for l in out.splitlines() if l.strip().startswith("Matrix"))
        assert "not configured" not in line


class TestFlagVarsAreParsedNotProbed:
    def test_whatsapp_false_is_not_configured(self, capsys, monkeypatch):
        """The regression: a non-empty 'false' read as enabled."""
        monkeypatch.setenv("WHATSAPP_ENABLED", "false")
        out = _platform_rows(capsys)
        line = next(l for l in out.splitlines() if l.strip().startswith("WhatsApp"))
        assert "not configured" in line, (
            f"WHATSAPP_ENABLED=false must not read as configured: {line!r}"
        )

    def test_whatsapp_true_is_configured(self, capsys, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")
        out = _platform_rows(capsys)
        line = next(l for l in out.splitlines() if l.strip().startswith("WhatsApp"))
        assert "not configured" not in line
