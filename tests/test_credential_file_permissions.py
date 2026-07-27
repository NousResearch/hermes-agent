"""Read-time permission warnings for on-disk credential/token files.

``utils.warn_if_credential_file_broadly_readable`` is the shared helper for
the class of bug PR #60009 reported for ``slack_tokens.json``: token files
provisioned by hand (or written by older Hermes versions without an explicit
mode) end up group/world-readable under the default umask, silently exposing
plaintext secrets to other local users. The helper warns with a remediation
hint; adapters call it on every read path.
"""

import logging
import os

import pytest

from utils import warn_if_credential_file_broadly_readable

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission-bit semantics required"
)


class TestWarnIfCredentialFileBroadlyReadable:
    def test_warns_on_world_readable(self, tmp_path, caplog):
        f = tmp_path / "slack_tokens.json"
        f.write_text("{}")
        f.chmod(0o644)

        with caplog.at_level(logging.WARNING):
            warned = warn_if_credential_file_broadly_readable(f, label="[Slack]")

        assert warned is True
        assert "group/world-readable" in caplog.text
        assert "chmod 600" in caplog.text
        assert "[Slack]" in caplog.text

    def test_warns_on_group_readable(self, tmp_path, caplog):
        f = tmp_path / "tokens.json"
        f.write_text("{}")
        f.chmod(0o640)

        with caplog.at_level(logging.WARNING):
            assert warn_if_credential_file_broadly_readable(f) is True

    def test_silent_on_0600(self, tmp_path, caplog):
        f = tmp_path / "tokens.json"
        f.write_text("{}")
        f.chmod(0o600)

        with caplog.at_level(logging.WARNING):
            assert warn_if_credential_file_broadly_readable(f) is False
        assert "group/world-readable" not in caplog.text

    def test_silent_on_missing_file(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert (
                warn_if_credential_file_broadly_readable(tmp_path / "nope.json")
                is False
            )
        assert caplog.text == ""

    def test_uses_provided_logger(self, tmp_path):
        f = tmp_path / "tokens.json"
        f.write_text("{}")
        f.chmod(0o644)

        records = []

        class _Sink(logging.Handler):
            def emit(self, record):
                records.append(record)

        log = logging.getLogger("test.credfile.sink")
        log.addHandler(_Sink())
        try:
            assert warn_if_credential_file_broadly_readable(f, log=log) is True
        finally:
            log.handlers.clear()
        assert records and "chmod 600" in records[0].getMessage()


class TestGoogleChatReadPathWarns:
    def test_load_user_credentials_warns_on_broad_perms(
        self, tmp_path, monkeypatch, caplog
    ):
        """The google_chat legacy token read path shares the Slack fix."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # google-auth may not be installed in this environment; the warning
        # fires before the import guard, so a None return is fine either way.
        token = tmp_path / "google_chat_user_token.json"
        token.write_text("{}")
        token.chmod(0o644)

        from plugins.platforms.google_chat.oauth import load_user_credentials

        with caplog.at_level(logging.WARNING):
            load_user_credentials()

        assert "group/world-readable" in caplog.text
        assert "chmod 600" in caplog.text
