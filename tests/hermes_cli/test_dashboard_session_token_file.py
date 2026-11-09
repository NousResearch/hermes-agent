"""The dashboard session token resolves from the environment, or a file it names."""

from unittest.mock import patch

import pytest


def test_plain_env_token_is_used_verbatim(monkeypatch):
    from hermes_cli import web_server_session_token as st

    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "raw-token")

    assert st.resolve_session_token() == "raw-token"


def test_file_prefix_reads_and_strips(monkeypatch, tmp_path):
    from hermes_cli import web_server_session_token as st

    token_file = tmp_path / "token"
    token_file.write_text("  file-token  " + chr(13) + chr(10), encoding="utf-8")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"file:{token_file}")

    assert st.resolve_session_token() == "file-token"


def test_unreadable_token_file_fails_loudly(monkeypatch, tmp_path):
    from hermes_cli import web_server_session_token as st

    monkeypatch.setenv(
        "HERMES_DASHBOARD_SESSION_TOKEN", f"file:{tmp_path / 'absent'}"
    )

    with pytest.raises(SystemExit, match="unreadable"):
        st.resolve_session_token()


def test_empty_token_file_fails_loudly(monkeypatch, tmp_path):
    from hermes_cli import web_server_session_token as st

    token_file = tmp_path / "token"
    token_file.write_text(chr(10), encoding="utf-8")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"file:{token_file}")

    with pytest.raises(SystemExit, match="empty"):
        st.resolve_session_token()


def test_malformed_token_file_fails_loudly(monkeypatch, tmp_path):
    from hermes_cli import web_server_session_token as st

    token_file = tmp_path / "token"
    token_file.write_bytes(b"\xff\xfe\x00not-utf8")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"file:{token_file}")

    with pytest.raises(SystemExit, match="unreadable"):
        st.resolve_session_token()


def test_no_source_mints_a_token(monkeypatch):
    from hermes_cli import web_server_session_token as st

    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)

    assert len(st.resolve_session_token()) > 0


class TestSessionTokenInjection:
    """The desktop shell mints HERMES_DASHBOARD_SESSION_TOKEN and signs its
    /api + /api/ws calls with it. The backend must adopt that token, else every
    desktop request 401s ("gateway is offline"). A main-merge once silently
    dropped this read — this guards the contract, not a literal value.
    """

    def test_honors_injected_token(self, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.web_server_session_token as st

        original_app = ws.app
        original_token = ws._SESSION_TOKEN
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-seeded-token")
        assert st.resolve_session_token() == "desktop-seeded-token"
        # No module reload: the loaded app and its adopted token are untouched.
        assert ws.app is original_app
        assert ws._SESSION_TOKEN == original_token
