"""SSH-isolated dashboard token must be read live, not copied at import."""

import os
from hermes_cli import web_server as ws

try:
    import pytest

    @pytest.fixture(autouse=True)
    def restore_session_token():
        orig_token = ws._SESSION_TOKEN
        orig_env = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN")
        orig_state = getattr(ws.app.state, "session_token", None)
        yield
        ws._SESSION_TOKEN = orig_token
        if orig_env is not None:
            os.environ["HERMES_DASHBOARD_SESSION_TOKEN"] = orig_env
        else:
            os.environ.pop("HERMES_DASHBOARD_SESSION_TOKEN", None)
        if hasattr(ws.app.state, "session_token"):
            if orig_state is not None:
                ws.app.state.session_token = orig_state
            else:
                delattr(ws.app.state, "session_token")

except ImportError:
    pass


def test_session_token_getter_tracks_ssh_apply():
    token = "ab" * 32
    ws._apply_ssh_session_token(token)
    assert ws._session_token() == token
    assert len(ws._session_token()) == 64


def test_from_import_of_module_global_does_not_track_rebind():
    """`from module import _SESSION_TOKEN` copies the str; apply() rebinds."""
    captured = ws._SESSION_TOKEN
    ws._apply_ssh_session_token("cd" * 32)
    assert captured != ws._SESSION_TOKEN
    assert ws._session_token() == ws._SESSION_TOKEN
    assert ws._session_token() != captured
