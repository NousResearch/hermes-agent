"""Unit tests for _print_loopback_ssh_hint() in hermes_cli/auth.py.

The helper warns users that loopback OAuth flows (Spotify) don't work over
SSH unless they set up an `ssh -L` port forward between their laptop's
browser and the remote host's loopback listener.

xAI Grok OAuth no longer uses this helper — its login is device-code-only —
but the Spotify integration still relies on it.
"""

from __future__ import annotations

import io
import contextlib
import socket


from hermes_cli import auth as auth_mod


def _cap(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_loopback_ssh_hint_silent_when_not_remote(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/spotify/callback", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert out == ""


def test_loopback_ssh_hint_has_visual_header(monkeypatch):
    """The hint should print a divider and header so it stands out in noisy output."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/callback"
    ))
    assert "Remote session detected" in out
    assert "---" in out  # divider is present


class TestSshUserAtHost:
    def test_resolves_user_and_hostname(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr(socket, "gethostname", lambda: "myserver")
        assert auth_mod._ssh_user_at_host() == "alice@myserver"


    def test_placeholder_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr(socket, "gethostname", lambda: "host1")
        assert auth_mod._ssh_user_at_host() == "<user>@host1"


    def test_placeholder_when_empty_hostname(self, monkeypatch):
        monkeypatch.setenv("USER", "dave")
        monkeypatch.setattr(socket, "gethostname", lambda: "")
        assert auth_mod._ssh_user_at_host() == "dave@<this-host>"
