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


def test_loopback_ssh_hint_prints_tunnel_command_on_ssh(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/spotify/callback", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert "ssh -N -L 43827:127.0.0.1:43827" in out
    # Must include the provider-specific docs URL
    assert auth_mod.SPOTIFY_DOCS_URL in out
    # Must always include the cross-provider SSH guide
    assert auth_mod.OAUTH_OVER_SSH_DOCS_URL in out


def test_loopback_ssh_hint_uses_actual_bound_port(monkeypatch):
    """When the preferred port is busy, the callback server falls back to an
    OS-assigned port. The hint must echo whichever port actually got bound,
    not a hardcoded constant."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:51234/callback", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert "ssh -N -L 51234:127.0.0.1:51234" in out
    assert "43827" not in out


def test_loopback_ssh_hint_silent_for_non_loopback_uri(monkeypatch):
    """Defense in depth: if a future caller passes a non-loopback redirect URI
    by mistake, we don't tell the user to forward an external port."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "https://example.com/callback", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert out == ""


def test_loopback_ssh_hint_silent_for_malformed_uri(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "not-a-uri", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert out == ""


def test_loopback_ssh_hint_works_without_provider_docs_url(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/spotify/callback"
    ))
    assert "ssh -N -L 43827:127.0.0.1:43827" in out
    # Generic SSH guide is always present even without a provider-specific URL
    assert auth_mod.OAUTH_OVER_SSH_DOCS_URL in out
    # Should not falsely show "Provider docs:" when no docs_url was passed
    assert "Provider docs:" not in out


def test_loopback_ssh_hint_accepts_localhost_hostname(monkeypatch):
    """Parsing tolerates `localhost` in case a future caller normalizes the
    URI differently."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://localhost:43827/callback"
    ))
    assert "ssh -N -L 43827:127.0.0.1:43827" in out


def test_loopback_ssh_hint_includes_user_at_host(monkeypatch):
    """The SSH command should include a detected user@host so the user can
    copy-paste it without manually substituting placeholders."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    monkeypatch.setattr(auth_mod, "_ssh_user_at_host", lambda: "alice@myserver.lan")
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/callback"
    ))
    assert "ssh -N -L 43827:127.0.0.1:43827 alice@myserver.lan" in out


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

    def test_falls_back_to_logname(self, monkeypatch):
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.setenv("LOGNAME", "bob")
        monkeypatch.setattr(socket, "gethostname", lambda: "host1")
        assert auth_mod._ssh_user_at_host() == "bob@host1"

    def test_placeholder_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr(socket, "gethostname", lambda: "host1")
        assert auth_mod._ssh_user_at_host() == "<user>@host1"

    def test_placeholder_when_socket_raises(self, monkeypatch):
        monkeypatch.setenv("USER", "charlie")
        def _raise():
            raise OSError("no network")
        monkeypatch.setattr(socket, "gethostname", _raise)
        assert auth_mod._ssh_user_at_host() == "charlie@<this-host>"

    def test_placeholder_when_empty_hostname(self, monkeypatch):
        monkeypatch.setenv("USER", "dave")
        monkeypatch.setattr(socket, "gethostname", lambda: "")
        assert auth_mod._ssh_user_at_host() == "dave@<this-host>"
