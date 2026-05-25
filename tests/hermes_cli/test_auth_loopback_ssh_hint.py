"""Unit tests for _print_loopback_ssh_hint() in hermes_cli/auth.py.

The helper exists to warn users that loopback OAuth flows (xAI Grok OAuth,
Spotify) don't work over SSH unless they set up an `ssh -L` port forward
between their laptop's browser and the remote host's loopback listener.
"""

from __future__ import annotations

import io
import contextlib

import pytest

from hermes_cli import auth as auth_mod


def _cap(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_loopback_ssh_hint_silent_when_not_remote(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:56121/callback", docs_url=auth_mod.XAI_OAUTH_DOCS_URL
    ))
    assert out == ""


def test_loopback_ssh_hint_prints_tunnel_command_on_ssh(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:56121/callback", docs_url=auth_mod.XAI_OAUTH_DOCS_URL
    ))
    # Must include the exact ssh -L command with the port from the redirect URI
    assert "ssh -N -L 56121:127.0.0.1:56121" in out
    # Must include the provider-specific docs URL
    assert auth_mod.XAI_OAUTH_DOCS_URL in out
    # Must always include the cross-provider SSH guide
    assert auth_mod.OAUTH_OVER_SSH_DOCS_URL in out


def test_loopback_ssh_hint_uses_actual_bound_port(monkeypatch):
    """When the preferred port is busy, _xai_start_callback_server falls back to
    an OS-assigned port. The hint must echo whichever port actually got bound,
    not the hardcoded constant."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:51234/callback", docs_url=auth_mod.XAI_OAUTH_DOCS_URL
    ))
    assert "ssh -N -L 51234:127.0.0.1:51234" in out
    assert "56121" not in out


def test_loopback_ssh_hint_silent_for_non_loopback_uri(monkeypatch):
    """Defense in depth: if a future caller passes a non-loopback redirect URI
    by mistake, we don't tell the user to forward an external port."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "https://example.com/callback", docs_url=auth_mod.XAI_OAUTH_DOCS_URL
    ))
    assert out == ""


def test_loopback_ssh_hint_silent_for_malformed_uri(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "not-a-uri", docs_url=auth_mod.XAI_OAUTH_DOCS_URL
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
    """The constant is 127.0.0.1, but parsing tolerates `localhost` too in case
    a future caller normalizes the URI differently."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://localhost:56121/callback"
    ))
    assert "ssh -N -L 56121:127.0.0.1:56121" in out
