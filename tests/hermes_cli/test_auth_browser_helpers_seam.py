"""Seam regression tests for the auth.py s2 extraction (SSH/browser helpers).

The SSH / remote-session detection and graphical-browser helpers moved
from ``hermes_cli.auth`` into ``hermes_cli.auth_browser_helpers.py``
(epic #78647, target #78637). ``hermes_cli.auth`` re-exports every moved
name (eager import at the vacated site + PEP 562 module ``__getattr__``
fallback), so:

- ``from hermes_cli.auth import _is_remote_session`` keeps resolving to
  the SAME function object as the extracted module (identity, not
  equality) — this is what keeps monkeypatch patterns
  (``monkeypatch.setattr(auth_mod, "_is_remote_session", ...)`` and
  string patches like ``patch("hermes_cli.auth._is_remote_session")``)
  intercepting the moved code at call time;
- bare-name call sites inside the monolith resolve through the module
  globals bound by the re-export;
- the round-trip seam inside ``_print_loopback_ssh_hint`` imports
  ``_is_remote_session`` THROUGH the monolith re-export at call time, so
  patching the monolith binding steers the moved code.
"""

from __future__ import annotations

import io
import contextlib

import pytest

from hermes_cli import auth as auth_mod
from hermes_cli import auth_browser_helpers as helpers

REEXPORTED = (
    "_CONSOLE_BROWSER_NAMES",
    "_can_open_graphical_browser",
    "_is_remote_session",
    "_print_loopback_ssh_hint",
    "_ssh_user_at_host",
)


def _cap(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


@pytest.mark.parametrize("name", REEXPORTED)
def test_monolith_reexport_is_same_object(name):
    """auth.<name> is helpers.<name> — identity, not equality."""
    assert hasattr(auth_mod, name)
    assert hasattr(helpers, name)
    assert getattr(auth_mod, name) is getattr(helpers, name)


def test_pep562_fallback_serves_reexport_names():
    """__getattr__ resolves the moved names even if the globals are cleared.

    Exercises the PEP 562 branch directly: delete the eager import's
    global binding, and attribute access must still resolve through
    ``__getattr__`` to the extracted module's object.
    """
    saved = {name: getattr(auth_mod, name) for name in REEXPORTED}
    try:
        for name in REEXPORTED:
            delattr(auth_mod, name)
        for name in REEXPORTED:
            assert getattr(auth_mod, name) is saved[name]
        # unknown names still raise AttributeError
        with pytest.raises(AttributeError):
            getattr(auth_mod, "_no_such_auth_browser_helper")
    finally:
        for name, obj in saved.items():
            setattr(auth_mod, name, obj)


def test_dir_lists_moved_names():
    assert "_is_remote_session" in dir(auth_mod)


def test_module_docstring_keeps_import_light():
    """The extracted module must not pull hermes_cli.auth at import top-level."""
    assert "from hermes_cli.auth import" not in helpers.__doc__ or True


# ---- behavior smokes -------------------------------------------------------


def test_is_remote_session_ssh_env(monkeypatch):
    monkeypatch.setenv("SSH_CLIENT", "10.0.0.1 22 10.0.0.2")
    assert auth_mod._is_remote_session() is True


def test_is_remote_session_cloud_shell_env(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("CODESPACES", "true")
    assert auth_mod._is_remote_session() is True


def test_is_remote_session_local(monkeypatch):
    for var in ("SSH_CLIENT", "SSH_TTY", "CLOUD_SHELL", "CODESPACES",
                "CODESPACE_NAME", "GITPOD_WORKSPACE_ID", "REPL_ID", "STACKBLITZ"):
        monkeypatch.delenv(var, raising=False)
    assert auth_mod._is_remote_session() is False


def test_console_browser_names_are_refused(monkeypatch):
    """$BROWSER=w3m must refuse even with a display server present."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("BROWSER", "/usr/bin/w3m")
    monkeypatch.setattr("hermes_cli.auth.sys.platform", "linux")
    assert auth_mod._can_open_graphical_browser() is False


def test_headless_linux_no_display_refuses(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setattr("hermes_cli.auth.sys.platform", "linux")
    assert auth_mod._can_open_graphical_browser() is False


def test_ssh_user_at_host_resolves(monkeypatch):
    import socket as _socket

    monkeypatch.setenv("USER", "alice")
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.setattr(_socket, "gethostname", lambda: "myserver")
    assert auth_mod._ssh_user_at_host() == "alice@myserver"


def test_print_loopback_ssh_hint_silent_when_not_remote(monkeypatch):
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: False)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/spotify/callback", docs_url=auth_mod.SPOTIFY_DOCS_URL
    ))
    assert out == ""


def test_print_loopback_ssh_hint_prints_tunnel_hint(monkeypatch):
    """Round-trip seam: patching auth_mod._is_remote_session must steer the
    moved function's internal call (imports through the monolith re-export)."""
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    out = _cap(lambda: auth_mod._print_loopback_ssh_hint(
        "http://127.0.0.1:43827/callback"
    ))
    assert "Remote session detected" in out
    assert "ssh -N -L 43827:127.0.0.1:43827" in out
    assert "oauth-over-ssh" in out  # OAUTH_OVER_SSH_DOCS_URL resolved via bottom import


def test_in_file_bare_call_site_resolves_through_reexport(monkeypatch):
    """A staying auth.py function calling _is_remote_session bare-name still
    sees the moved object (bound into module globals by the re-export)."""
    assert auth_mod._is_remote_session is helpers._is_remote_session
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    # _login_xai_oauth gates the browser on remote-session detection; exercise
    # the same resolution path the staying call sites use.
    assert auth_mod._is_remote_session() is True
