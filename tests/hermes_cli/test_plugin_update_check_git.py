"""The plugin update check probes git remotes the way ``hermes plugins update`` pulls them.

``default_ls_remote`` backs ``hermes plugins check-updates`` and the gateway's scheduled update check.
It ran ``git ls-remote`` with the caller's ambient git: the user's credential helpers, askpass and
terminal prompt. From the headless gateway, a private plugin's remote refusing anonymous access
started whatever helper the user's git names, including Git for Windows' ``helper-selector`` picker
window, which nothing answers (#148552 class). It now runs under ``noninteractive_git_env`` and
reaches stored credentials only through the shared ``run_git_with_credential_fallback``.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import git_credentials
from hermes_cli.plugins_updates import default_ls_remote


class _Unauthorized(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="probe"')
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def private_remote(tmp_path, monkeypatch):
    """A loopback git remote that refuses anonymous access, and a user git config whose credential
    helper records that it ran (and answers, so ambient git never falls through to a prompt)."""
    marker = tmp_path / "helper-ran"
    helper = tmp_path / "git-credential-recorder"
    helper.write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\nprintf 'username=u\\npassword=p\\n'\n",
                      encoding="utf-8", newline="\n")
    helper.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f"[credential]\n\thelper = {helper.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    server = HTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/acme/private-plugin.git", marker
    finally:
        server.shutdown()


@pytest.mark.platforms("any")
def test_update_check_never_runs_the_users_credential_helper_itself(private_remote):
    url, helper_ran = private_remote

    with pytest.raises(RuntimeError):
        default_ls_remote(url)
    assert not helper_ran.exists(), "the probe must not start the user's credential helper"


@pytest.mark.platforms("any")
def test_a_refused_update_check_goes_through_the_shared_credential_fallback(private_remote, monkeypatch):
    """A private plugin stays checkable: the refusal is answered with the credentials the user owns,
    the same way ``hermes plugins update`` answers it."""
    url, _helper_ran = private_remote
    asked: list[str] = []
    monkeypatch.setattr(git_credentials, "with_git_auth", lambda env, u, *_: asked.append(u) or dict(env))

    with pytest.raises(RuntimeError):
        default_ls_remote(url)
    assert asked == [url]
