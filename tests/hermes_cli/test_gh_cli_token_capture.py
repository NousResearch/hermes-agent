"""Regression for #138444: a wrapper around the gh CLI that prints to stdout
must not become the GitHub credential. Only a token-shaped line of the
``gh auth token`` capture may be used; a capture without one is no token at
all, and the ladder falls through to the anonymous rung."""
import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.real_release_channels

from hermes_cli import github_api, source_releases

SHA = "818c13be1dc4fd28987e1e881a9408224afd4535"
TOKEN = "gho_captureProbe0123456789"
MISE_NOISE = "mise ~/.config/mise/config.toml tools: gh@2.102.0"


@pytest.mark.parametrize(
    "stdout, expected",
    [
        (TOKEN, TOKEN),                                        # the clean capture
        (f"{MISE_NOISE}\n{TOKEN}", TOKEN),                     # noise ahead of the token (#138444)
        (f"{MISE_NOISE}\n{TOKEN}\nwrapper done", TOKEN),       # noise on both sides
        (f"\n\n{TOKEN}\n", TOKEN),                             # blank padding
        ("github_pat_fine_grained_probe_0123456789", "github_pat_fine_grained_probe_0123456789"),
        ("ghp_classicPATprobe012345678", "ghp_classicPATprobe012345678"),
        (MISE_NOISE, None),                                    # noise only, no token line
        ("", None),
        ("gh-short_1", None),                                  # prefix without a token body
    ],
)
def test_captured_gh_cli_token_takes_only_the_token_line(stdout, expected):
    assert github_api.captured_gh_cli_token(stdout) == expected


def _fresh_probe(monkeypatch, stdout, returncode=0):
    """Point ``_gh_cli_token`` at a faked gh subprocess and reset its cache."""
    monkeypatch.setattr(github_api, "_gh_cli_cache", None, raising=False)
    monkeypatch.setattr(github_api, "_gh_cli_probed", False, raising=False)
    result = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(github_api.subprocess, "run", lambda *args, **kwargs: result)


def test_gh_cli_token_drops_wrapper_noise_from_the_capture(monkeypatch):
    _fresh_probe(monkeypatch, f"{MISE_NOISE}\n{TOKEN}\n")
    assert github_api._gh_cli_token() == TOKEN


def test_gh_cli_token_without_a_token_line_goes_anonymous(monkeypatch):
    _fresh_probe(monkeypatch, f"{MISE_NOISE}\n")
    assert github_api._gh_cli_token() is None


def test_gh_cli_token_ignores_a_failing_gh(monkeypatch):
    _fresh_probe(monkeypatch, TOKEN, returncode=1)
    assert github_api._gh_cli_token() is None


@pytest.fixture
def github(monkeypatch):
    """A local api.github.com that records every Authorization header."""
    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            if self.path.endswith("/releases/latest"):
                body = json.dumps({"tag_name": "v0.21.6", "draft": False, "prerelease": False}).encode()
                self.send_response(200)
            elif self.path.endswith("/commits/v0.21.6"):
                body = json.dumps({"sha": SHA}).encode()
                self.send_response(200)
            else:
                body = b"{}"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_urlopen = urllib.request.urlopen

    def local(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        rerouted = urllib.request.Request(
            url.replace("https://api.github.com", f"http://127.0.0.1:{server.server_port}"),
            headers=dict(request.header_items()) if isinstance(request, urllib.request.Request) else {})
        return real_urlopen(rerouted, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    yield seen
    server.shutdown()
    server.server_close()


def test_stable_lookup_survives_a_wrapper_that_prints_to_stdout(github, monkeypatch):
    _fresh_probe(monkeypatch, f"{MISE_NOISE}\n{TOKEN}\n")
    target = source_releases._resolve_stable("NousResearch/hermes-agent")
    assert (target.commit, target.version) == (SHA, "0.21.6")
    assert github and all(auth == f"Bearer {TOKEN}" for auth in github)
