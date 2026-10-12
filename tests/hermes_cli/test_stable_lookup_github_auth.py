"""The stable lookup spends the user's GitHub credential, not the shared anonymous budget."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import urllib.request

import pytest

pytestmark = pytest.mark.real_release_channels

from hermes_cli import github_api, source_releases

SHA = "818c13be1dc4fd28987e1e881a9408224afd4535"


@pytest.fixture
def github(monkeypatch):
    """A local api.github.com: answers releases/latest + commits/<tag>, or 403s the quota."""
    seen: list[tuple[str, str | None]] = []
    state = {"limited": False, "reject_token": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            auth = self.headers.get("Authorization")
            seen.append((self.path, auth))
            if state["reject_token"] and auth:
                body = b'{"message":"Bad credentials"}'
                self.send_response(401)
            elif state["limited"]:
                body = b'{"message":"API rate limit exceeded"}'
                self.send_response(403)
                self.send_header("x-ratelimit-remaining", "0")
                self.send_header("x-ratelimit-reset", "9999999999")
            elif self.path.endswith("/releases/latest"):
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real = urllib.request.urlopen

    def local(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        assert url.startswith("https://api.github.com/"), url  # the token must never leave GitHub
        rerouted = urllib.request.Request(
            url.replace("https://api.github.com", f"http://127.0.0.1:{server.server_port}"),
            headers=dict(request.header_items()) if isinstance(request, urllib.request.Request) else {})
        return real(rerouted, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local)
    monkeypatch.setattr(github_api, "_gh_cli_token", lambda: None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    yield seen, state
    server.shutdown()
    server.server_close()
    thread.join()


def test_stable_lookup_sends_the_configured_token(github, monkeypatch):
    seen, _ = github
    monkeypatch.setenv("GH_TOKEN", "ghp_user")
    target = source_releases._resolve_stable("NousResearch/hermes-agent")
    assert (target.commit, target.version) == (SHA, "0.21.6")
    assert seen and all(auth == "Bearer ghp_user" for _, auth in seen)


@pytest.mark.parametrize("token", [None, "ghp_user"])
def test_a_rate_limit_is_reported_as_one_not_as_a_missing_release(github, monkeypatch, token):
    _, state = github
    state["limited"] = True
    if token:
        monkeypatch.setenv("GH_TOKEN", token)
    else:
        monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent")
    message = str(caught.value)
    assert "rate limit" in message and "No published stable release" not in message
    assert ("GITHUB_TOKEN in the environment" in message) is (token is None)


def test_a_rejected_token_then_anonymous_limit_names_the_anonymous_quota(github, monkeypatch):
    seen, state = github
    state["limited"] = state["reject_token"] = True
    monkeypatch.setenv("GH_TOKEN", "ghp_revoked")
    with pytest.raises(ValueError) as caught:
        source_releases._resolve_stable("NousResearch/hermes-agent")
    # The 403 came from the anonymous retry, so the user's token quota is not the one to wait on.
    assert [auth for _, auth in seen] == ["Bearer ghp_revoked", None]
    assert "anonymous requests are limited" in str(caught.value)
    assert "for your GITHUB_TOKEN" not in str(caught.value)
