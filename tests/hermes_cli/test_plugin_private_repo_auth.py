"""Private-repo plugin installs attach the user's stored HTTPS credential to git without persisting it.

Public-repo installs (the catalog default) must attempt the clone *anonymously* first and only fall
back to the stored credential when the server actually demands one — regression for #114526 where
injecting ``Authorization: basic`` against a public GitHub URL breaks the anonymous clone path
(GitHub rejects the Basic header on a public clone URL and git falls back to a Username prompt
that ``GIT_TERMINAL_PROMPT=0`` blocks with "could not read Username ... terminal prompts disabled").
"""

import base64
import subprocess
import sys

import pytest

from hermes_cli import git_credentials, plugins_cmd
from hermes_cli._subprocess_compat import noninteractive_git_env


def _auth_headers_for(env: dict, origin: str) -> list[str]:
    """``Authorization:`` extraheaders bound to *origin* in a git env block, in declared order."""
    count = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    out = []
    for i in range(count):
        if env.get(f"GIT_CONFIG_KEY_{i}") == f"http.{origin}/.extraheader":
            out.append(env[f"GIT_CONFIG_VALUE_{i}"])
    return out


def _seed_bare_upstream(tmp_path) -> None:
    """Spin up a local bare repo with one commit so anonymous clones can succeed when the test
    redirects a public URL at it."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True)
    (work / "plugin.yaml").write_text("name: probe\ndescription: d\nversion: '1'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "add", "."], check=True)
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "HEAD"], check=True)


def test_private_clone_falls_back_to_auth_after_credential_required_error(tmp_path, monkeypatch):
    """A clone from a remote that rejects anonymous access must (1) try anonymous first and
    only (2) retry with the stored credential when the server asks for one. The installed
    checkout carries no trace of the credential either way. (#114526 invariant for private remotes.)"""
    _seed_bare_upstream(tmp_path)

    clone_calls: list[list[str]] = []
    real_run = subprocess.run
    target_url = "https://git.example.test/acme/probe.git"

    def spy_run(argv, *a, **kw):
        env = kw.get("env") or {}
        if "clone" in argv:
            headers = _auth_headers_for(env, "https://git.example.test")
            clone_calls.append(headers)
            if len(clone_calls) == 1:
                # First attempt is anonymous; the private remote refuses with the canonical
                # "could not read Username ... terminal prompts disabled" message.
                return subprocess.CompletedProcess(
                    argv, returncode=128, stdout="",
                    stderr="fatal: could not read Username for 'https://git.example.test': "
                           "terminal prompts disabled\n",
                )
            argv = [a_ if a_ != target_url else str(tmp_path / "upstream.git") for a_ in argv]
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(plugins_cmd.subprocess, "run", spy_run)
    monkeypatch.setattr(git_credentials, "resolve_git_basic_auth", lambda url: ("alice", "s3cret"))

    dest = tmp_path / "clone"
    plugins_cmd._clone_plugin_repo(dest, target_url, None)

    expected = base64.b64encode(b"alice:s3cret").decode()
    assert len(clone_calls) == 2, f"expected anonymous + auth fallback, got {len(clone_calls)} clone attempts"
    assert clone_calls[0] == [], "first (anonymous) clone attempt must not carry an Authorization header"
    assert clone_calls[1] == [f"Authorization: basic {expected}"], "fallback clone must inject the stored credential"
    assert "s3cret" not in (dest / ".git" / "config").read_text(encoding="utf-8")
    assert expected not in (dest / ".git" / "config").read_text(encoding="utf-8")
    # Non-HTTPS URLs get no header; the hardened base env is otherwise untouched.
    base = noninteractive_git_env()
    assert git_credentials.with_git_auth(base, "git@github.com:acme/probe.git") == dict(base)


def test_public_clone_attempts_anonymously_when_credential_resolves(tmp_path, monkeypatch):
    """A public repo whose URL would resolve a stored GitHub credential via ``gh auth login`` must
    still be cloned anonymously: the Basic header would otherwise break the public clone path
    (see #114526). The fallback runs only when the server actually demands a credential."""
    _seed_bare_upstream(tmp_path)

    clone_calls: list[list[str]] = []
    real_run = subprocess.run
    target_url = "https://github.com/robbyczgw-cla/hermes-web-search-plus.git"
    origin = "https://github.com"

    def spy_run(argv, *a, **kw):
        env = kw.get("env") or {}
        if "clone" in argv:
            clone_calls.append(_auth_headers_for(env, origin))
            argv = [a_ if a_ != target_url else str(tmp_path / "upstream.git") for a_ in argv]
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(plugins_cmd.subprocess, "run", spy_run)
    # Simulate exactly the failing user state from #114526: ``gh auth login`` has populated the
    # credential resolver, so for any https://github.com URL it returns a non-None basic auth pair.
    monkeypatch.setattr(git_credentials, "resolve_git_basic_auth",
                        lambda url: ("x-access-token", "ghp_fake") if "github.com" in url else None)

    dest = tmp_path / "clone"
    plugins_cmd._clone_plugin_repo(dest, target_url, None)

    assert len(clone_calls) == 1, (
        f"public repo must clone in a single anonymous attempt, got {len(clone_calls)}: {clone_calls}"
    )
    assert clone_calls[0] == [], "public repo clone must not inject an Authorization header"
    # No fallback ever ran, so the stored token must not have leaked through any extraheader.
    assert "ghp_fake" not in str(clone_calls)


@pytest.mark.parametrize("verb", ["fetch", "pull"])
@pytest.mark.parametrize("outcome", ["ok", "refused", "not_found"])
def test_ref_fetch_and_update_pull_attach_credential_only_after_anonymous_refusal(
        tmp_path, monkeypatch, verb, outcome):
    """The pinned-ref fetch (``--ref`` install) and ``hermes plugins update``'s pull are the
    clone's siblings: with a stored GitHub credential resolvable they still run anonymously
    against a public remote, attach the credential only after the remote refuses, and surface a
    failure that is not about credentials (missing repo, bad commit, network) as-is — the stored
    credential is then never even resolved, let alone sent."""
    _seed_bare_upstream(tmp_path)
    repo = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", str(tmp_path / "upstream.git"), str(repo)], check=True)
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    public_url = "https://github.com/acme/public-plugin.git"
    subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin", public_url], check=True)

    attempts: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(argv, *a, **kw):
        if verb not in argv:
            return real_run(argv, *a, **kw)
        attempts.append(_auth_headers_for(kw.get("env") or {}, "https://github.com"))
        if outcome == "refused" and len(attempts) == 1:
            return subprocess.CompletedProcess(
                argv, 128, stdout="",
                stderr="fatal: could not read Username for 'https://github.com': terminal prompts disabled\n")
        if outcome == "not_found":
            return subprocess.CompletedProcess(
                argv, 128, stdout="", stderr=f"fatal: repository '{public_url}/' not found\n")
        return subprocess.CompletedProcess(argv, 0, stdout="Already up to date.\n", stderr="")

    monkeypatch.setattr(plugins_cmd.subprocess, "run", spy_run)
    if outcome == "not_found":
        monkeypatch.setattr(git_credentials, "resolve_git_basic_auth",
                            lambda url: pytest.fail("credential must not be resolved for a non-credential failure"))
    else:
        monkeypatch.setattr(git_credentials, "resolve_git_basic_auth", lambda url: ("x-access-token", "ghp_fake"))

    if verb == "fetch" and outcome == "not_found":
        with pytest.raises(plugins_cmd.PluginOperationError, match="not found"):
            plugins_cmd._checkout_exact_revision(repo, "git", revision, source_url=public_url)
    elif verb == "fetch":
        plugins_cmd._checkout_exact_revision(repo, "git", revision, source_url=public_url)
    else:
        ok, message = plugins_cmd._git_pull_plugin_dir(repo)
        assert ok is (outcome != "not_found"), message
        assert ("not found" in message) is (outcome == "not_found")

    expected = base64.b64encode(b"x-access-token:ghp_fake").decode()
    assert attempts == ([[], [f"Authorization: basic {expected}"]] if outcome == "refused" else [[]])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell askpass stub + local HTTP server")
def test_anonymous_attempt_fails_fast_under_inherited_askpass(tmp_path, monkeypatch):
    """With an inherited ``GIT_ASKPASS`` (VS Code terminal, ksshaskpass) the anonymous attempt
    against a remote answering 401 must still fail fast with the classifiable "could not read
    Username" refusal so the credential fallback fires, instead of handing the prompt to an
    askpass helper nobody answers and dying on the timeout with no second attempt."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Unauthorized(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="probe"')
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Unauthorized)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/acme/private.git"
        askpass = tmp_path / "askpass.sh"
        askpass.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
        askpass.chmod(0o755)
        monkeypatch.setenv("GIT_ASKPASS", str(askpass))
        # Plain-http local remote: stand in for the https credential lookup so the fallback's
        # second attempt is observable without a TLS fixture.
        monkeypatch.setattr(git_credentials, "with_git_auth",
                            lambda env, u: {**env, "HERMES_TEST_AUTH_ATTACHED": "1"})
        attempts: list[dict] = []
        real_run = subprocess.run
        monkeypatch.setattr(git_credentials.subprocess, "run",
                            lambda argv, **kw: attempts.append(kw["env"]) or real_run(argv, **kw))

        result = git_credentials.run_git_with_credential_fallback(
            ["git", "clone", url, str(tmp_path / "dest")], url, env=noninteractive_git_env(),
            capture_output=True, text=True, timeout=10)
    finally:
        server.shutdown()

    assert result.returncode != 0 and "could not read Username" in result.stderr
    assert len(attempts) == 2, "anonymous refusal must be classified and the credential fallback must fire"
    assert all("GIT_ASKPASS" not in env for env in attempts)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub credential helper")
def test_credential_fill_uses_stored_helper_and_never_prompts(tmp_path, monkeypatch):
    helper = tmp_path / "helper.sh"
    helper.write_text("#!/bin/sh\n[ \"$1\" = get ] && printf 'username=bob\\npassword=pw-from-helper\\n'\n", encoding="utf-8")
    helper.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f'[credential "https://git.example.test"]\n\thelper = !{helper}\n', encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_ASKPASS", "/nonexistent/askpass-must-not-run")

    assert git_credentials.resolve_git_basic_auth("https://git.example.test/acme/x.git") == ("bob", "pw-from-helper")
    # Unknown host: no helper answers → None quickly, no prompt attempt escaped.
    assert git_credentials.resolve_git_basic_auth("https://nothing.example.test/x.git") is None


def _refused(argv):
    return subprocess.CompletedProcess(
        argv, 128, stdout="",
        stderr="fatal: could not read Username for 'https://github.com': terminal prompts disabled\n")


def test_rejected_env_token_falls_back_to_the_next_owned_credential(monkeypatch):
    """An expired GITHUB_TOKEN in .env must not shadow a live ``gh auth login`` (#115257): after
    the remote refuses the first credential, the run retries with the next one and stops there."""
    monkeypatch.setattr(git_credentials, "iter_git_basic_auth", lambda url: iter([
        ("GITHUB_TOKEN/GH_TOKEN", ("x-access-token", "ghp_dead")),
        ("gh auth token", ("x-access-token", "gho_live")),
        ("git credential helper", ("bob", "never-needed")),
    ]))
    attempts: list[list[str]] = []

    def fake_run(argv, **kw):
        headers = _auth_headers_for(kw["env"], "https://github.com")
        attempts.append(headers)
        live = base64.b64encode(b"x-access-token:gho_live").decode()
        if headers == [f"Authorization: basic {live}"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return _refused(argv)

    monkeypatch.setattr(git_credentials.subprocess, "run", fake_run)
    url = "https://github.com/acme/private.git"
    result = git_credentials.run_git_with_credential_fallback(
        ["git", "clone", url, "dest"], url, env=noninteractive_git_env(), capture_output=True, text=True)

    dead = base64.b64encode(b"x-access-token:ghp_dead").decode()
    live = base64.b64encode(b"x-access-token:gho_live").decode()
    assert result.returncode == 0
    assert attempts == [[], [f"Authorization: basic {dead}"], [f"Authorization: basic {live}"]]


def test_every_credential_rejected_names_the_dead_env_token(monkeypatch):
    """When GitHub refuses every owned credential the git error gains a hint naming the .env token,
    and a failure that stops being about credentials ends the retries without one."""
    monkeypatch.setattr(git_credentials, "iter_git_basic_auth", lambda url: iter([
        ("GITHUB_TOKEN/GH_TOKEN", ("x-access-token", "ghp_dead")),
        ("gh auth token", ("x-access-token", "gho_dead_too")),
    ]))
    monkeypatch.setattr(git_credentials.subprocess, "run", lambda argv, **kw: _refused(argv))
    url = "https://github.com/acme/private.git"
    result = git_credentials.run_git_with_credential_fallback(
        ["git", "clone", url, "dest"], url, env=noninteractive_git_env(), capture_output=True, text=True)
    assert result.returncode != 0 and "GITHUB_TOKEN/GH_TOKEN in your .env was rejected" in result.stderr

    calls = []

    def not_found_once_authed(argv, **kw):
        calls.append(kw["env"])
        if _auth_headers_for(kw["env"], "https://github.com"):
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr=f"fatal: repository '{url}/' not found\n")
        return _refused(argv)

    monkeypatch.setattr(git_credentials.subprocess, "run", not_found_once_authed)
    result = git_credentials.run_git_with_credential_fallback(
        ["git", "clone", url, "dest"], url, env=noninteractive_git_env(), capture_output=True, text=True)
    assert len(calls) == 2 and "not found" in result.stderr and "hint:" not in result.stderr
