"""Private-repo plugin installs attach the user's stored HTTPS credential to git without persisting it.

Public-repo installs (the catalog default) must attempt the clone *anonymously* first and only fall
back to the stored credential when the server actually demands one — regression for #114526 where
injecting ``Authorization: basic`` against a public GitHub URL breaks the anonymous clone path
(GitHub rejects the Basic header on a public clone URL and git falls back to a Username prompt
that ``GIT_TERMINAL_PROMPT=0`` blocks with "could not read Username ... terminal prompts disabled").
"""

import base64
import subprocess

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


def _check_credential_fill_uses_stored_helper_and_never_prompts(tmp_path, monkeypatch):
    helper = tmp_path / "helper.sh"
    helper.write_text("#!/bin/sh\n[ \"$1\" = get ] && printf 'username=bob\\npassword=pw-from-helper\\n'\n", encoding="utf-8")
    helper.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        f'[credential]\n\tuseHttpPath = true\n'
        f'[credential "https://git.example.test/acme/x.git"]\n\thelper = !{helper}\n',
        encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_ASKPASS", "/nonexistent/askpass-must-not-run")

    assert git_credentials.resolve_git_basic_auth("https://git.example.test/acme/x.git") == ("bob", "pw-from-helper")
    # Unknown host: no helper answers → None quickly, no prompt attempt escaped.
    assert git_credentials.resolve_git_basic_auth("https://nothing.example.test/x.git") is None


@pytest.mark.linux_only
def test_credential_fill_uses_stored_helper_linux(tmp_path, monkeypatch):
    _check_credential_fill_uses_stored_helper_and_never_prompts(tmp_path, monkeypatch)


@pytest.mark.macos_only
def test_credential_fill_uses_stored_helper_macos(tmp_path, monkeypatch):
    _check_credential_fill_uses_stored_helper_and_never_prompts(tmp_path, monkeypatch)


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
