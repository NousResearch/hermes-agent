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
    (work / "plugin.yaml").write_text("name: probe\ndescription: d\nversion: '1'\n")
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
    assert "s3cret" not in (dest / ".git" / "config").read_text()
    assert expected not in (dest / ".git" / "config").read_text()
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


def test_non_https_clone_skips_auth_header_even_when_credential_resolves(tmp_path, monkeypatch):
    """SSH-style ``git@github.com:owner/repo.git`` URLs reach git via the local ssh agent, not via
    the https Authorization extraheader — the credential path must be a true no-op for them."""
    clone_calls: list[list[str]] = []
    real_run = subprocess.run
    target_url = "git@github.com:robbyczgw-cla/hermes-web-search-plus.git"

    def spy_run(argv, *a, **kw):
        env = kw.get("env") or {}
        if "clone" in argv:
            clone_calls.append(_auth_headers_for(env, "github.com"))
            # Skip the real clone; the SSH path isn't under test here.
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")
        # Subsequent post-clone git calls (rev-parse, etc.) aren't under test; swallow them so
        # the absence of a real upstream doesn't cascade into a fixture error.
        if isinstance(argv, (list, tuple)) and argv and argv[0].endswith("git") and len(argv) > 1:
            return subprocess.CompletedProcess(argv, returncode=0, stdout="deadbeef\n", stderr="")
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(plugins_cmd.subprocess, "run", spy_run)
    monkeypatch.setattr(git_credentials, "resolve_git_basic_auth", lambda url: ("x-access-token", "ghp_fake"))

    dest = tmp_path / "clone"
    plugins_cmd._clone_plugin_repo(dest, target_url, None)

    assert len(clone_calls) == 1, (
        f"non-HTTPS clone must be a single anonymous attempt, got {len(clone_calls)}: {clone_calls}"
    )
    assert clone_calls[0] == [], (
        f"non-HTTPS URL must not receive an Authorization extraheader, got {clone_calls[0]}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub credential helper")
def test_credential_fill_uses_stored_helper_and_never_prompts(tmp_path, monkeypatch):
    helper = tmp_path / "helper.sh"
    helper.write_text("#!/bin/sh\n[ \"$1\" = get ] && printf 'username=bob\\npassword=pw-from-helper\\n'\n")
    helper.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f'[credential "https://git.example.test"]\n\thelper = !{helper}\n')
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_ASKPASS", "/nonexistent/askpass-must-not-run")

    assert git_credentials.resolve_git_basic_auth("https://git.example.test/acme/x.git") == ("bob", "pw-from-helper")
    # Unknown host: no helper answers → None quickly, no prompt attempt escaped.
    assert git_credentials.resolve_git_basic_auth("https://nothing.example.test/x.git") is None
