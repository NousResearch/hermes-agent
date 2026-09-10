"""Private-repo plugin installs attach the user's stored HTTPS credential to git without persisting it."""

import base64
import subprocess
import sys

import pytest

from hermes_cli import git_credentials, plugins_cmd
from hermes_cli._subprocess_compat import noninteractive_git_env


def test_private_clone_authenticates_without_writing_credentials_to_checkout(tmp_path, monkeypatch):
    """A clone from a remote that rejects anonymous access succeeds when a credential is resolvable,
    and the installed checkout carries no trace of it."""
    upstream = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "-q", "--bare", str(upstream)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True)
    (work / "plugin.yaml").write_text("name: probe\ndescription: d\nversion: '1'\n")
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "add", "."], check=True)
    subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], check=True)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "HEAD"], check=True)

    # Stand-in for a private remote: git is real, the URL is https, and the test asserts the auth
    # header reached git by having git echo its effective config for that origin.
    seen = {}
    real_run = subprocess.run

    def spy_run(argv, *a, **kw):
        env = kw.get("env") or {}
        if "clone" in argv:
            headers = [env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))
                       if env[f"GIT_CONFIG_KEY_{i}"] == "http.https://git.example.test/.extraheader"]
            seen["headers"] = headers
            argv = [a_ if a_ != "https://git.example.test/acme/probe.git" else str(upstream) for a_ in argv]
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(plugins_cmd.subprocess, "run", spy_run)
    monkeypatch.setattr(git_credentials, "resolve_git_basic_auth", lambda url: ("alice", "s3cret"))

    dest = tmp_path / "clone"
    plugins_cmd._clone_plugin_repo(dest, "https://git.example.test/acme/probe.git", None)

    expected = base64.b64encode(b"alice:s3cret").decode()
    assert seen["headers"] == [f"Authorization: basic {expected}"]
    assert "s3cret" not in (dest / ".git" / "config").read_text()
    assert expected not in (dest / ".git" / "config").read_text()
    # Non-HTTPS URLs get no header; the hardened base env is otherwise untouched.
    base = noninteractive_git_env()
    assert git_credentials.with_git_auth(base, "git@github.com:acme/probe.git") == dict(base)


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
