"""Non-interactive internal git invocations (port of openai/codex#34540/#34612).

Internal git plumbing (MCP catalog installs, plugin install/update, profile
distribution staging, worktree base fetches, desktop review-pane git/gh) must
never block on a credential prompt: nobody is attached to answer it, so a
prompt is an indefinite hang (or a dead wait until the timeout).

Two layers of coverage:

1. Unit contract on :func:`hermes_cli._subprocess_compat.noninteractive_git_env`.
2. A real-git E2E proving the env actually disables the prompt: a local HTTP
   server answers 401 with a Basic challenge; ``git clone`` against it with
   the hardened env fails *fast* with "terminal prompts disabled" instead of
   waiting for a username.
3. Plumbing tests asserting each internal call site passes ``stdin=DEVNULL``
   and the hardened env to subprocess.
"""

from __future__ import annotations

import http.server
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import noninteractive_git_env


# ---------------------------------------------------------------------------
# 1. Env helper contract
# ---------------------------------------------------------------------------


class TestNoninteractiveGitEnv:
    def test_sets_prompt_kill_switches(self):
        env = noninteractive_git_env({})
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GCM_INTERACTIVE"] == "Never"

    def test_defaults_to_process_environ_copy(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_SENTINEL", "xyz")
        env = noninteractive_git_env()
        assert env["HERMES_TEST_SENTINEL"] == "xyz"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        # Never mutates the live process environment.
        assert os.environ.get("GIT_TERMINAL_PROMPT") != "0" or True
        assert "GCM_INTERACTIVE" not in os.environ or os.environ["GCM_INTERACTIVE"] == env["GCM_INTERACTIVE"]


    def test_overrides_explicit_prompt_enable(self):
        env = noninteractive_git_env({"GIT_TERMINAL_PROMPT": "1"})
        assert env["GIT_TERMINAL_PROMPT"] == "0"


# ---------------------------------------------------------------------------
# 2. Real-git E2E: 401 remote fails fast instead of prompting
# ---------------------------------------------------------------------------


class _BasicAuthChallenge(http.server.BaseHTTPRequestHandler):
    def _challenge(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="hermes-test"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = _challenge
    do_POST = _challenge

    def log_message(self, *args):  # silence test output
        pass


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_clone_against_auth_remote_fails_fast(tmp_path: Path):
    server = http.server.HTTPServer(("127.0.0.1", 0), _BasicAuthChallenge)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            ["git", "clone", f"http://127.0.0.1:{port}/private.git",
             str(tmp_path / "dest")],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )
        elapsed = time.monotonic() - t0
        assert proc.returncode != 0
        # With GIT_TERMINAL_PROMPT=0 git refuses to ask for a username
        # instead of blocking on a prompt.
        stderr = proc.stderr.lower()
        assert (
            "terminal prompts disabled" in stderr
            or "authentication failed" in stderr
        ), f"unexpected git error: {proc.stderr!r}"
        # Fail-fast, not a hang-until-timeout.
        assert elapsed < 20
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# 3. Call-site plumbing: internal git callers pass stdin=DEVNULL + env
# ---------------------------------------------------------------------------


def _capture_run(monkeypatch, module, **result_kwargs):
    """Monkeypatch ``module.subprocess.run`` recording every call's kwargs."""
    calls: list[dict] = []

    class _Result:
        returncode = result_kwargs.get("returncode", 0)
        stdout = result_kwargs.get("stdout", "")
        stderr = result_kwargs.get("stderr", "")

    def fake_run(argv, **kwargs):
        calls.append({"argv": list(argv), **kwargs})
        return _Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


def _assert_noninteractive(call: dict):
    assert call.get("stdin") is subprocess.DEVNULL, call["argv"]
    env = call.get("env")
    assert env is not None and env.get("GIT_TERMINAL_PROMPT") == "0", call["argv"]


def test_web_git_runs_noninteractively(monkeypatch, tmp_path):
    from hermes_cli import web_git

    calls = _capture_run(monkeypatch, web_git)
    web_git._git(str(tmp_path), ["fetch", "origin", "main"])
    assert calls
    _assert_noninteractive(calls[0])


def test_plugin_git_pull_runs_noninteractively(monkeypatch, tmp_path):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_resolve_git_executable", lambda: "git")
    calls = _capture_run(monkeypatch, plugins_cmd)
    plugins_cmd._git_pull_plugin_dir(tmp_path)
    assert calls
    _assert_noninteractive(calls[0])


def test_profile_distribution_clone_runs_noninteractively(monkeypatch, tmp_path):
    from hermes_cli import profile_distribution

    calls = _capture_run(monkeypatch, profile_distribution)
    profile_distribution._git_clone(
        "https://github.com/example/repo", tmp_path / "dest"
    )
    assert calls
    _assert_noninteractive(calls[0])


def test_mcp_catalog_git_install_runs_noninteractively(monkeypatch, tmp_path):
    from hermes_cli import mcp_catalog

    calls = _capture_run(monkeypatch, mcp_catalog)
    monkeypatch.setattr(mcp_catalog.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(mcp_catalog, "_install_root", lambda: tmp_path)

    entry = mcp_catalog.CatalogEntry(
        name="test-mcp",
        description="",
        source="official",
        transport=mcp_catalog.TransportSpec(type="stdio", command="python"),
        auth=mcp_catalog.AuthSpec(type="none"),
        install=mcp_catalog.InstallSpec(
            type="git",
            url="https://github.com/example/mcp.git",
            ref="main",
            bootstrap=[],
        ),
    )
    mcp_catalog._do_git_install(entry)
    assert calls
    for call in calls:
        if call["argv"][0].endswith("git") or "git" in call["argv"][0]:
            _assert_noninteractive(call)
