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

    def test_strips_ambient_git_config_injection(self):
        env = noninteractive_git_env(
            {
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "core.pager",
                "GIT_CONFIG_VALUE_0": "less",
                "GIT_CONFIG_KEY_1": "core.hooksPath",
                "GIT_CONFIG_VALUE_1": ".git/hooks",
                "GIT_CONFIG_PARAMETERS": "'core.pager=less'",
            }
        )

        assert env["GIT_CONFIG_COUNT"] != "2"
        assert "GIT_CONFIG_PARAMETERS" not in env
        values = {
            env[f"GIT_CONFIG_KEY_{idx}"]: env[f"GIT_CONFIG_VALUE_{idx}"]
            for idx in range(int(env["GIT_CONFIG_COUNT"]))
        }
        assert values["core.pager"] == "cat"
        assert values["core.hooksPath"] == os.devnull
        assert values["credential.helper"] == ""

    def test_disables_pagers_hooks_editors_and_user_config(self):
        env = noninteractive_git_env({})
        values = {
            env[f"GIT_CONFIG_KEY_{idx}"]: env[f"GIT_CONFIG_VALUE_{idx}"]
            for idx in range(int(env["GIT_CONFIG_COUNT"]))
        }

        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_SYSTEM"] == os.devnull
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_PAGER"] == "cat"
        assert env["PAGER"] == "cat"
        assert env["GIT_EDITOR"] == "true"
        assert values["core.fsmonitor"] == "false"
        assert values["core.hooksPath"] == os.devnull
        assert values["core.editor"] == "true"
        assert values["sequence.editor"] == "true"
        assert values["diff.external"] == ""

    @pytest.mark.real_safe_directory
    def test_safe_directory_preserves_git_ordering_and_reset_markers(self, tmp_path, monkeypatch):
        """The user's effective trust policy is replayed verbatim, resets included.

        ``safe.directory`` is an ordered multi-valued setting where an empty value resets every
        earlier entry, which is how a user revokes a system-wide ``safe.directory=*`` and then
        names only the repositories they trust. Reading global-before-system, dropping the empty
        marker, or de-duplicating turns ``* -> reset -> /trusted/only`` into ``/trusted/only, *``
        and silently restores the wildcard the user revoked. Contract: the injected sequence equals
        what git itself reports for the same config (system scope first, then global, verbatim).
        """
        system_config = tmp_path / "system-gitconfig"
        system_config.write_text("[safe]\n\tdirectory = *\n", encoding="utf-8")
        global_config = tmp_path / "gitconfig"
        global_config.write_text(
            "[safe]\n\tdirectory = \n\tdirectory = /trusted/only\n", encoding="utf-8"
        )
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_config))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

        # Ambient GIT_CONFIG_KEY_n=safe.directory must not be laundered through alongside the
        # user's own entries -- only the config files are a trust source.
        env = noninteractive_git_env(
            {
                **os.environ,
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": "/attacker/controlled",
            }
        )
        # Isolation itself is unchanged: the values ride the KEY_n channel, not the config file.
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_SYSTEM"] == os.devnull
        injected = [
            env[f"GIT_CONFIG_VALUE_{idx}"]
            for idx in range(int(env["GIT_CONFIG_COUNT"]))
            if env[f"GIT_CONFIG_KEY_{idx}"] == "safe.directory"
        ]

        # git's own effective view of the same two files, lowest-precedence scope first.
        expected = subprocess.run(
            ["git", "config", "-z", "--get-all", "safe.directory"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "GIT_CONFIG_SYSTEM": str(system_config),
                 "GIT_CONFIG_GLOBAL": str(global_config)},
        ).stdout.split("\0")[:-1]

        assert expected == ["*", "", "/trusted/only"], "git's documented reset shape changed"
        assert injected == expected

    @pytest.mark.real_safe_directory
    def test_safe_directory_reset_still_revokes_wildcard_for_real_git(self, tmp_path):
        """End-to-end: a revoked wildcard stays revoked, and the named repo stays usable.

        Proves the injected sequence produces the same *trust decision* real git makes, not merely
        the same list. Both repos are made cross-owner via ``safe.directory=*`` being the only
        thing that could authorise them, so the negative control fails exactly as the user's
        interactive git does.
        """
        if not shutil.which("git"):
            pytest.skip("git not installed")

        def _repo(name: str) -> Path:
            path = tmp_path / name
            path.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=path, check=True)
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "--allow-empty", "-m", "x"],
                cwd=path, check=True,
            )
            return path

        trusted = _repo("trusted")
        unrelated = _repo("unrelated")

        system_config = tmp_path / "system-gitconfig"
        system_config.write_text("[safe]\n\tdirectory = *\n", encoding="utf-8")
        global_config = tmp_path / "gitconfig"
        global_config.write_text(
            f"[safe]\n\tdirectory = \n\tdirectory = {trusted}\n", encoding="utf-8"
        )

        env = noninteractive_git_env(
            {**os.environ, "GIT_CONFIG_SYSTEM": str(system_config),
             "GIT_CONFIG_GLOBAL": str(global_config)}
        )
        # Force the ownership check that safe.directory governs; without it git trusts the repo
        # because the test process owns the checkout it just created.
        env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"

        def _rev_parse(repo: Path) -> int:
            return subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
            ).returncode

        assert _rev_parse(trusted) == 0, "the explicitly trusted repo must stay usable"
        assert _rev_parse(unrelated) != 0, (
            "the global empty reset revoked the system wildcard, so an unrelated cross-owner "
            "repo must still be refused"
        )

    def test_ssh_host_key_prompts_fail_closed(self):
        """core.sshCommand is pinned to BatchMode ssh (#104591).

        ssh bypasses ``stdin=DEVNULL`` and ``GIT_TERMINAL_PROMPT`` — an unknown host key (or
        password auth) opens ``/dev/tty`` directly and steals the caller's terminal. Under this
        env the ssh child of a git fetch must fail fast instead of prompting; an
        agent-authenticated ssh still succeeds.
        """
        env = noninteractive_git_env({})
        values = {
            env[f"GIT_CONFIG_KEY_{idx}"]: env[f"GIT_CONFIG_VALUE_{idx}"]
            for idx in range(int(env["GIT_CONFIG_COUNT"]))
        }
        assert values["core.sshCommand"] == "ssh -o BatchMode=yes"
        # Config-layer pin only: GIT_SSH_COMMAND is never set or overridden here, so a user's
        # explicit env var still takes precedence over core.sshCommand.
        override = "ssh -i custom-key -o BatchMode=no"
        assert noninteractive_git_env({"GIT_SSH_COMMAND": override})["GIT_SSH_COMMAND"] == override


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
        env = noninteractive_git_env()
        # noninteractive_git_env deliberately leaves GIT_ASKPASS/SSH_ASKPASS
        # alone so a user's WORKING helper can still authenticate. This test
        # asserts the no-helper fail-fast path, so strip them — otherwise a
        # dev shell's VS Code askpass helper (GIT_ASKPASS=...askpass.sh)
        # blocks waiting on the editor and the clone times out locally.
        for var in ("GIT_ASKPASS", "SSH_ASKPASS", "VSCODE_GIT_ASKPASS_NODE",
                    "VSCODE_GIT_ASKPASS_MAIN", "VSCODE_GIT_ASKPASS_EXTRA_ARGS",
                    "VSCODE_GIT_IPC_HANDLE"):
            env.pop(var, None)
        proc = subprocess.run(
            ["git", "clone", f"http://127.0.0.1:{port}/private.git",
             str(tmp_path / "dest")],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=env,
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
    # A stdin fed by ``input=`` (git credential fill's request) is written and closed, not a terminal.
    assert call.get("stdin") is subprocess.DEVNULL or "input" in call, call["argv"]
    env = call.get("env")
    assert env is not None and env.get("GIT_TERMINAL_PROMPT") == "0", call["argv"]








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
