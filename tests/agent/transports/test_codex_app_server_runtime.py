"""Tests for the optional codex app-server runtime gate.

These are unit tests for the api_mode rewriter and the wire-level transport
module. They do NOT require the `codex` CLI to be installed — that's
covered by a separate live test gated on `codex --version`.
"""

from __future__ import annotations

import sys
import threading

import pytest

from hermes_cli.runtime_provider import (
    _VALID_API_MODES,
    _maybe_apply_codex_app_server_runtime,
)


class TestApiModeRegistration:
    """The new api_mode must be registered or downstream parsing rejects it."""

    def test_codex_app_server_is_a_valid_api_mode(self) -> None:
        assert "codex_app_server" in _VALID_API_MODES



class TestMaybeApplyCodexAppServerRuntime:
    """The opt-in helper that rewrites api_mode → codex_app_server."""

    @pytest.mark.parametrize(
        "model_cfg",
        [
            None,
            {},
            {"openai_runtime": ""},
            {"openai_runtime": "auto"},
            {"openai_runtime": "AUTO"},
            {"other_key": "codex_app_server"},  # wrong key
        ],
    )
    def test_default_off_for_openai(self, model_cfg) -> None:
        """Default behavior is preserved when the flag is unset/auto."""
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai", api_mode="chat_completions", model_cfg=model_cfg
        )
        assert got == "chat_completions"

    def test_opt_in_rewrites_openai(self) -> None:
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai",
            api_mode="chat_completions",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "codex_app_server"



    @pytest.mark.parametrize(
        "provider",
        [
            "anthropic",
            "openrouter",
            "xai",
            "qwen-oauth",
            "opencode-zen",
            "bedrock",
            "",
        ],
    )
    def test_other_providers_never_rerouted(self, provider) -> None:
        """Non-OpenAI providers MUST NOT be rerouted even with the flag set —
        codex's app-server can only run OpenAI/Codex auth flows."""
        got = _maybe_apply_codex_app_server_runtime(
            provider=provider,
            api_mode="anthropic_messages",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "anthropic_messages", (
            f"provider={provider!r} should not be rerouted to codex_app_server"
        )


class TestCodexAppServerModule:
    """Module-surface tests for the JSON-RPC speaker. Don't require codex CLI."""

    @pytest.mark.parametrize(
        "config,kanban_root,expected",
        [
            (
                {},
                None,
                [],
            ),
            (
                {
                    "model_reasoning_effort": "high",
                    "sandbox_mode": "workspace-write",
                    "ask_for_approval": "on-request",
                    "network_access": True,
                    "writable_roots": ["/work", "/work", "/tmp/😀"],
                },
                "/kanban",
                [
                    'model_reasoning_effort="high"',
                    'sandbox_mode="workspace-write"',
                    'approval_policy="on-request"',
                    "sandbox_workspace_write.network_access=false",
                    'sandbox_workspace_write.writable_roots=["/kanban"]',
                ],
            ),
            (
                {
                    "sandbox_mode": "danger-full-access",
                    "network_access": False,
                    "writable_roots": ["/ignored"],
                },
                "/kanban",
                [
                    'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=false",
                    'sandbox_workspace_write.writable_roots=["/kanban"]',
                ],
            ),
            (
                {
                    "sandbox_mode": "danger-full-access",
                    "network_access": True,
                    "writable_roots": ["/work"],
                },
                None,
                ['sandbox_mode="danger-full-access"'],
            ),
            (
                {},
                "/kanban",
                [
                    'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=false",
                    'sandbox_workspace_write.writable_roots=["/kanban"]',
                ],
            ),
        ],
    )
    def test_launch_config_resolves_session_scoped_overrides_without_conflicts(
        self, config, kanban_root, expected
    ) -> None:
        from agent.transports.codex_app_server import resolve_codex_app_server_args

        args = resolve_codex_app_server_args(
            config=config,
            kanban_root=kanban_root,
        )

        assert args == [part for value in expected for part in ("-c", value)]

    @pytest.mark.parametrize(
        "config",
        [
            {"model_reasoning_effort": "extreme"},
            {"sandbox_mode": "host-write"},
            {"ask_for_approval": "always"},
            {"sandbox_mode": "workspace-write", "network_access": "yes"},
            {"sandbox_mode": "workspace-write", "writable_roots": ["relative/path"]},
        ],
    )
    def test_launch_config_rejects_invalid_values(self, config) -> None:
        from agent.transports.codex_app_server import resolve_codex_app_server_args

        with pytest.raises(ValueError, match="model.codex_app_server"):
            resolve_codex_app_server_args(config=config)

    def test_check_binary_handles_missing_executable(self) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        ok, msg = check_codex_binary(codex_bin="/nonexistent/codex/binary/path")
        assert ok is False
        assert msg



class TestCodexAppServerClose:
    """Lifecycle tests for retiring the optional Codex app-server transport."""

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.platforms("posix")  # start_new_session/setsid is POSIX-only
    def test_close_reaps_independent_descendant_process_group(self, tmp_path):
        """A Codex-owned MCP child that calls setsid must not survive close().

        Killing only the app-server root lets independently grouped stdio MCP
        descendants live past client retirement. The fake codex binary below
        spawns a long-lived child in its own session, records its PID, and exits
        promptly on root SIGTERM; close() must still reap the child.
        """
        import os
        import stat
        import sys
        import time

        import psutil

        from agent.transports.codex_app_server import CodexAppServerClient

        child_pid_file = tmp_path / "child.pid"
        fake_codex = tmp_path / "fake_codex.py"
        fake_codex.write_text(
            f"#!{sys.executable}\n"
            """
import os
import signal
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True,
)
with open(os.environ["CHILD_PID_FILE"], "w", encoding="utf-8") as fh:
    fh.write(str(child.pid))

def _exit(_sig, _frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, _exit)
while True:
    time.sleep(1)
""".lstrip()
        )
        fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)

        client = CodexAppServerClient(
            codex_bin=str(fake_codex),
            env={"CHILD_PID_FILE": str(child_pid_file)},
        )
        try:
            deadline = time.time() + 5
            while not child_pid_file.exists() and time.time() < deadline:
                time.sleep(0.05)
            assert child_pid_file.exists(), "fake codex did not report child PID"
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            assert psutil.pid_exists(child_pid)

            client.close(timeout=1.0)

            deadline = time.time() + 5
            while psutil.pid_exists(child_pid) and time.time() < deadline:
                time.sleep(0.05)
            assert not psutil.pid_exists(child_pid), (
                "Codex app-server close() left an independently grouped "
                f"descendant alive: pid={child_pid}"
            )
        finally:
            client.close(timeout=0.1)
            if child_pid_file.exists():
                try:
                    os.kill(int(child_pid_file.read_text(encoding="utf-8")), 9)
                except ProcessLookupError:
                    pass

    def test_close_escalates_to_tree_kill_when_root_ignores_sigterm(self, monkeypatch):
        """When the root outlives the graceful wait, close() must kill the tree AND
        run the post-kill wait so a codex ignoring SIGTERM cannot leak."""
        import subprocess
        from unittest import mock

        from agent.transports import codex_app_server as mod

        proc = mock.MagicMock()
        proc.pid = 4242
        proc.stdin = None
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="codex", timeout=0.01), 0]
        killed: list[int] = []
        monkeypatch.setattr(mod, "kill_process_tree", lambda pid, **kw: killed.append(pid) or True)
        monkeypatch.setattr(mod, "_snapshot_descendants", lambda pid: [])

        client = mod.CodexAppServerClient.__new__(mod.CodexAppServerClient)
        client._proc = proc
        client._closed = False
        client._pending, client._pending_lock = {}, threading.Lock()
        client.close(timeout=0.01)

        assert killed == [4242]
        assert proc.wait.call_args_list == [mock.call(timeout=0.01), mock.call(timeout=1.0)]


class TestSpawnEnvIsolation:
    """The codex spawn must NOT rewrite HOME — codex's shell tool spawns
    subprocesses (gh, git, npm, aws, gcloud, ...) that need to find their
    config in the real user $HOME. CODEX_HOME isolates codex's own state,
    HOME stays unchanged.

    OpenClaw hit this footgun (openclaw/openclaw#81562) — they were
    rewriting HOME to a synthetic per-agent dir alongside CODEX_HOME,
    and then `gh auth status` / git config / etc. all broke inside codex
    shell calls. We avoid the same bug by only overlaying CODEX_HOME and
    RUST_LOG on top of os.environ.copy().
    """


    def test_spawn_env_sets_CODEX_HOME_when_provided(self, monkeypatch):
        """CODEX_HOME isolation must still work — that's the whole point
        of the codex_home arg."""
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(
            codex_bin="codex", codex_home="/tmp/profile/codex"
        )
        client._closed = True

        assert captured["env"].get("CODEX_HOME") == "/tmp/profile/codex"
        # And HOME still passes through unchanged
        assert captured["env"].get("HOME") == "/users/alice"

    def test_kanban_worker_adds_only_kanban_writable_root(self, monkeypatch):
        """Codex-runtime Kanban workers need to write board state outside
        their scratch/worktree workspace, but should not fall back to
        danger-full-access. Hermes passes a narrow app-server config override
        for the Kanban root only and keeps it authoritative over legacy args.
        """
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["cmd"] = list(cmd)
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")
        monkeypatch.setenv("HERMES_HOME", "/users/alice/.hermes/profiles/backend-worker")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
        )

        client = cas.CodexAppServerClient(
            codex_bin="codex",
            extra_args=["-c", 'sandbox_mode="danger-full-access"'],
        )
        client._closed = True

        cmd = captured["cmd"]
        assert cmd[:2] == ["codex", "app-server"]
        assert cmd[-6:] == [
            "-c",
            'sandbox_mode="workspace-write"',
            "-c",
            "sandbox_workspace_write.network_access=false",
            "-c",
            'sandbox_workspace_write.writable_roots=["/users/alice/.hermes/kanban/boards/smoke"]',
        ]

    @pytest.mark.parametrize(
        "permissive,session_expected",
        [
            (
                {
                    "sandbox_mode": "danger-full-access",
                    "network_access": True,
                    "writable_roots": ["/"],
                },
                ['sandbox_mode="danger-full-access"'],
            ),
            (
                {
                    "sandbox_mode": "workspace-write",
                    "network_access": True,
                    "writable_roots": ["/"],
                },
                [
                    'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=true",
                    'sandbox_workspace_write.writable_roots=["/"]',
                ],
            ),
        ],
    )
    def test_profile_config_widens_normal_session_but_not_owned_kanban_worker(
        self, monkeypatch, permissive, session_expected
    ):
        """The same permissive ``model.codex_app_server`` config is honoured for a
        normal session, while a dispatcher-owned worker stays bounded to
        workspace-write + no network + only the dispatcher's Kanban root."""
        import subprocess
        from agent.transports import codex_app_server as cas

        spawned = []

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                spawned.append(list(cmd))
                self.stdin = self.stdout = self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HERMES_HOME", "/users/alice/.hermes")
        for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_ROOT"):
            monkeypatch.delenv(key, raising=False)

        normal = cas.CodexAppServerClient(codex_bin="codex", config=permissive)
        normal._closed = True

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", "/users/alice/.hermes/kanban/boards/smoke/kanban.db"
        )
        worker = cas.CodexAppServerClient(codex_bin="codex", config=permissive)
        worker._closed = True

        def overrides(cmd):
            return [
                cmd[i + 1]
                for i, part in enumerate(cmd)
                if part == "-c" and not cmd[i + 1].startswith("mcp_servers.")
            ]

        assert overrides(spawned[0]) == session_expected
        assert overrides(spawned[1]) == [
            'sandbox_mode="workspace-write"',
            "sandbox_workspace_write.network_access=false",
            'sandbox_workspace_write.writable_roots=["/users/alice/.hermes/kanban/boards/smoke"]',
        ]

    @pytest.mark.parametrize("context", ["delegated_child", "non_dispatcher_owned"])
    def test_kanban_descendants_stay_bounded_by_permissive_profile_config(
        self, monkeypatch, context
    ):
        """A delegate child (or cron run) inside a Kanban worker does not own the
        task, but is still bounded: the generic profile config must not widen it."""
        import subprocess
        from agent.delegation_context import (
            delegated_child_context,
            non_dispatcher_owned_context,
        )
        from agent.transports import codex_app_server as cas

        spawned = []

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                spawned.append(list(cmd))
                self.stdin = self.stdout = self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HERMES_HOME", "/users/alice/.hermes")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", "/users/alice/.hermes/kanban/boards/smoke/kanban.db"
        )
        enter = {
            "delegated_child": delegated_child_context,
            "non_dispatcher_owned": non_dispatcher_owned_context,
        }[context]
        permissive = {
            "sandbox_mode": "danger-full-access",
            "network_access": True,
            "writable_roots": ["/"],
        }
        with enter():
            client = cas.CodexAppServerClient(codex_bin="codex", config=permissive)
        client._closed = True

        cmd = spawned[0]
        assert [
            cmd[i + 1]
            for i, part in enumerate(cmd)
            if part == "-c" and not cmd[i + 1].startswith("mcp_servers.")
        ] == [
            'sandbox_mode="workspace-write"',
            "sandbox_workspace_write.network_access=false",
        ]
        # Bounded, but no writable root and no ownership of the task via MCP env.
        assert not any(part.startswith("mcp_servers.") for part in cmd)

    @pytest.mark.parametrize(
        "config,db,expected",
        [
            (None, None, []),
            (
                {
                    "sandbox_mode": "workspace-write",
                    "network_access": True,
                    "writable_roots": ["/work"],
                },
                None,
                [
                    'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=true",
                    'sandbox_workspace_write.writable_roots=["/work"]',
                ],
            ),
            (
                {
                    "sandbox_mode": "danger-full-access",
                    "network_access": True,
                    "writable_roots": ["/"],
                },
                "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
                [
                    'sandbox_mode="workspace-write"',
                    "sandbox_workspace_write.network_access=false",
                ],
            ),
        ],
        ids=["marker_only", "marker_only_permissive", "worker_descendant"],
    )
    def test_delegate_marker_is_never_a_writable_root(
        self, monkeypatch, config, db, expected
    ):
        """The marker's value is the fenced (write-denied) root. A marker without
        Kanban lineage changes nothing vs. a normal session; a worker descendant
        (marker + board DB, no task) is clamped but gets no writable root."""
        import subprocess
        from agent.transports import codex_app_server as cas

        spawned = []

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                spawned.append(list(cmd))
                self.stdin = self.stdout = self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HERMES_HOME", "/users/alice/.hermes")
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_ROOT", raising=False)
        if db:
            monkeypatch.setenv("HERMES_KANBAN_DB", db)
        else:
            monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        marker = "/users/alice/.hermes"
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", marker)
        client = cas.CodexAppServerClient(codex_bin="codex", config=config)
        client._closed = True

        cmd = spawned[0]
        applied = [
            cmd[i + 1]
            for i, part in enumerate(cmd)
            if part == "-c" and not cmd[i + 1].startswith("mcp_servers.")
        ]
        assert applied == expected
        assert not any(marker in item for item in applied)
        assert not any(part.startswith("mcp_servers.") for part in cmd)

    @pytest.mark.parametrize(
        "config",
        [{"network_access": "yes"}, {"writable_roots": ["relative"]}],
    )
    def test_invalid_values_raise_even_when_kanban_clamps(self, config):
        from agent.transports import codex_app_server as cas

        with pytest.raises(ValueError):
            cas.resolve_codex_app_server_args(config=config, kanban_root="/k")

    @pytest.mark.parametrize(
        "config,kanban_root",
        [
            ({"sandbox_mode": "danger-full-access"}, "/k"),
            ({"network_access": True, "writable_roots": ["/a"]}, None),
        ],
    )
    def test_ignored_keys_log_a_warning(self, config, kanban_root, caplog):
        import logging
        from agent.transports import codex_app_server as cas

        with caplog.at_level(logging.WARNING, logger=cas.logger.name):
            cas.resolve_codex_app_server_args(config=config, kanban_root=kanban_root)

        assert any("ignored" in r.getMessage() for r in caplog.records)


class TestSpawnEnvSecretStripping:
    """codex app-server routes its spawn env through hermes_subprocess_env(
    inherit_credentials=True) instead of a raw os.environ.copy().

    codex is a model-driving CLI executor: it legitimately needs LLM provider
    credentials to authenticate, but it must NOT inherit Tier-1 Hermes secrets
    (gateway bot tokens, GitHub/infra auth, dashboard session token) or the
    dynamic-internal secrets (AUXILIARY_*_API_KEY / _BASE_URL side-LLM keys,
    GATEWAY_RELAY_* relay-auth) — a coding subprocess has no use for those and
    a model-controlled action could exfiltrate them. This closes the #29157
    sibling spawn-site gap (copilot_acp_client already routes through the
    helper; codex app-server predated it).
    """

    @staticmethod
    def _capture_spawn_env(monkeypatch):
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True
        return captured["env"]

    def test_tier1_and_internal_secrets_stripped_from_spawn_env(self, monkeypatch):
        for var, val in {
            "GH_TOKEN": "ghp-secret",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "HERMES_DASHBOARD_SESSION_TOKEN": "dash-secret",
            "AUXILIARY_VISION_API_KEY": "aux-secret",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "GATEWAY_RELAY_ID": "relay-id",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery",
        }.items():
            monkeypatch.setenv(var, val)

        env = self._capture_spawn_env(monkeypatch)
        for var in (
            "GH_TOKEN", "TELEGRAM_BOT_TOKEN", "MODAL_TOKEN_SECRET",
            "HERMES_DASHBOARD_SESSION_TOKEN", "AUXILIARY_VISION_API_KEY",
            "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_DELIVERY_KEY",
        ):
            assert var not in env, f"{var} leaked into codex app-server spawn env"

    def test_provider_credentials_still_reach_codex(self, monkeypatch):
        """codex authenticates against the model endpoint — provider keys must
        still flow through (inherit_credentials=True)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-codex-needs-this")
        env = self._capture_spawn_env(monkeypatch)
        assert env.get("OPENAI_API_KEY") == "sk-codex-needs-this"

