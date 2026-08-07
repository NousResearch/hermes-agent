"""Tests for the optional codex app-server runtime gate.

These are unit tests for the api_mode rewriter and the wire-level transport
module. They do NOT require the `codex` CLI to be installed — that's
covered by a separate live test gated on `codex --version`.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.runtime_provider import (
    _VALID_API_MODES,
    _maybe_apply_codex_app_server_runtime,
)


class TestApiModeRegistration:
    """The new api_mode must be registered or downstream parsing rejects it."""

    def test_codex_app_server_is_a_valid_api_mode(self) -> None:
        assert "codex_app_server" in _VALID_API_MODES

    def test_existing_api_modes_still_present(self) -> None:
        # Regression guard: don't accidentally delete other api_modes when
        # touching this set.
        for mode in (
            "chat_completions",
            "codex_responses",
            "anthropic_messages",
            "bedrock_converse",
        ):
            assert mode in _VALID_API_MODES


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




    def test_check_binary_handles_missing_executable(self) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        ok, msg = check_codex_binary(codex_bin="/nonexistent/codex/binary/path")
        assert ok is False
        assert "not found" in msg.lower() or "no such" in msg.lower()

    def test_codex_error_class_is_runtimeerror(self) -> None:
        from agent.transports.codex_app_server import CodexAppServerError

        err = CodexAppServerError(code=-32600, message="boom")
        assert isinstance(err, RuntimeError)
        assert "boom" in str(err)
        assert "-32600" in str(err)


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

    def test_spawn_env_preserves_HOME(self, monkeypatch):
        """The spawn env must contain the parent process's HOME unchanged.
        Verifies via a subprocess-monkey-patch."""
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                # Provide minimal Popen surface so __init__ doesn't crash
                # on attribute access during construction.
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def communicate(self, input=None, timeout=None):
                return "", ""

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True  # so close() is a no-op

        # The spawn env must have HOME=/users/alice unchanged
        assert captured["env"].get("HOME") == "/users/alice", (
            f"HOME got rewritten in codex spawn env: "
            f"{captured['env'].get('HOME')!r}. Codex's shell tool's "
            "subprocesses (gh, git, aws, npm) need the user's real HOME."
        )

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

            def communicate(self, input=None, timeout=None):
                return "", ""

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

    @staticmethod
    def _spawn_kanban_client(monkeypatch, workspace):
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["cmd"] = list(cmd)
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def poll(self):
                return None

            def communicate(self, input=None, timeout=None):
                return "", ""

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
        )
        client = cas.CodexAppServerClient(codex_bin="codex", workspace_cwd=str(workspace))
        client._closed = True
        return json.loads(
            next(
                part.split("=", 1)[1]
                for part in captured["cmd"]
                if part.startswith("sandbox_workspace_write.writable_roots=")
            )
        )

    def test_kanban_linked_worktree_adds_board_and_git_writable_roots(
        self, monkeypatch, tmp_path
    ):
        """Linked-worktree workers need the board and shared Git metadata,
        without falling back to danger-full-access.
        """
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}
        repo = tmp_path / "repo"
        workspace = repo / ".worktrees" / "t_smoke"
        git_dir = repo / ".git" / "worktrees" / "t_smoke"
        workspace.mkdir(parents=True)
        git_dir.mkdir(parents=True)
        (workspace / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{workspace / '.git'}\n")

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

            def communicate(self, input=None, timeout=None):
                return "", ""

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
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
        )

        client = cas.CodexAppServerClient(
            codex_bin="codex", workspace_cwd=str(workspace)
        )
        client._closed = True

        cmd = captured["cmd"]
        assert cmd[:2] == ["codex", "app-server"]
        assert 'sandbox_mode="workspace-write"' in cmd
        assert (
            "sandbox_workspace_write.writable_roots="
            f'["/users/alice/.hermes/kanban/boards/smoke", "{repo / ".git"}"]'
            in cmd
        )
        assert "sandbox_workspace_write.network_access=false" in cmd
        assert all("danger" not in part for part in cmd)

    def test_kanban_worker_rejects_self_consistent_foreign_git_admin(
        self, monkeypatch, tmp_path
    ):
        """Workspace-controlled Git metadata cannot authorize a foreign root."""
        workspace = tmp_path / "workspace"
        git_common = tmp_path / "foreign-admin.git"
        git_dir = git_common / "worktrees" / "slot"
        marker = workspace / ".git"
        workspace.mkdir()
        git_dir.mkdir(parents=True)
        marker.write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{marker}\n")

        assert self._spawn_kanban_client(monkeypatch, workspace) == [
            "/users/alice/.hermes/kanban/boards/smoke"
        ]

    @pytest.mark.parametrize(
        "case",
        (
            "root_gitdir",
            "symlink_marker",
            "symlink_commondir",
            "symlink_backlink",
            "malformed_metadata",
            "unreadable_metadata",
            "foreign_backlink",
        ),
    )
    def test_kanban_linked_worktree_rejects_untrusted_git_metadata(
        self, monkeypatch, tmp_path, case
    ):
        """Untrusted worktree metadata cannot widen Codex's writable roots."""
        repo = tmp_path / "repo"
        workspace = repo / ".worktrees" / "t_smoke"
        git_dir = repo / ".git" / "worktrees" / "t_smoke"
        marker = workspace / ".git"
        workspace.mkdir(parents=True)
        git_dir.mkdir(parents=True)
        marker.write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{marker}\n")

        if case == "root_gitdir":
            marker.write_text("gitdir: /\n")
        elif case == "symlink_marker":
            marker.unlink()
            symlink_target = tmp_path / "marker-target"
            symlink_target.write_text(f"gitdir: {git_dir}\n")
            marker.symlink_to(symlink_target)
        elif case == "symlink_commondir":
            (git_dir / "commondir").unlink()
            (git_dir / "commondir").symlink_to(repo / ".git")
        elif case == "symlink_backlink":
            (git_dir / "gitdir").unlink()
            (git_dir / "gitdir").symlink_to(marker)
        elif case == "malformed_metadata":
            (git_dir / "commondir").write_text("\n")
        elif case == "unreadable_metadata":
            (git_dir / "commondir").chmod(0)
        elif case == "foreign_backlink":
            foreign_marker = tmp_path / "foreign" / ".git"
            foreign_marker.parent.mkdir()
            foreign_marker.write_text("gitdir: ignored\n")
            (git_dir / "gitdir").write_text(f"{foreign_marker}\n")

        assert self._spawn_kanban_client(monkeypatch, workspace) == [
            "/users/alice/.hermes/kanban/boards/smoke"
        ]


    def test_explicit_worktree_target_can_use_custom_directory_name(
        self, monkeypatch, tmp_path
    ):
        """Dispatcher-owned worktree metadata must not require task-id basenames."""
        import subprocess
        from agent.transports import codex_app_server as cas

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        (repo / "README").write_text("seed\n")
        subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=Hermes", "-c", "user.email=hermes@example.invalid", "commit", "-qm", "seed"],
            check=True,
        )
        workspace = repo / ".worktrees" / "custom-target"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "worktree",
                "add",
                "-qb",
                "wt/custom-target",
                str(workspace),
            ],
            check=True,
        )

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["cmd"] = list(cmd)
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def poll(self):
                return None

            def communicate(self, input=None, timeout=None):
                return "", ""

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        real_popen = subprocess.Popen

        def _popen(cmd, *args, **kwargs):
            if cmd and cmd[0] == "git":
                return real_popen(cmd, *args, **kwargs)
            return FakePopen(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", _popen)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE_KIND", "worktree")
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
        )

        cas.CodexAppServerClient(codex_bin="codex", workspace_cwd=str(workspace))
        writable = json.loads(
            next(
                part.split("=", 1)[1]
                for part in captured["cmd"]
                if part.startswith("sandbox_workspace_write.writable_roots=")
            )
        )
        assert str(repo / ".git") in writable


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

            def communicate(self, input=None, timeout=None):
                return "", ""

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

