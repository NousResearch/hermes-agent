"""Tests for the optional codex app-server runtime gate.

These are unit tests for the api_mode rewriter and the wire-level transport
module. They do NOT require the `codex` CLI to be installed — that's
covered by a separate live test gated on `codex --version`.
"""

from __future__ import annotations

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

    def test_opt_in_rewrites_openai_codex(self) -> None:
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai-codex",
            api_mode="codex_responses",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "codex_app_server"

    def test_case_insensitive(self) -> None:
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai",
            api_mode="chat_completions",
            model_cfg={"openai_runtime": "Codex_App_Server"},
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

    def test_module_imports(self) -> None:
        from agent.transports import codex_app_server

        assert codex_app_server.MIN_CODEX_VERSION >= (0, 1, 0)
        assert callable(codex_app_server.parse_codex_version)
        assert callable(codex_app_server.check_codex_binary)

    def test_parse_codex_version_valid(self) -> None:
        from agent.transports.codex_app_server import parse_codex_version

        assert parse_codex_version("codex-cli 0.130.0") == (0, 130, 0)
        assert parse_codex_version("codex-cli 1.2.3 (extra metadata)") == (1, 2, 3)
        assert parse_codex_version("codex 99.0.1\n") == (99, 0, 1)

    def test_parse_codex_version_invalid(self) -> None:
        from agent.transports.codex_app_server import parse_codex_version

        assert parse_codex_version("nope") is None
        assert parse_codex_version("") is None
        assert parse_codex_version(None) is None  # type: ignore[arg-type]

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

    def test_kanban_normal_checkout_adds_exact_git_metadata_root(
        self, monkeypatch, tmp_path
    ):
        """Normal checkouts grant only their exact in-workspace Git metadata."""
        import json
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
        board_dir = tmp_path / "boards" / "smoke"
        board_dir.mkdir(parents=True)
        board_db = board_dir / "kanban.db"
        board_db.touch()
        workspace = tmp_path / "workspaces" / "t_smoke"
        workspace.mkdir(parents=True)
        (workspace / ".git").mkdir()

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))

        client = cas.CodexAppServerClient(
            codex_bin="codex", workspace_cwd=str(workspace)
        )
        client._closed = True

        cmd = captured["cmd"]
        assert cmd[:2] == ["codex", "app-server"]
        assert 'sandbox_mode="workspace-write"' in cmd
        root_arg = next(
            part
            for part in cmd
            if part.startswith("sandbox_workspace_write.writable_roots=")
        )
        assert json.loads(root_arg.split("=", 1)[1]) == [
            str(board_dir.resolve()),
            str(workspace.resolve()),
            str((workspace / ".git").resolve()),
        ]
        assert "sandbox_workspace_write.network_access=false" in cmd
        assert all("danger" not in part for part in cmd)

    def test_kanban_worktree_adds_linked_git_metadata_writable_roots(
        self, tmp_path
    ):
        """A linked worktree can commit without repository-wide authority."""
        import subprocess
        from pathlib import Path

        from agent.transports.codex_app_server import _kanban_writable_roots

        repo = tmp_path / "repo"
        workspace = tmp_path / "task"
        sibling = tmp_path / "sibling"
        subprocess.run(["git", "init", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.name", "Hermes Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        (repo / "tracked.txt").write_text("seed\n")
        subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "seed"], check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-b", "task", str(workspace)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-b", "sibling", str(sibling)],
            check=True,
        )

        repo_git = repo / ".git"
        git_dir = Path(
            subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "--absolute-git-dir"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        sibling_git_dir = Path(
            subprocess.run(
                ["git", "-C", str(sibling), "rev-parse", "--absolute-git-dir"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        board_dir = tmp_path / "boards" / "smoke"
        board_dir.mkdir(parents=True)
        board_db = board_dir / "kanban.db"
        board_db.touch()
        roots = [
            Path(root)
            for root in _kanban_writable_roots(
                {
                    "HERMES_KANBAN_DB": str(board_db),
                    "HERMES_KANBAN_WORKSPACE": str(workspace),
                },
                workspace_cwd=str(workspace),
            )
        ]

        assert roots == [
            board_dir,
            workspace,
            git_dir,
            repo_git / "objects",
            repo_git / "refs" / "heads",
            repo_git / "logs" / "refs" / "heads",
        ]
        for forbidden in (repo_git / "config", repo_git / "hooks", sibling_git_dir):
            assert not any(
                forbidden == root or forbidden.is_relative_to(root) for root in roots
            )

        protected = [repo_git, *(path for path in repo_git.rglob("*") if path.is_dir())]
        for path in protected:
            path.chmod(0o555)
        for root in roots:
            if root.is_relative_to(repo_git):
                for path in (root, *(p for p in root.rglob("*") if p.is_dir())):
                    path.chmod(0o755)
        try:
            (workspace / "tracked.txt").write_text("task commit\n")
            subprocess.run(
                ["git", "-C", str(workspace), "add", "tracked.txt"], check=True
            )
            subprocess.run(
                ["git", "-C", str(workspace), "commit", "-m", "canary"],
                check=True,
            )
        finally:
            for path in protected:
                path.chmod(0o755)

        assert (
            subprocess.run(
                ["git", "-C", str(workspace), "log", "-1", "--format=%s"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == "canary"
        )

    def test_bare_home_linked_worktree_fails_closed(
        self, monkeypatch, tmp_path
    ):
        import subprocess
        from pathlib import Path

        from agent.transports.codex_app_server import _kanban_writable_roots

        home = tmp_path / "home"
        seed = tmp_path / "seed"
        workspace = tmp_path / "workspace"
        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()

        subprocess.run(["git", "init", "--bare", str(home)], check=True)
        subprocess.run(["git", "init", str(seed)], check=True)
        subprocess.run(
            ["git", "-C", str(seed), "config", "user.name", "Hermes Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(seed), "config", "user.email", "test@example.com"],
            check=True,
        )
        (seed / "tracked.txt").write_text("seed\n")
        subprocess.run(["git", "-C", str(seed), "add", "tracked.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(seed), "commit", "-m", "seed"], check=True
        )
        subprocess.run(
            ["git", "-C", str(seed), "push", str(home), "HEAD:main"],
            check=True,
        )
        subprocess.run(
            ["git", f"--git-dir={home}", "worktree", "add", str(workspace), "main"],
            check=True,
        )

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setenv("HOME", str(home))

        assert _kanban_writable_roots(
            {
                "HERMES_KANBAN_DB": str(db),
                "HERMES_KANBAN_WORKSPACE": str(workspace),
            },
            workspace_cwd=str(workspace),
        ) == []

    def test_normal_repository_is_not_treated_as_linked_worktree(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    def test_symlinked_normal_git_metadata_is_not_granted(self, tmp_path):
        from agent.transports.codex_app_server import _kanban_writable_roots

        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_git = tmp_path / "external-git"
        external_git.mkdir()
        (workspace / ".git").symlink_to(external_git, target_is_directory=True)

        assert _kanban_writable_roots(
            {
                "HERMES_KANBAN_DB": str(db),
                "HERMES_KANBAN_WORKSPACE": str(workspace),
            },
            workspace_cwd=str(workspace),
        ) == [str(board), str(workspace)]

    def test_arbitrary_gitdir_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".git").write_text("gitdir: /\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    @pytest.mark.parametrize("commondir", [None, "missing\n", "\n"])
    def test_missing_or_malformed_commondir_fails_closed(
        self, tmp_path, commondir
    ):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "repo" / ".git"
        git_dir = common / "worktrees" / "task"
        workspace = tmp_path / "workspace"
        git_dir.mkdir(parents=True)
        workspace.mkdir()
        (workspace / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "gitdir").write_text(f"{workspace / '.git'}\n")
        if commondir is not None:
            (git_dir / "commondir").write_text(commondir)

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    def test_non_worktree_gitfile_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "common"
        git_dir = tmp_path / "external-gitdir"
        workspace = tmp_path / "workspace"
        for root in (common / "objects", common / "refs", common / "logs"):
            root.mkdir(parents=True, exist_ok=True)
        git_dir.mkdir()
        workspace.mkdir()
        (workspace / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text(f"{common}\n")
        (git_dir / "gitdir").write_text(f"{workspace / '.git'}\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    def test_mismatched_worktree_backlink_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "repo" / ".git"
        git_dir = common / "worktrees" / "task"
        workspace = tmp_path / "workspace"
        other = tmp_path / "other" / ".git"
        for root in (common / "objects", common / "refs", common / "logs"):
            root.mkdir(parents=True, exist_ok=True)
        git_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
        other.parent.mkdir()
        other.write_text("gitdir: nowhere\n")
        (workspace / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{other}\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    @pytest.mark.parametrize("name", ["objects", "refs", "logs"])
    def test_common_metadata_symlink_escape_fails_closed(
        self, tmp_path, name
    ):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "repo" / ".git"
        git_dir = common / "worktrees" / "task"
        workspace = tmp_path / "workspace"
        git_dir.mkdir(parents=True)
        workspace.mkdir()
        for root_name in ("objects", "refs", "logs"):
            root = common / root_name
            if root_name == name:
                root.symlink_to("/", target_is_directory=True)
            else:
                root.mkdir()
        (workspace / ".git").write_text(f"gitdir: {git_dir}\n")
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{workspace / '.git'}\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    def test_symlinked_gitfile_marker_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "repo" / ".git"
        git_dir = common / "worktrees" / "task"
        workspace = tmp_path / "workspace"
        marker_target = tmp_path / "git-marker"
        for root in (common / "objects", common / "refs", common / "logs"):
            root.mkdir(parents=True, exist_ok=True)
        git_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
        marker_target.write_text(f"gitdir: {git_dir}\n")
        (workspace / ".git").symlink_to(marker_target)
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(f"{marker_target}\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []


    def test_invalid_utf8_git_metadata_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".git").write_bytes(b"gitdir: \xff\n")

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    @pytest.mark.parametrize("metadata", ["marker", "commondir", "backlink"])
    def test_nul_git_metadata_fails_closed(self, tmp_path, metadata):
        from agent.transports.codex_app_server import (
            _linked_worktree_metadata_roots,
        )

        common = tmp_path / "repo" / ".git"
        git_dir = common / "worktrees" / "task"
        workspace = tmp_path / "workspace"
        for root in (common / "objects", common / "refs", common / "logs"):
            root.mkdir(parents=True, exist_ok=True)
        git_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
        marker = workspace / ".git"
        commondir = git_dir / "commondir"
        backlink = git_dir / "gitdir"
        marker.write_text(f"gitdir: {git_dir}\n")
        commondir.write_text("../..\n")
        backlink.write_text(f"{marker}\n")

        bad_path, bad_value = {
            "marker": (marker, f"gitdir: {git_dir}\0\n"),
            "commondir": (commondir, "../..\0\n"),
            "backlink": (backlink, f"{marker}\0\n"),
        }[metadata]
        bad_path.write_text(bad_value)

        assert _linked_worktree_metadata_roots(str(workspace)) == []

    @pytest.mark.parametrize("path_source", ["db", "workspace", "cwd"])
    def test_nul_kanban_env_path_fails_closed(self, tmp_path, path_source):
        from agent.transports.codex_app_server import _kanban_writable_roots

        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        env = {
            "HERMES_KANBAN_DB": str(db),
            "HERMES_KANBAN_WORKSPACE": str(workspace),
        }
        cwd = str(workspace)
        if path_source == "db":
            env["HERMES_KANBAN_DB"] += "\0"
        elif path_source == "workspace":
            env["HERMES_KANBAN_WORKSPACE"] += "\0"
        else:
            cwd += "\0"

        assert _kanban_writable_roots(env, workspace_cwd=cwd) == []

    @pytest.mark.parametrize("symlink_target", ["workspace", "board"])
    def test_symlinked_kanban_roots_fail_closed(
        self, monkeypatch, tmp_path, symlink_target
    ):
        import subprocess

        from agent.transports import codex_app_server as cas

        class UnexpectedPopen:
            def __init__(self, *args, **kwargs):
                raise AssertionError("invalid Kanban roots must fail before spawning Codex")

        real_workspace = tmp_path / "real-workspace"
        real_workspace.mkdir()
        workspace = tmp_path / "workspace"
        workspace.symlink_to(real_workspace, target_is_directory=True)
        real_board = tmp_path / "real-board"
        real_board.mkdir()
        (real_board / "kanban.db").touch()
        board = tmp_path / "board"
        board.symlink_to(real_board, target_is_directory=True)

        monkeypatch.setattr(subprocess, "Popen", UnexpectedPopen)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv(
            "HERMES_KANBAN_WORKSPACE",
            str(workspace if symlink_target == "workspace" else real_workspace),
        )
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            str((board if symlink_target == "board" else real_board) / "kanban.db"),
        )

        with pytest.raises(ValueError, match="validated Kanban writable roots"):
            cas.CodexAppServerClient(
                codex_bin="codex",
                workspace_cwd=str(
                    workspace if symlink_target == "workspace" else real_workspace
                ),
            )

    def test_inherited_broad_roots_are_ignored(self, tmp_path):
        from agent.transports.codex_app_server import _kanban_writable_roots

        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        roots = _kanban_writable_roots(
            {
                "HERMES_KANBAN_DB": str(db),
                "HERMES_KANBAN_WORKSPACE": str(workspace),
                "HERMES_KANBAN_ROOT": "/",
                "HERMES_KANBAN_WORKSPACES_ROOT": str(tmp_path),
            },
            workspace_cwd=str(workspace),
        )

        assert roots == [str(board), str(workspace)]

    def test_mismatched_workspace_cwd_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import _kanban_writable_roots

        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sibling = tmp_path / "workspace-sibling"
        sibling.mkdir()

        assert _kanban_writable_roots(
            {
                "HERMES_KANBAN_DB": str(db),
                "HERMES_KANBAN_WORKSPACE": str(workspace),
            },
            workspace_cwd=str(sibling),
        ) == []

    def test_filesystem_root_workspace_fails_closed(self, tmp_path):
        from agent.transports.codex_app_server import _kanban_writable_roots

        board = tmp_path / "board"
        board.mkdir()
        db = board / "kanban.db"
        db.touch()
        filesystem_root = str(tmp_path.anchor)

        assert _kanban_writable_roots(
            {
                "HERMES_KANBAN_DB": str(db),
                "HERMES_KANBAN_WORKSPACE": filesystem_root,
            },
            workspace_cwd=filesystem_root,
        ) == []

    def test_writable_roots_are_toml_safe_unicode(self):
        import tomllib

        from agent.transports.codex_app_server import _toml_string_array

        roots = ['C:\\workspaces\\quoted"\\작업-😀']
        encoded = _toml_string_array(roots)

        assert "작업-😀" in encoded
        assert tomllib.loads(f"roots = {encoded}")["roots"] == roots


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

    def test_home_still_preserved_through_helper(self, monkeypatch):
        """Regression guard: routing through hermes_subprocess_env must not
        rewrite HOME (codex's shell tool spawns gh/git/aws that need it)."""
        monkeypatch.setenv("HOME", "/users/alice")
        env = self._capture_spawn_env(monkeypatch)
        assert env.get("HOME") == "/users/alice"
