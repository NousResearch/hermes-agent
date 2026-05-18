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
            "google-gemini-cli",
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

    def test_kanban_worker_adds_only_kanban_writable_root(self, monkeypatch):
        """Codex-runtime Kanban workers need to write board state outside
        their scratch/worktree workspace, but should not fall back to
        danger-full-access. Hermes passes a narrow app-server config override
        for the Kanban root only.
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

        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True

        cmd = captured["cmd"]
        assert cmd[:2] == ["codex", "app-server"]
        assert 'sandbox_mode="workspace-write"' in cmd
        assert (
            'sandbox_workspace_write.writable_roots=["/users/alice/.hermes/kanban/boards/smoke"]'
            in cmd
        )
        assert "sandbox_workspace_write.network_access=false" in cmd
        assert all("danger" not in part for part in cmd)


# ---------------------------------------------------------------------------
# Issue #27941 regression tests -- writable_roots must cover every Kanban
# path the dispatcher pinned, not just dirname(HERMES_KANBAN_DB).
#
# Before the fix, codex_app_server workers received only the DB-dir root, so
# artifact writes under HERMES_KANBAN_WORKSPACES_ROOT / HERMES_KANBAN_WORKSPACE
# (often on a different mount, e.g. /media/.../kanban-workspaces/) were
# silently blocked by the Codex sandbox with a misleading
# ``Errno 30 Read-only file system``.
# ---------------------------------------------------------------------------


class _FakePopen:
    """Minimal subprocess.Popen stand-in -- captures argv + env."""

    captured: dict = {}

    def __init__(self, cmd, *args, **kwargs):
        type(self).captured = {
            "cmd": list(cmd),
            "env": dict(kwargs.get("env", {})),
        }
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


def _spawn_and_capture(monkeypatch, env: dict) -> list[str]:
    """Spawn a CodexAppServerClient with ``env`` set and return its argv."""
    import subprocess
    from agent.transports import codex_app_server as cas

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    client = cas.CodexAppServerClient(codex_bin="codex")
    client._closed = True
    return list(_FakePopen.captured["cmd"])


def _writable_roots_arg(cmd: list[str]) -> str:
    """Return the single ``writable_roots=...`` -c override from the argv."""
    for i, part in enumerate(cmd):
        if part.startswith("sandbox_workspace_write.writable_roots="):
            return part
    raise AssertionError(f"writable_roots override missing from cmd: {cmd!r}")


class TestKanbanWritableRootsIssue27941:
    """Pin the post-fix writable-roots construction across the documented
    dispatcher env-var combinations."""

    DB = "/users/alice/.hermes/kanban/boards/smoke/kanban.db"
    DB_DIR = "/users/alice/.hermes/kanban/boards/smoke"

    def test_workspaces_root_on_separate_mount_is_included(self, monkeypatch):
        """The exact #27941 scenario: workspaces pinned to /media/...,
        DB under ~/.hermes/.  Both must appear in writable_roots."""
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACES_ROOT": "/media/data/kanban-workspaces",
        })
        roots = _writable_roots_arg(cmd)
        assert self.DB_DIR in roots, "DB dir must remain in writable_roots"
        assert "/media/data/kanban-workspaces" in roots, (
            "pinned workspaces root must be added so artifact writes succeed"
        )

    def test_per_task_workspace_outside_db_dir_is_included(self, monkeypatch):
        """The per-task workspace (HERMES_KANBAN_WORKSPACE) may also live
        outside dirname(HERMES_KANBAN_DB) -- e.g. when the dispatcher
        resolves it to a repo clone elsewhere on disk."""
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACE": "/media/data/Tools/staged-codex-scorecard",
        })
        roots = _writable_roots_arg(cmd)
        assert self.DB_DIR in roots
        assert "/media/data/Tools/staged-codex-scorecard" in roots

    def test_workspaces_root_and_per_task_workspace_both_included(self, monkeypatch):
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACES_ROOT": "/media/data/kanban-workspaces",
            "HERMES_KANBAN_WORKSPACE": "/media/data/kanban-workspaces/t_smoke",
        })
        roots = _writable_roots_arg(cmd)
        assert self.DB_DIR in roots
        assert "/media/data/kanban-workspaces" in roots
        assert "/media/data/kanban-workspaces/t_smoke" in roots

    def test_no_extra_roots_when_only_kanban_db_set(self, monkeypatch):
        """Backward-compat: the pre-#27941 happy path (only DB set) still
        yields a single-entry list with the DB dir.  Pins the existing
        test_kanban_worker_adds_only_kanban_writable_root contract."""
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_HOME": "/users/alice/.hermes/profiles/backend-worker",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
        })
        assert (
            f'sandbox_workspace_write.writable_roots=["{self.DB_DIR}"]' in cmd
        )

    def test_duplicate_paths_are_deduplicated(self, monkeypatch):
        """If a setup has workspaces_root == dirname(DB), the rendered
        TOML array must not list the same path twice (Codex CLI would
        accept it but the duplicate is noise + risks log diffs)."""
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACES_ROOT": self.DB_DIR,  # same path
        })
        roots = _writable_roots_arg(cmd)
        assert roots.count(self.DB_DIR) == 1, (
            f"duplicate path leaked into writable_roots: {roots!r}"
        )

    def test_empty_or_whitespace_env_vars_are_ignored(self, monkeypatch):
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACES_ROOT": "   ",
            "HERMES_KANBAN_WORKSPACE": "",
        })
        # Should fall back to the single-DB-dir contract.
        assert (
            f'sandbox_workspace_write.writable_roots=["{self.DB_DIR}"]' in cmd
        )

    def test_order_db_dir_before_pinned_roots(self, monkeypatch):
        """The DB dir is the most-needed root (board writes always go
        there); the pinned workspaces follow, then the per-task
        workspace.  Order matters for human inspection of the override."""
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_KANBAN_TASK": "t_smoke",
            "HERMES_KANBAN_DB": self.DB,
            "HERMES_KANBAN_WORKSPACES_ROOT": "/media/data/kanban-workspaces",
            "HERMES_KANBAN_WORKSPACE": "/media/data/kanban-workspaces/t_smoke",
        })
        roots = _writable_roots_arg(cmd)
        i_db = roots.index(self.DB_DIR)
        i_root = roots.index("/media/data/kanban-workspaces")
        i_ws = roots.index("/media/data/kanban-workspaces/t_smoke")
        assert i_db < i_root < i_ws

    def test_legacy_fallback_when_kanban_db_missing(self, monkeypatch):
        """If only HERMES_KANBAN_TASK is set (no DB env), the legacy
        ``HERMES_KANBAN_ROOT`` / ``HERMES_HOME/kanban`` fallback path
        keeps working unchanged."""
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
            "HERMES_HOME": "/users/alice/.hermes/profiles/backend-worker",
            "HERMES_KANBAN_TASK": "t_smoke",
        })
        roots = _writable_roots_arg(cmd)
        assert "/users/alice/.hermes/profiles/backend-worker/kanban" in roots

    def test_no_writable_roots_override_when_not_a_kanban_worker(self, monkeypatch):
        """Without HERMES_KANBAN_TASK we must NOT inject any sandbox
        override -- that would be a behaviour change for every non-Kanban
        codex_app_server caller."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
        cmd = _spawn_and_capture(monkeypatch, {
            "HOME": "/users/alice",
        })
        assert all(
            not p.startswith("sandbox_") for p in cmd
        ), f"non-Kanban codex_app_server invocations must not inject sandbox -c: {cmd!r}"


class TestBuildKanbanWritableRootsHelper:
    """Direct unit tests for the helper -- not via Popen, so they're cheap
    and pinpoint the dedup/normalisation rules without environment leaks."""

    def test_returns_db_dir_only(self):
        from agent.transports.codex_app_server import _build_kanban_writable_roots
        roots = _build_kanban_writable_roots({
            "HERMES_KANBAN_DB": "/x/y/kanban.db",
        })
        assert roots == ["/x/y"]

    def test_appends_pinned_roots_in_documented_order(self):
        from agent.transports.codex_app_server import _build_kanban_writable_roots
        roots = _build_kanban_writable_roots({
            "HERMES_KANBAN_DB": "/x/y/kanban.db",
            "HERMES_KANBAN_WORKSPACES_ROOT": "/m/workspaces",
            "HERMES_KANBAN_WORKSPACE": "/m/workspaces/t1",
        })
        assert roots == ["/x/y", "/m/workspaces", "/m/workspaces/t1"]

    def test_deduplicates(self):
        from agent.transports.codex_app_server import _build_kanban_writable_roots
        roots = _build_kanban_writable_roots({
            "HERMES_KANBAN_DB": "/x/y/kanban.db",
            "HERMES_KANBAN_WORKSPACES_ROOT": "/x/y",
        })
        assert roots == ["/x/y"]

    def test_strips_whitespace(self):
        from agent.transports.codex_app_server import _build_kanban_writable_roots
        roots = _build_kanban_writable_roots({
            "HERMES_KANBAN_DB": "/x/y/kanban.db",
            "HERMES_KANBAN_WORKSPACES_ROOT": "   /m/w   ",
        })
        assert roots == ["/x/y", "/m/w"]

    def test_skips_empty_and_blank_env_values(self):
        from agent.transports.codex_app_server import _build_kanban_writable_roots
        roots = _build_kanban_writable_roots({
            "HERMES_KANBAN_DB": "/x/y/kanban.db",
            "HERMES_KANBAN_WORKSPACES_ROOT": "",
            "HERMES_KANBAN_WORKSPACE": "   ",
        })
        assert roots == ["/x/y"]


class TestFormatTomlPathArray:
    """The TOML array renderer must produce valid TOML for paths with
    backslashes / quotes / control chars (Codex CLI parses ``-c`` as TOML)."""

    def test_simple_posix_path(self):
        from agent.transports.codex_app_server import _format_toml_path_array
        assert _format_toml_path_array(["/x/y"]) == '["/x/y"]'

    def test_multiple_paths_joined_with_comma_no_spaces(self):
        from agent.transports.codex_app_server import _format_toml_path_array
        assert _format_toml_path_array(["/a", "/b"]) == '["/a","/b"]'

    def test_backslash_is_escaped(self):
        from agent.transports.codex_app_server import _format_toml_path_array
        # Windows path -- must not produce TOML escape sequences.
        result = _format_toml_path_array(["C:\\Users\\alice\\.hermes"])
        assert result == '["C:\\\\Users\\\\alice\\\\.hermes"]'

    def test_embedded_quote_is_escaped(self):
        from agent.transports.codex_app_server import _format_toml_path_array
        result = _format_toml_path_array(['/x/"odd"/dir'])
        assert result == '["/x/\\"odd\\"/dir"]'

    def test_empty_list_renders_as_empty_array(self):
        from agent.transports.codex_app_server import _format_toml_path_array
        assert _format_toml_path_array([]) == "[]"
