import argparse
import ast
import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mcp_profile_router
from mcp_profile_router import (
    COST_CLASS_CALLS_HERMES_AGENT_MODEL,
    COST_CLASS_EXTERNAL_API_NO_MODEL,
    COST_CLASS_NO_MODEL,
    MAX_CONTEXT_FILE_BYTES,
    MAX_SESSION_SNIPPET_CHARS,
    MAX_TERMINAL_OUTPUT_CHARS,
    MAX_TERMINAL_TIMEOUT_SECONDS,
    MAX_VIKING_OVERVIEW_CHARS,
    PROFILE_ROUTER_CAPABILITY_GROUPS,
    PROFILE_ROUTER_TOOL_GROUP,
    FORBIDDEN_MODEL_LOOP_TOOL_NAMES,
    HERMES_CATALOG_BLOCKED_TOOL_NAMES,
    HERMES_CATALOG_MODEL_BACKED_TOOL_NAMES,
    HERMES_REGISTRY_TOOL_NAMES,
    ProfileRouterError,
    RouterToolMetadata,
    _build_terminal_sanitized_env,
    _prepare_terminal_subprocess_plan,
    _shape_terminal_subprocess_result,
    assert_default_tools_are_no_model,
    assert_no_model_loop_tools_absent,
    classify_terminal_command,
    create_workspace_metadata,
    cron_create_script_only,
    cron_list,
    cron_pause,
    cron_resume,
    cron_run,
    directory_create,
    file_delete,
    file_move,
    file_patch,
    file_read,
    file_search,
    file_write,
    git_branch,
    git_add,
    git_commit,
    git_push,
    git_checkout,
    git_restore,
    git_rebase,
    git_merge,
    github_pr_status,
    github_pr_create,
    github_pr_update,
    github_pr_ready,
    github_pr_merge,
    github_issue_view,
    github_issue_comment,
    git_diff,
    git_log,
    git_status,
    hermes_catalog_blocked_tool,
    get_router_tool_metadata,
    load_profile_router_policy,
    message_send,
    parse_profile_ref,
    patch_apply,
    process_start,
    process_kill,
    process_list,
    process_log,
    process_poll,
    profile_context_get,
    profile_get,
    profile_health,
    profile_memory_add,
    profile_memory_list,
    profile_memory_remove,
    profile_memory_replace,
    profile_skill_create,
    profile_skill_delete,
    profile_skill_patch,
    profile_skill_write_file,
    profiles_list,
    require_fresh_workspace_context,
    resolve_workspace_path,
    session_search,
    skill_view,
    skills_list,
    telegram_send,
    terminal_run,
    workspace_python_run,
    workspace_production_action_list,
    workspace_production_action_status,
    workspace_production_action_run,
    server_alias_list,
    server_status_check,
    server_command_run,
    server_shell_run,
    workspace_web_fetch,
    viking_read,
    viking_search,
    workspace_close,
    workspace_context_status,
    workspace_diff,
    workspace_file_list,
    workspace_file_read,
    workspace_file_stat,
    workspace_file_search,
    workspace_status_probe,
    workspace_scratch_smoke,
    workspace_get,
    workspace_instructions_get,
    workspace_open,
)
from mcp_profile_router_auth import (
    DEFAULT_PROFILE_ROUTER_SCOPES,
    PROFILE_ROUTER_CRON_SCOPE,
    PROFILE_ROUTER_MESSAGING_SCOPE,
    PROFILE_ROUTER_TERMINAL_SCOPE,
    PROFILE_ROUTER_WRITE_SCOPE,
    ProfileRouterAuditLogger,
    ProfileRouterBearerTokenVerifier,
    ProfileRouterTokenStore,
    VIKING_PROFILE_ROUTER_SCOPE,
    extract_result_audit_fields,
)


class _FakeTool:
    def __init__(self, fn):
        self.name = fn.__name__
        self.description = fn.__doc__ or ""
        self.fn = fn


class _FakeToolManager:
    def __init__(self):
        self._tools = {}

    def add_tool(self, fn):
        self._tools[fn.__name__] = _FakeTool(fn)


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._tool_manager = _FakeToolManager()

    def tool(self):
        def decorator(fn):
            self._tool_manager.add_tool(fn)
            return fn

        return decorator


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    for name in ("main-bot", "maker"):
        profile_dir = profiles_root / name
        (profile_dir / "skills").mkdir(parents=True)
    return tmp_path


def _write_router_config(hermes_home, *, profiles=None, host_roots=None, context=None, extra_profile_router=None):
    host_roots = host_roots or [str(hermes_home)]
    profiles = profiles or {
        "local:main-bot": {
            "enabled": True,
            "display_name": "Main Bot",
            "allowed_roots": [str(hermes_home)],
        }
    }
    profile_router = {
        "hosts": {
            "local": {
                "enabled": True,
                "allowed_roots": host_roots,
            }
        },
        "profiles": profiles,
    }
    if context is not None:
        profile_router["context"] = context
    if extra_profile_router:
        profile_router.update(extra_profile_router)
    config = {"profile_router": profile_router}
    (hermes_home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return config


def _write_skill(
    hermes_home,
    profile,
    skill_path,
    content,
    *,
    linked_files=None,
):
    skill_dir = hermes_home / "profiles" / profile / "skills" / skill_path
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for relative_path, file_content in (linked_files or {}).items():
        target = skill_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            target.write_bytes(file_content)
        else:
            target.write_text(file_content, encoding="utf-8")
    return skill_dir


def _write_session_db(hermes_home, profile, sessions):
    db_path = hermes_home / "profiles" / profile / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                model TEXT,
                started_at REAL NOT NULL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        for session in sessions:
            conn.execute(
                "INSERT INTO sessions(id, source, model, started_at, title) VALUES (?, ?, ?, ?, ?)",
                (
                    session["id"],
                    session.get("source", "cli"),
                    session.get("model", "test/model"),
                    session.get("started_at", 1.0),
                    session.get("title"),
                ),
            )
            for message in session.get("messages", []):
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?)",
                    (
                        session["id"],
                        message["role"],
                        message.get("content", ""),
                        message.get("timestamp", session.get("started_at", 1.0)),
                        message.get("active", 1),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _git(repo, *args):
    try:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"git executable unavailable: {exc}")


def test_parse_profile_ref_requires_fully_qualified_ref():
    assert parse_profile_ref("local:main-bot").value == "local:main-bot"
    assert parse_profile_ref("LOCAL:Main-Bot").value == "local:main-bot"
    assert parse_profile_ref("mac:maker").value == "mac:maker"

    for bad_ref in ("main-bot", "local:", ":main-bot", "remote:main-bot", "local:hermes"):
        with pytest.raises(ProfileRouterError):
            parse_profile_ref(bad_ref)


def _registry_tool_names_from_source() -> set[str]:
    tools_root = Path(mcp_profile_router.__file__).resolve().parent / "tools"
    names: set[str] = set()
    for tool_file in tools_root.glob("*.py"):
        tree = ast.parse(tool_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register"
            ):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    names.add(keyword.value.value)
    return names


def test_phase9_static_hermes_registry_inventory_matches_source():
    assert set(HERMES_REGISTRY_TOOL_NAMES) == _registry_tool_names_from_source()


def test_router_tool_metadata_is_explicitly_no_model_by_default():
    metadata = get_router_tool_metadata()
    public_tools = {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "skills_list",
        "skill_view",
        "session_search",
        "viking_search",
        "viking_read",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_diff",
    }
    disabled_power_tools = {
        "file_read",
        "file_search",
        "file_patch",
        "patch_apply",
        "file_write",
        "workspace_status_probe",
        "workspace_scratch_smoke",
        "file_move",
        "file_delete",
        "directory_create",
        "terminal_run",
        "process_start",
        "process_list",
        "process_poll",
        "process_log",
        "process_kill",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "git_add",
        "git_commit",
        "git_push",
        "git_checkout",
        "git_restore",
        "git_rebase",
        "git_merge",
        "github_pr_status",
        "github_pr_create",
        "github_pr_update",
        "github_pr_ready",
        "github_pr_merge",
        "github_issue_view",
        "github_issue_comment",
        "cron_list",
        "cron_pause",
        "cron_resume",
        "cron_run",
        "cron_create_script_only",
        "message_send",
        "telegram_send",
        "workspace_production_action_list",
        "workspace_production_action_status",
        "workspace_production_action_run",
        "workspace_web_fetch",
        "workspace_python_run",
    }
    profile_server_tools = {
        "server_alias_list",
        "server_status_check",
        "server_service_logs",
        "server_docker_ps",
        "server_docker_logs",
        "server_port_check",
        "server_command_run",
        "server_shell_run",
    }
    profile_action_tools = {
        "profile_skill_create",
        "profile_skill_patch",
        "profile_skill_edit",
        "profile_skill_write_file",
        "profile_skill_remove_file",
        "profile_skill_delete",
        "profile_memory_add",
        "profile_memory_replace",
        "profile_memory_remove",
        "profile_memory_list",
    }
    blocked_catalog_tools = set(HERMES_CATALOG_BLOCKED_TOOL_NAMES)
    assert set(metadata) == public_tools | disabled_power_tools | profile_server_tools | profile_action_tools | blocked_catalog_tools
    assert {"delegate_task", "image_generate", "vision_analyze"}.issubset(blocked_catalog_tools)
    assert set(HERMES_REGISTRY_TOOL_NAMES).issubset(set(metadata))
    workspace_required_tools = {
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_diff",
        "file_read",
        "file_search",
        "file_patch",
        "patch_apply",
        "file_write",
        "workspace_status_probe",
        "workspace_scratch_smoke",
        "file_move",
        "file_delete",
        "directory_create",
        "terminal_run",
        "process_start",
        "process_list",
        "process_poll",
        "process_log",
        "process_kill",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "git_add",
        "git_commit",
        "git_push",
        "git_checkout",
        "git_restore",
        "git_rebase",
        "git_merge",
        "github_pr_status",
        "github_pr_create",
        "github_pr_update",
        "github_pr_ready",
        "github_pr_merge",
        "github_issue_view",
        "github_issue_comment",
        "cron_list",
        "cron_pause",
        "cron_resume",
        "cron_run",
        "cron_create_script_only",
        "message_send",
        "telegram_send",
        "workspace_production_action_list",
        "workspace_production_action_status",
        "workspace_production_action_run",
        "workspace_web_fetch",
        "workspace_python_run",
    }
    for name, tool in metadata.items():
        assert tool["allowed_by_default"] is tool["enabled_by_default"]
        assert tool["side_effects"] is tool["mutates_state"]
        assert tool["requires_workspace"] is (name in workspace_required_tools)
        assert tool["requires_approval"] is tool["mutates_state"]
        assert tool["capability_group"] in {
            "profile",
            "workspace",
            *PROFILE_ROUTER_CAPABILITY_GROUPS,
        }
        assert tool["llm_calls"] == 0

    for name in public_tools:
        tool = metadata[name]
        expected_cost_class = (
            COST_CLASS_EXTERNAL_API_NO_MODEL
            if name in {"viking_search", "viking_read"}
            else COST_CLASS_NO_MODEL
        )
        assert tool["cost_class"] == expected_cost_class
        assert tool["enabled_by_default"] is True
        assert tool["mutates_state"] is False
        assert tool["requires_context"] is (
            name
            in {
                "workspace_file_list",
                "workspace_file_read",
                "workspace_file_stat",
                "workspace_file_search",
                "workspace_diff",
            }
        )
        assert tool["execution_status"] == "executable_no_model"

    for name in disabled_power_tools:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["enabled_by_default"] is False
        assert tool["requires_context"] is True
        assert tool["execution_status"] == "executable_no_model"

    for name in profile_server_tools:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["enabled_by_default"] is False
        assert tool["requires_profile_ref"] is True
        assert tool["requires_context"] is False
        assert tool["execution_status"] == "executable_no_model"

    for name in blocked_catalog_tools:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["enabled_by_default"] is False
        assert tool["mutates_state"] is False
        assert tool["requires_context"] is False
        assert tool["requires_workspace"] is False
        assert tool["execution_status"] == "blocked_no_model"
        expected_reason = (
            "model_backed_tool_blocked"
            if name in HERMES_CATALOG_MODEL_BACKED_TOOL_NAMES
            else "requires_no_model_implementation"
        )
        assert tool["blocked_reason"] == expected_reason

    for name in set(metadata) & FORBIDDEN_MODEL_LOOP_TOOL_NAMES:
        assert metadata[name]["execution_status"] == "blocked_no_model"
        assert metadata[name]["blocked_reason"] == "model_backed_tool_blocked"
    for name in {"skills_list", "skill_view"}:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["enabled_by_default"] is True
        assert tool["mutates_state"] is False
        assert tool["requires_profile_ref"] is True
        assert tool["requires_context_policy"] == "context.skills.read"
    session_tool = metadata["session_search"]
    assert session_tool["cost_class"] == COST_CLASS_NO_MODEL
    assert session_tool["enabled_by_default"] is True
    assert session_tool["mutates_state"] is False
    assert session_tool["requires_profile_ref"] is True
    assert session_tool["requires_context_policy"] == "context.sessions.search"
    for name in {"viking_search", "viking_read"}:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_EXTERNAL_API_NO_MODEL
        assert tool["enabled_by_default"] is True
        assert tool["mutates_state"] is False
        assert tool["requires_context_policy"] == "profile_router.context.viking.read"
    assert metadata["workspace_diff"]["mutates_state"] is False
    for name in {"file_patch", "patch_apply", "file_write", "workspace_scratch_smoke", "file_move", "file_delete", "directory_create", "terminal_run", "workspace_python_run", "process_kill", "git_add", "git_commit", "git_push", "git_checkout", "git_restore", "git_rebase", "git_merge", "github_pr_create", "github_pr_update", "github_pr_ready", "github_pr_merge", "github_issue_comment", "message_send", "telegram_send", "workspace_production_action_run", "server_command_run", "server_shell_run", "profile_skill_create", "profile_skill_patch", "profile_skill_edit", "profile_skill_write_file", "profile_skill_remove_file", "profile_skill_delete", "profile_memory_add", "profile_memory_replace", "profile_memory_remove"}:
        assert metadata[name]["mutates_state"] is True
    for name in {"workspace_status_probe", "process_list", "process_poll", "process_log", "git_status", "git_diff", "git_log", "git_branch", "github_pr_status", "github_issue_view", "workspace_production_action_list", "workspace_production_action_status", "server_alias_list", "server_status_check", "server_service_logs", "server_docker_ps", "server_docker_logs", "server_port_check", "workspace_web_fetch", "profile_memory_list"}:
        assert metadata[name]["mutates_state"] is False
    for name in {"git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit", "git_push", "git_checkout", "git_restore", "git_rebase", "git_merge", "github_pr_status", "github_pr_create", "github_pr_update", "github_pr_ready", "github_pr_merge", "github_issue_view", "github_issue_comment"}:
        assert metadata[name]["capability_group"] == "git"
    for name in {"message_send", "telegram_send"}:
        assert metadata[name]["capability_group"] == "messaging"

    assert_default_tools_are_no_model()


def test_phase9_web_browser_api_model_surfaces_are_catalog_visible_but_blocked():
    metadata = get_router_tool_metadata()
    phase9_blocked_tools = {
        "browser_back",
        "browser_cdp",
        "browser_click",
        "browser_console",
        "browser_get_images",
        "browser_navigate",
        "browser_press",
        "browser_snapshot",
        "browser_type",
        "browser_vision",
        "web_extract",
        "web_search",
        "x_search",
    }
    assert phase9_blocked_tools.issubset(set(metadata))
    for name in phase9_blocked_tools:
        assert metadata[name]["execution_status"] == "blocked_no_model"
        assert metadata[name]["llm_calls"] == 0
        assert metadata[name]["enabled_by_default"] is False
    assert metadata["web_search"]["capability_group"] == "web"
    assert metadata["browser_navigate"]["capability_group"] == "browser"
    assert metadata["x_search"]["blocked_reason"] == "model_backed_tool_blocked"
    assert metadata["x_search"]["route_hint"] == "use_client_native"
    assert metadata["browser_vision"]["blocked_reason"] == "model_backed_tool_blocked"
    assert metadata["web_search"]["route_hint"] == "requires_deterministic_no_model_wrapper"


def test_hermes_catalog_blocked_tool_returns_sanitized_llm_zero_error():
    payload = json.loads(hermes_catalog_blocked_tool("delegate_task"))
    assert payload["ok"] is False
    assert payload["llm_calls"] == 0
    assert payload["cost_class"] == COST_CLASS_NO_MODEL
    assert payload["error"]["code"] == "model_backed_tool_blocked"
    assert payload["catalog_tool"] == {
        "name": "delegate_task",
        "execution_status": "blocked_no_model",
        "native_side_effects": True,
        "capability_group": "api",
        "route_hint": "use_client_native",
        "root_exposed": False,
    }
    assert payload["route_hint"] == "use_client_native"
    assert "run_conversation" not in json.dumps(payload)


def test_no_model_guard_fails_closed_for_model_spending_default_tool():
    with pytest.raises(ProfileRouterError, match="Default profile-router tools must be no-model"):
        assert_default_tools_are_no_model(
            {
                "unsafe": RouterToolMetadata(
                    name="unsafe",
                    description="would call an agent loop",
                    cost_class=COST_CLASS_CALLS_HERMES_AGENT_MODEL,
                    llm_calls=1,
                )
            }
        )


def test_no_model_guard_fails_closed_for_unblocked_forbidden_model_loop_tool_name():
    with pytest.raises(ProfileRouterError, match="Model-loop tools must be blocked no-model"):
        assert_no_model_loop_tools_absent(
            {
                "run_conversation": RouterToolMetadata(
                    name="run_conversation",
                    description="would route to Hermes agent loop",
                    cost_class=COST_CLASS_NO_MODEL,
                    llm_calls=0,
                    enabled_by_default=False,
                )
            }
        )


def test_terminal_command_classifier_blocks_model_destructive_git_and_deploy_commands():
    blocked_cases = {
        "codex exec 'fix this'": "model_command",
        "hermes --profile maker chat -q hello": "model_command",
        "hermes --profile maker": "model_command",
        "python -c 'from run_agent import run_conversation'": "model_command",
        "python -c 'delegate_task({})'": "model_command",
        "rm -rf build": "destructive_command",
        "git reset --hard HEAD": "destructive_command",
        "git push origin main": "protected_git_command",
        "kubectl rollout restart deployment/app": "deploy_command",
    }
    for command, reason_code in blocked_cases.items():
        classification = classify_terminal_command(command)
        assert classification["cost_class"] == COST_CLASS_NO_MODEL
        assert classification["llm_calls"] == 0
        assert classification["executes"] is False
        assert classification["uses_shell"] is False
        assert classification["blocked"] is True
        assert classification["decision"] == "blocked"
        assert reason_code in {reason["code"] for reason in classification["reasons"]}
        assert command not in json.dumps(classification)

    low = classify_terminal_command("git status --short")
    assert low["blocked"] is False
    assert low["decision"] == "disabled_pending_execution_policy"
    assert low["risk_level"] == "low_unexecuted"
    assert low["reasons"] == []


def test_terminal_subprocess_plan_uses_sanitized_no_inheritance_env(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    monkeypatch.setenv("PATH", "/tmp/should-not-inherit")

    env = _build_terminal_sanitized_env()

    assert "OPENAI_API_KEY" not in env
    assert "HERMES_HOME" not in env
    assert "should-not-leak" not in json.dumps(env)
    assert "should-not-inherit" not in json.dumps(env)
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert all(isinstance(value, str) and "\x00" not in value for value in env.values())
    assert all(
        marker not in key.upper()
        for key in env
        for marker in ("API_KEY", "AUTH", "CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN")
    )

    plan = _prepare_terminal_subprocess_plan(
        "git status --short",
        resolved_cwd=tmp_path,
        public_cwd=".",
        timeout_seconds=7,
        max_output_chars=1234,
    )
    assert plan.argv == ("git", "status", "--short")
    assert plan.cwd == tmp_path
    assert plan.public_cwd == "."
    assert plan.env == env
    assert plan.timeout_seconds == 7
    assert plan.max_output_chars == 1234
    assert plan.uses_shell is False
    assert plan.executes is False

    with pytest.raises(ProfileRouterError) as shell_control:
        _prepare_terminal_subprocess_plan(
            "pwd && git status",
            resolved_cwd=tmp_path,
            public_cwd=".",
            timeout_seconds=7,
            max_output_chars=1234,
        )
    assert shell_control.value.code == "terminal_shell_control_not_allowed"


def test_terminal_result_shape_bounds_streams_status_and_audit(tmp_path):
    plan = _prepare_terminal_subprocess_plan(
        "git status --short",
        resolved_cwd=tmp_path,
        public_cwd="subdir",
        timeout_seconds=5,
        max_output_chars=8,
    )

    failed = _shape_terminal_subprocess_result(
        plan,
        returncode=2,
        stdout="abcdef",
        stderr="123456",
    )
    assert failed["status"] == "failed"
    assert failed["returncode"] == 2
    assert failed["timed_out"] is False
    assert failed["stdout"] == {
        "text": "abcdef",
        "truncated": False,
        "original_chars": 6,
        "returned_chars": 6,
    }
    assert failed["stderr"] == {
        "text": "12",
        "truncated": True,
        "original_chars": 6,
        "returned_chars": 2,
    }
    assert failed["output"] == {
        "max_output_chars": 8,
        "returned_chars": 8,
        "truncated": True,
        "stdout_truncated": False,
        "stderr_truncated": True,
    }
    assert failed["working_directory"] == "subdir"
    assert failed["audit"] == {
        "tool": "terminal_run",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "executes": False,
        "execution_attempted": False,
        "subprocess_run_allowed": False,
        "subprocess_run_called": False,
        "argv_redacted": True,
        "env_values_exposed": False,
        "public_mcp_exposure": "disabled_pending_http_auth_config_review",
    }
    dumped = json.dumps(failed)
    assert str(tmp_path) not in dumped
    assert "git status --short" not in dumped
    assert "/usr/bin" not in dumped

    success = _shape_terminal_subprocess_result(
        plan,
        returncode=0,
        stdout=b"ok",
        stderr=b"",
    )
    assert success["status"] == "success"
    assert success["stdout"]["text"] == "ok"

    timed_out = _shape_terminal_subprocess_result(
        plan,
        returncode=None,
        stdout="partial",
        stderr="",
        timed_out=True,
    )
    assert timed_out["status"] == "timeout"
    assert timed_out["returncode"] is None


def test_profile_router_policy_is_deny_by_default_for_execution_groups(hermes_home):
    policy = load_profile_router_policy(config={})
    assert policy.hosts == {}
    assert policy.profiles == {}

    config = _write_router_config(hermes_home)
    policy = load_profile_router_policy(config=config)
    route_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))

    assert route_policy.allowed_tool_groups == (PROFILE_ROUTER_TOOL_GROUP,)
    assert route_policy.allowed_roots == (str(hermes_home),)
    assert tuple(route_policy.capability_groups) == PROFILE_ROUTER_CAPABILITY_GROUPS
    assert all(value is False for value in route_policy.capability_groups.values())
    assert route_policy.allow_filesystem_read is False
    assert route_policy.allow_filesystem_write is False
    assert route_policy.allow_terminal is False
    assert route_policy.allow_messaging is False
    assert route_policy.allow_cron is False
    assert route_policy.allow_git_push is False
    assert route_policy.allow_deploy is False
    assert route_policy.allow_model_tools is False
    assert route_policy.allow_context_skills_read is False
    assert route_policy.allow_context_sessions_search is False
    assert route_policy.allowed_cost_classes == (COST_CLASS_NO_MODEL,)
    assert route_policy.terminal_execution_policy.enabled is False
    assert route_policy.terminal_execution_policy.allowed_commands == ()
    assert route_policy.terminal_execution_policy.allowed_command_prefixes == ()
    assert route_policy.terminal_execution_policy.require_no_shell is True


def test_profile_router_policy_parses_explicit_capability_groups(tmp_path):
    allowed_root = tmp_path / "allowed"
    config = {
        "profile_router": {
            "hosts": {"local": {"enabled": True, "allowed_roots": [str(allowed_root)]}},
            "profiles": {
                "local:main-bot": {
                    "enabled": True,
                    "allowed_roots": [str(allowed_root)],
                    "filesystem": {"read": True, "write": True},
                    "terminal": {"enabled": True},
                    "git": {"enabled": True},
                    "cron": {"enabled": True},
                    "messaging": {"enabled": True},
                    "skills": {"enabled": True},
                    "memory": {"enabled": True},
                    "session": {"enabled": True},
                    "web": {"enabled": True},
                    "browser": {"enabled": True},
                    "api": {"enabled": True},
                }
            },
        }
    }

    policy = load_profile_router_policy(config=config)
    route_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))

    assert tuple(route_policy.capability_groups) == PROFILE_ROUTER_CAPABILITY_GROUPS
    assert all(route_policy.capability_enabled(group) is True for group in PROFILE_ROUTER_CAPABILITY_GROUPS)
    assert route_policy.allowed_tool_groups == (PROFILE_ROUTER_TOOL_GROUP,)
    assert route_policy.allow_model_tools is False
    assert route_policy.allowed_cost_classes == (COST_CLASS_NO_MODEL,)

    with pytest.raises(ProfileRouterError, match="Unknown profile-router capability group"):
        route_policy.capability_enabled("model")


def test_profile_router_profile_defaults_and_auto_profiles_cover_future_profiles(hermes_home, tmp_path):
    worktree_parent = tmp_path / "worktrees"
    main_root = worktree_parent / "main-bot"
    future_root = worktree_parent / "future-bot"
    main_root.mkdir(parents=True)
    future_root.mkdir(parents=True)
    future_profile_dir = hermes_home / "profiles" / "future-bot"
    future_profile_dir.mkdir(parents=True)

    config = {
        "profile_router": {
            "hosts": {
                "local": {
                    "enabled": True,
                    "allowed_roots": [str(worktree_parent)],
                }
            },
            "profile_defaults": {
                "enabled": True,
                "allowed_tool_groups": [PROFILE_ROUTER_TOOL_GROUP],
                "context": {
                    "skills": {"read": True},
                    "sessions": {"search": True},
                },
                "filesystem": {"read": True, "write": True},
                "terminal": {
                    "enabled": True,
                    "execution": {
                        "enabled": True,
                        "allowed_commands": ["pwd", "git status --short"],
                        "require_no_shell": True,
                    },
                },
                "git": {"enabled": True, "allow_push": False},
                "cron": {"enabled": True, "allowed_scripts": []},
                "messaging": {
                    "enabled": True,
                    "allowed_recipients": ["telegram:dry-run-test"],
                },
                "skills": {"enabled": True, "write": True, "delete": True},
                "memory": {"enabled": True, "write": True},
                "model_tools": {
                    "allow_model_tools": False,
                    "allowed_cost_classes": [COST_CLASS_NO_MODEL],
                },
            },
            "auto_profiles": {
                "enabled": True,
                "root_patterns": [str(worktree_parent / "{profile}")],
                "metadata_only_without_root": True,
            },
            "profiles": {
                "local:main-bot": {
                    "enabled": True,
                    "allowed_roots": [str(main_root)],
                }
            },
        }
    }

    policy = load_profile_router_policy(config=config)

    explicit_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))
    assert explicit_policy.allowed_roots == (str(main_root),)
    assert explicit_policy.allow_filesystem_write is True
    assert explicit_policy.allow_terminal is True
    assert explicit_policy.terminal_execution_policy.enabled is True
    assert explicit_policy.capability_enabled("git") is True
    assert explicit_policy.cron_policy.enabled is True
    assert explicit_policy.allow_messaging is True
    assert explicit_policy.allow_skills_write is True
    assert explicit_policy.allow_memory_write is True

    future_policy = policy.get_profile_policy(parse_profile_ref("local:future-bot"))
    assert future_policy.allowed_roots == (str(future_root),)
    assert future_policy.allow_filesystem_write is True
    assert future_policy.allow_terminal is True
    assert future_policy.capability_enabled("git") is True

    metadata_only_policy = policy.get_profile_policy(parse_profile_ref("local:maker"))
    assert metadata_only_policy.allowed_roots == ()
    assert metadata_only_policy.allow_filesystem_read is False
    assert metadata_only_policy.allow_filesystem_write is False
    assert metadata_only_policy.allow_terminal is False
    assert metadata_only_policy.capability_enabled("git") is False
    assert metadata_only_policy.allow_skills_write is True
    assert metadata_only_policy.allow_memory_write is True



def test_profile_router_project_discovery_adds_safe_child_repos_without_home_roots(hermes_home, tmp_path):
    container = tmp_path / "project-containers"
    repo = container / "alpha-app"
    repo_git = repo / ".git"
    not_repo = container / "notes"
    repo_git.mkdir(parents=True)
    not_repo.mkdir(parents=True)

    config = {
        "profile_router": {
            "hosts": {"local": {"enabled": True, "allowed_roots": [str(container)]}},
            "project_discovery": {
                "enabled": True,
                "containers": [{"label": "worktrees", "path": str(container)}],
            },
            "profiles": {
                "local:main-bot": {
                    "enabled": True,
                    "allowed_roots": [],
                    "filesystem": {"read": True},
                }
            },
        }
    }

    policy = load_profile_router_policy(config=config)
    route_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))

    assert route_policy.allowed_roots == (str(repo.resolve()),)
    context = mcp_profile_router._public_policy_context(route_policy, policy)
    assert context["allowed_roots_count"] == 1
    assert context["project_discovery"] == {
        "enabled": True,
        "mode": "owner",
        "container_count": 1,
        "attach_to_profiles": True,
        "root_exposed": False,
    }
    root = context["workspace_roots"][0]
    assert root["root_label"] == "alpha-app"
    assert root["root_index"] == 0
    assert root["container_label"] is None
    assert root["git_repo"] is True
    assert root["root_exposed"] is False
    assert "path" not in root
    assert str(container) not in json.dumps(root)

    with pytest.raises(ProfileRouterError, match="project container"):
        load_profile_router_policy(
            config={
                "profile_router": {
                    "hosts": {"local": {"enabled": True, "allowed_roots": ["/"]}},
                    "project_discovery": {
                        "enabled": True,
                        "containers": [{"label": "root", "path": "/"}],
                    },
                }
            }
        )

def test_profile_router_auto_profile_root_patterns_must_be_profile_scoped(tmp_path):
    with pytest.raises(ProfileRouterError, match="must include"):
        load_profile_router_policy(
            config={
                "profile_router": {
                    "hosts": {"local": {"enabled": True, "allowed_roots": [str(tmp_path)]}},
                    "auto_profiles": {"enabled": True, "root_patterns": [str(tmp_path)]},
                }
            }
        )


def test_profile_router_policy_rejects_invalid_hosts_and_outside_roots(tmp_path):
    with pytest.raises(ProfileRouterError, match="Unsupported profile host"):
        load_profile_router_policy(
            config={"profile_router": {"hosts": {"remote": {"enabled": True}}}}
        )

    host_root = tmp_path / "allowed"
    outside_root = tmp_path / "elsewhere"
    with pytest.raises(ProfileRouterError, match="outside host local allowed_roots"):
        load_profile_router_policy(
            config={
                "profile_router": {
                    "hosts": {
                        "local": {
                            "enabled": True,
                            "allowed_roots": [str(host_root)],
                        }
                    },
                    "profiles": {
                        "local:main-bot": {
                            "enabled": True,
                            "allowed_roots": [str(outside_root)],
                        }
                    },
                }
            }
        )


def test_terminal_execution_policy_requires_no_shell_explicit_allowlist(tmp_path):
    host_root = tmp_path / "allowed"
    base_config = {
        "profile_router": {
            "hosts": {"local": {"enabled": True, "allowed_roots": [str(host_root)]}},
            "profiles": {
                "local:main-bot": {
                    "enabled": True,
                    "allowed_roots": [str(host_root)],
                    "terminal": {"enabled": True},
                }
            },
        }
    }

    with pytest.raises(ProfileRouterError, match="requires allowed_commands"):
        config = json.loads(json.dumps(base_config))
        config["profile_router"]["profiles"]["local:main-bot"]["terminal"][
            "execution"
        ] = {"enabled": True}
        load_profile_router_policy(config=config)

    with pytest.raises(ProfileRouterError, match="cannot be false"):
        config = json.loads(json.dumps(base_config))
        config["profile_router"]["profiles"]["local:main-bot"]["terminal"][
            "execution"
        ] = {"require_no_shell": False}
        load_profile_router_policy(config=config)

    config = json.loads(json.dumps(base_config))
    config["profile_router"]["profiles"]["local:main-bot"]["terminal"][
        "execution"
    ] = {
        "enabled": True,
        "allowed_commands": ["pwd"],
        "allowed_command_prefixes": ["git status"],
    }
    policy = load_profile_router_policy(config=config)
    route_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))
    assert route_policy.terminal_execution_policy.enabled is True
    assert route_policy.terminal_execution_policy.allowed_commands == ("pwd",)
    assert route_policy.terminal_execution_policy.allowed_command_prefixes == (
        "git status",
    )
    assert route_policy.terminal_execution_policy.require_no_shell is True


def test_workspace_metadata_requires_explicit_filesystem_read_policy(hermes_home):
    _write_router_config(hermes_home)

    with pytest.raises(ProfileRouterError, match="Filesystem read is disabled"):
        create_workspace_metadata(
            "local:main-bot",
            str(hermes_home),
            workspace_id="ws_denied",
        )


def test_workspace_metadata_resolves_paths_and_blocks_secrets_and_escapes(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    safe_file = workspace_root / "notes.txt"
    safe_file.write_text("safe\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (workspace_root / ".ssh").mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )

    workspace = create_workspace_metadata(
        "local:main-bot",
        str(workspace_root),
        workspace_id="ws_test",
    )
    assert workspace.to_public_dict() == {
        "workspace_id": "ws_test",
        "profile_ref": "local:main-bot",
        "host": "local",
        "profile": "main-bot",
        "root": str(workspace_root.resolve()),
        "mode": "checkout",
        "read_only": True,
        "cost_class": COST_CLASS_NO_MODEL,
        "llm_calls": 0,
    }
    assert resolve_workspace_path(workspace, "notes.txt") == str(safe_file.resolve())
    assert resolve_workspace_path(workspace, "new.txt", require_exists=False) == str(
        workspace_root.resolve() / "new.txt"
    )

    with pytest.raises(ProfileRouterError, match="workspace-relative"):
        resolve_workspace_path(workspace, str(safe_file))
    with pytest.raises(ProfileRouterError, match="escapes workspace root"):
        resolve_workspace_path(workspace, "../outside/file.txt", require_exists=False)
    for secret_path in (".env", ".env.local", ".ssh/id_rsa", "auth.json", "mcp_tokens"):
        with pytest.raises(ProfileRouterError, match="secret denylist"):
            resolve_workspace_path(workspace, secret_path, require_exists=False)
    with pytest.raises(ProfileRouterError, match="outside allowed roots"):
        create_workspace_metadata("local:main-bot", str(outside_root))
    with pytest.raises(ProfileRouterError, match="secret denylist"):
        create_workspace_metadata("local:main-bot", str(workspace_root / ".ssh"))


def test_resolve_workspace_path_rejects_symlink_traversal(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_file = outside_root / "leak.txt"
    outside_file.write_text("leak\n", encoding="utf-8")
    link = workspace_root / "link"
    try:
        link.symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    workspace = create_workspace_metadata("local:main-bot", str(workspace_root))

    with pytest.raises(ProfileRouterError, match="path escapes workspace root"):
        resolve_workspace_path(workspace, "link")


def test_workspace_open_file_read_and_search_are_policy_gated_and_bounded(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    notes = workspace_root / "notes.md"
    notes.write_text("alpha\nbeta\nalpha again\n", encoding="utf-8")
    (workspace_root / "config.md").write_text("api_key=abc123\nvisible=yes\n", encoding="utf-8")
    (workspace_root / "tokens.txt").write_text("token=abc123\n", encoding="utf-8")
    git_dir = workspace_root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[remote]\nurl=https://token@example.invalid/repo.git\n", encoding="utf-8")
    hermes_metadata_dir = workspace_root / ".hermes" / "plans"
    hermes_metadata_dir.mkdir(parents=True)
    (hermes_metadata_dir / "implementation-plan.md").write_text(
        "# Implementation plan\nRead-only plan context.\n",
        encoding="utf-8",
    )
    (hermes_metadata_dir / "state.json").write_text("local planning state\n", encoding="utf-8")
    (workspace_root / ".env.local").write_text("SECRET=1\n", encoding="utf-8")
    (workspace_root / "binary.bin").write_bytes(b"alpha\x00secret")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    assert opened["ok"] is True
    assert opened["llm_calls"] == 0
    workspace = opened["workspace"]
    assert workspace["workspace_id"].startswith("ws_")
    assert workspace["profile_ref"] == "local:main-bot"
    assert workspace["read_only"] is True
    assert "root" not in workspace
    assert opened["context"] == {
        "required_before_powerful_tools": True,
        "state": "not_loaded",
        "next_tool": "workspace_instructions_get",
    }

    read_without_context = json.loads(workspace_file_read(workspace["workspace_id"], "notes.md"))
    assert read_without_context["ok"] is False
    assert read_without_context["error"]["code"] == "context_not_loaded"

    search_without_context = json.loads(workspace_file_search(workspace["workspace_id"], "alpha"))
    assert search_without_context["ok"] is False
    assert search_without_context["error"]["code"] == "context_not_loaded"
    assert search_without_context["llm_calls"] == 0

    stat_without_context = json.loads(workspace_file_stat(workspace["workspace_id"], "notes.md"))
    assert stat_without_context["ok"] is False
    assert stat_without_context["error"]["code"] == "context_not_loaded"
    assert stat_without_context["llm_calls"] == 0

    token = json.loads(workspace_instructions_get(workspace["workspace_id"]))["context"]["context_token"]
    list_result = json.loads(
        workspace_file_list(workspace["workspace_id"], file_glob="*.md", context_token=token)
    )
    assert list_result["ok"] is True
    assert any(entry["path"] == "notes.md" for entry in list_result["file_list"]["entries"])

    read_result = json.loads(
        workspace_file_read(workspace["workspace_id"], "notes.md", offset=2, limit=1, context_token=token)
    )
    assert read_result["ok"] is True
    assert read_result["llm_calls"] == 0
    assert read_result["file"]["content"] == "beta\n"
    assert read_result["file"]["truncated"] is True

    stat_result = json.loads(
        workspace_file_stat(workspace["workspace_id"], "notes.md", context_token=token)
    )
    assert stat_result["ok"] is True
    assert stat_result["llm_calls"] == 0
    assert stat_result["stat"]["path"] == "notes.md"
    assert stat_result["stat"]["type"] == "file"
    assert stat_result["stat"]["size_bytes"] == notes.stat().st_size
    assert stat_result["stat"]["within_file_read_size_cap"] is True
    assert stat_result["stat"]["audit"] == {
        "tool": "workspace_file_stat",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert str(workspace_root) not in json.dumps(stat_result)

    secret = json.loads(workspace_file_read(workspace["workspace_id"], ".env.local", context_token=token))
    assert secret["ok"] is False
    assert secret["error"]["code"] == "secret_path_denied"
    assert secret["llm_calls"] == 0

    secret_stat = json.loads(workspace_file_stat(workspace["workspace_id"], ".env.local", context_token=token))
    assert secret_stat["ok"] is False
    assert secret_stat["error"]["code"] == "secret_path_denied"
    assert secret_stat["llm_calls"] == 0

    binary = json.loads(workspace_file_read(workspace["workspace_id"], "binary.bin", context_token=token))
    assert binary["ok"] is False
    assert binary["error"]["code"] == "binary_file_not_supported"
    assert binary["llm_calls"] == 0

    redacted = json.loads(workspace_file_read(workspace["workspace_id"], "config.md", context_token=token))
    assert redacted["ok"] is True
    assert "api_key=[REDACTED]" in redacted["file"]["content"]
    assert "abc123" not in redacted["file"]["content"]
    assert "visible=yes" in redacted["file"]["content"]

    secret_name = json.loads(workspace_file_read(workspace["workspace_id"], "tokens.txt", context_token=token))
    assert secret_name["ok"] is False
    assert secret_name["error"]["code"] == "secret_path_denied"

    git_metadata = json.loads(workspace_file_read(workspace["workspace_id"], ".git/config", context_token=token))
    assert git_metadata["ok"] is False
    assert git_metadata["error"]["code"] == "secret_path_denied"

    plan_list = json.loads(
        workspace_file_list(
            workspace["workspace_id"],
            path=".hermes/plans",
            file_glob="*.md",
            context_token=token,
        )
    )
    assert plan_list["ok"] is True
    assert [entry["path"] for entry in plan_list["file_list"]["entries"]] == [
        ".hermes/plans/implementation-plan.md"
    ]
    plan_read = json.loads(
        workspace_file_read(
            workspace["workspace_id"],
            ".hermes/plans/implementation-plan.md",
            context_token=token,
        )
    )
    assert plan_read["ok"] is True
    assert "Read-only plan context" in plan_read["file"]["content"]

    hermes_metadata = json.loads(
        workspace_file_read(workspace["workspace_id"], ".hermes/plans/state.json", context_token=token)
    )
    assert hermes_metadata["ok"] is False
    assert hermes_metadata["error"]["code"] == "secret_path_denied"

    search_result = json.loads(
        file_search(workspace["workspace_id"], "alpha", file_glob="*.md", context_token=token)
    )
    assert search_result["ok"] is True
    assert search_result["llm_calls"] == 0
    assert [match["line"] for match in search_result["search"]["matches"]] == [1, 3]
    assert search_result["search"]["skipped"]["binary"] == 0

    workspace_search = json.loads(
        workspace_file_search(
            workspace["workspace_id"],
            "alpha",
            file_glob="*.md",
            context_token=token,
        )
    )
    assert workspace_search["ok"] is True
    assert workspace_search["llm_calls"] == 0
    assert [match["line"] for match in workspace_search["search"]["matches"]] == [1, 3]

    files_only = json.loads(
        file_search(
            workspace["workspace_id"],
            "alpha",
            file_glob="*.md",
            output_mode="files_only",
            context_token=token,
        )
    )
    assert files_only["ok"] is True
    assert files_only["search"]["files"] == ["notes.md"]
    assert files_only["llm_calls"] == 0

    missing_workspace = json.loads(workspace_file_read("ws_missing", "notes.md", context_token=token))
    assert missing_workspace["ok"] is False
    assert missing_workspace["error"]["code"] == "workspace_not_found"
    assert missing_workspace["llm_calls"] == 0


def test_workspace_file_list_stops_after_limit_without_oversized_skipped(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    for index in range(40):
        (workspace_root / f"file-{index:02d}.md").write_text("alpha\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    listed = json.loads(workspace_file_list(workspace_id, limit=5, context_token=token))

    assert listed["ok"] is True
    assert listed["llm_calls"] == 0
    file_list = listed["file_list"]
    assert len(file_list["entries"]) == 5
    assert file_list["truncated"] is True
    assert len(file_list["skipped"]) <= 5
    assert any(item["reason"] == "file_limit_exceeded" for item in file_list["skipped"])
    assert str(workspace_root) not in json.dumps(listed)


def test_workspace_get_and_close_inspect_and_cleanup_registry(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "notes.md").write_text("alpha\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    inspected = json.loads(workspace_get(workspace_id))
    assert inspected["ok"] is True
    assert inspected["llm_calls"] == 0
    assert inspected["workspace"]["workspace_id"] == workspace_id
    assert inspected["workspace"]["profile_ref"] == "local:main-bot"
    assert "root" not in inspected["workspace"]

    closed = json.loads(workspace_close(workspace_id))
    assert closed["ok"] is True
    assert closed["closed"] is True
    assert closed["llm_calls"] == 0
    assert closed["workspace"]["workspace_id"] == workspace_id
    assert "root" not in closed["workspace"]

    after_close = json.loads(workspace_get(workspace_id))
    assert after_close["ok"] is False
    assert after_close["error"]["code"] == "workspace_not_found"
    assert after_close["llm_calls"] == 0

    read_after_close = json.loads(file_read(workspace_id, "notes.md"))
    assert read_after_close["ok"] is False
    assert read_after_close["error"]["code"] == "workspace_not_found"
    assert read_after_close["llm_calls"] == 0

    missing_close = json.loads(workspace_close("ws_missing"))
    assert missing_close["ok"] is False
    assert missing_close["error"]["code"] == "workspace_not_found"
    assert missing_close["llm_calls"] == 0


def test_profile_context_get_loads_bounded_soul_and_policy_without_secrets(
    hermes_home,
):
    profile_dir = hermes_home / "profiles" / "main-bot"
    profile_dir.joinpath("SOUL.md").write_text(
        "# Main Bot\nUse Spanish.\nOPENAI_API_KEY=super-secret-value\n",
        encoding="utf-8",
    )
    profile_dir.joinpath(".env").write_text("SHOULD_NOT_LEAK=1\n", encoding="utf-8")
    _write_router_config(hermes_home)

    result = json.loads(profile_context_get("local:main-bot"))
    assert result["ok"] is True
    assert result["llm_calls"] == 0
    context = result["context"]
    assert context["profile_ref"] == "local:main-bot"
    assert context["policy"]["allowed_roots"] == [str(hermes_home)]
    assert context["profile_instructions"][0]["path"] == "SOUL.md"
    assert "Use Spanish" in context["profile_instructions"][0]["excerpt"]
    dumped = json.dumps(context)
    assert "super-secret-value" not in dumped
    assert "SHOULD_NOT_LEAK" not in dumped
    assert context["secret_handling"]["funciones_txt_content_excluded"] is True


def test_profile_context_allows_safe_profile_soul_under_dot_hermes_only(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    profile_dir = hermes_home / "profiles" / "main-bot"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_dir.joinpath("SOUL.md").write_text(
        "# Main Bot\nSafe profile instructions.\nPASSWORD=must-not-leak\n",
        encoding="utf-8",
    )
    profile_dir.joinpath(".env").write_text("SHOULD_NOT_LEAK=1\n", encoding="utf-8")
    profile_dir.joinpath("auth.json").write_text(
        '{"access_token":"SHOULD_NOT_LEAK"}\n',
        encoding="utf-8",
    )
    profile_dir.joinpath("memories").mkdir()
    profile_dir.joinpath("memories", "MEMORY.md").write_text(
        "private memory should not be included\n",
        encoding="utf-8",
    )
    _write_router_config(hermes_home)

    result = json.loads(profile_context_get("local:main-bot"))

    assert result["ok"] is True
    assert result["llm_calls"] == 0
    context = result["context"]
    assert [item["path"] for item in context["profile_instructions"]] == ["SOUL.md"]
    assert "Safe profile instructions" in context["profile_instructions"][0]["excerpt"]
    dumped = json.dumps(context)
    assert "must-not-leak" not in dumped
    assert "SHOULD_NOT_LEAK" not in dumped
    assert "private memory" not in dumped


def test_profile_context_rejects_symlinked_profile_directory(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    external_profile = tmp_path / "external-profile"
    external_profile.mkdir()
    external_profile.joinpath("SOUL.md").write_text(
        "# External profile\nShould not be read through a profile symlink.\n",
        encoding="utf-8",
    )
    try:
        profiles_root.joinpath("main-bot").symlink_to(external_profile, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_router_config(hermes_home)

    result = json.loads(profile_context_get("local:main-bot"))

    assert result["ok"] is False
    assert result["error"]["code"] == "profile_symlink_denied"
    assert "Should not be read" not in json.dumps(result)


def test_profile_context_rejects_symlinked_profiles_parent(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    external_profiles = tmp_path / "external-profiles"
    external_profile = external_profiles / "main-bot"
    external_profile.mkdir(parents=True)
    external_profile.joinpath("SOUL.md").write_text(
        "# External profile\nShould not be read through a profiles-parent symlink.\n",
        encoding="utf-8",
    )
    try:
        hermes_home.joinpath("profiles").symlink_to(external_profiles, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_router_config(hermes_home)

    result = json.loads(profile_context_get("local:main-bot"))

    assert result["ok"] is False
    assert result["error"]["code"] == "profile_symlink_denied"
    assert "profiles-parent symlink" not in json.dumps(result)


def test_workspace_context_hydration_tracks_instruction_staleness_and_blocks_powerful_tools(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    agents = workspace_root / "AGENTS.md"
    agents.write_text("# Agents\nFollow project policy.\n", encoding="utf-8")
    (workspace_root / "funciones.txt").write_text(
        "private deployment notes that must not be returned\n",
        encoding="utf-8",
    )
    (hermes_home / "profiles" / "main-bot" / "SOUL.md").write_text(
        "# Profile\nProfile policy text.\n",
        encoding="utf-8",
    )

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    pre_status = json.loads(workspace_context_status(workspace_id))
    assert pre_status["ok"] is True
    assert pre_status["context_status"]["state"] == "not_loaded"
    with pytest.raises(ProfileRouterError, match="Workspace context must be loaded"):
        require_fresh_workspace_context(workspace_id)

    hydrated = json.loads(workspace_instructions_get(workspace_id))
    assert hydrated["ok"] is True
    assert hydrated["llm_calls"] == 0
    context = hydrated["context"]
    token = context["context_token"]
    assert context["workspace_instructions"][0]["path"] == "AGENTS.md"
    assert "Follow project policy" in context["workspace_instructions"][0]["excerpt"]
    assert context["funciones_txt"] == {
        "path": "funciones.txt",
        "exists": True,
        "content_included": False,
        "git_policy": "never_stage_commit_push_or_include_in_pr",
        "status": "excluded_from_context_bundle",
    }
    assert "private deployment notes" not in json.dumps(context)
    assert require_fresh_workspace_context(workspace_id, context_token=token).workspace_id == workspace_id

    loaded_status = json.loads(workspace_context_status(workspace_id))
    assert loaded_status["context_status"]["state"] == "loaded"

    agents.write_text("# Agents\nChanged policy.\n", encoding="utf-8")
    stale_status = json.loads(workspace_context_status(workspace_id))
    assert stale_status["context_status"]["state"] == "stale"
    with pytest.raises(ProfileRouterError, match="Workspace context is stale"):
        require_fresh_workspace_context(workspace_id, context_token=token)


def test_powerful_tool_wrappers_require_fresh_context_before_write_or_terminal(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    agents = workspace_root / "AGENTS.md"
    agents.write_text("# Agents\nInitial policy.\n", encoding="utf-8")
    notes = workspace_root / "notes.md"
    notes.write_text("alpha\n", encoding="utf-8")
    other_notes = workspace_root / "other.md"
    other_notes.write_text("one\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True, "write": True},
                "terminal": {"enabled": True},
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    patch_without_context = json.loads(
        file_patch(workspace_id, "notes.md", "alpha", "beta")
    )
    assert patch_without_context["ok"] is False
    assert patch_without_context["error"]["code"] == "context_not_loaded"
    assert patch_without_context["llm_calls"] == 0

    batch_patch_without_context = json.loads(
        patch_apply(
            workspace_id,
            [{"path": "notes.md", "old_string": "alpha", "new_string": "beta"}],
        )
    )
    assert batch_patch_without_context["ok"] is False
    assert batch_patch_without_context["error"]["code"] == "context_not_loaded"
    assert batch_patch_without_context["llm_calls"] == 0

    directory_without_context = json.loads(
        directory_create(workspace_id, "new-dir")
    )
    assert directory_without_context["ok"] is False
    assert directory_without_context["error"]["code"] == "context_not_loaded"
    assert directory_without_context["llm_calls"] == 0

    move_without_context = json.loads(
        file_move(workspace_id, "notes.md", "moved.md")
    )
    assert move_without_context["ok"] is False
    assert move_without_context["error"]["code"] == "context_not_loaded"
    assert move_without_context["llm_calls"] == 0
    assert notes.exists()
    assert not (workspace_root / "moved.md").exists()

    delete_without_context = json.loads(file_delete(workspace_id, "notes.md"))
    assert delete_without_context["ok"] is False
    assert delete_without_context["error"]["code"] == "context_not_loaded"
    assert delete_without_context["llm_calls"] == 0
    assert notes.exists()

    terminal_without_context = json.loads(
        terminal_run(workspace_id, "codex exec 'should not classify before context'")
    )
    assert terminal_without_context["ok"] is False
    assert terminal_without_context["error"]["code"] == "context_not_loaded"
    assert "terminal_command" not in terminal_without_context

    hydrated = json.loads(workspace_instructions_get(workspace_id))
    stale_token = hydrated["context"]["context_token"]
    agents.write_text("# Agents\nChanged policy.\n", encoding="utf-8")

    write_with_stale_context = json.loads(
        file_write(workspace_id, "notes.md", "beta\n", context_token=stale_token)
    )
    assert write_with_stale_context["ok"] is False
    assert write_with_stale_context["error"]["code"] == "context_stale"
    assert notes.read_text(encoding="utf-8") == "alpha\n"

    batch_patch_with_stale_context = json.loads(
        patch_apply(
            workspace_id,
            [{"path": "notes.md", "old_string": "alpha", "new_string": "beta"}],
            context_token=stale_token,
        )
    )
    assert batch_patch_with_stale_context["ok"] is False
    assert batch_patch_with_stale_context["error"]["code"] == "context_stale"
    assert notes.read_text(encoding="utf-8") == "alpha\n"

    directory_with_stale_context = json.loads(
        directory_create(workspace_id, "stale-dir", context_token=stale_token)
    )
    assert directory_with_stale_context["ok"] is False
    assert directory_with_stale_context["error"]["code"] == "context_stale"
    assert not (workspace_root / "stale-dir").exists()

    move_with_stale_context = json.loads(
        file_move(workspace_id, "notes.md", "stale-moved.md", context_token=stale_token)
    )
    assert move_with_stale_context["ok"] is False
    assert move_with_stale_context["error"]["code"] == "context_stale"
    assert notes.exists()
    assert not (workspace_root / "stale-moved.md").exists()

    delete_with_stale_context = json.loads(
        file_delete(workspace_id, "notes.md", context_token=stale_token)
    )
    assert delete_with_stale_context["ok"] is False
    assert delete_with_stale_context["error"]["code"] == "context_stale"
    assert notes.exists()

    refreshed = json.loads(workspace_instructions_get(workspace_id))
    fresh_token = refreshed["context"]["context_token"]
    patch_result = json.loads(
        file_patch(workspace_id, "notes.md", "alpha", "beta", context_token=fresh_token)
    )
    assert patch_result["ok"] is True
    assert patch_result["llm_calls"] == 0
    assert patch_result["patch"]["replacements"] == 1
    assert patch_result["patch"]["audit"] == {
        "tool": "file_patch",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert "-alpha" in patch_result["patch"]["diff"]["unified"]
    assert "+beta" in patch_result["patch"]["diff"]["unified"]
    assert str(workspace_root) not in json.dumps(patch_result)
    assert notes.read_text(encoding="utf-8") == "beta\n"

    batch_patch_result = json.loads(
        patch_apply(
            workspace_id,
            [
                {"path": "notes.md", "old_string": "beta", "new_string": "gamma"},
                {"path": "other.md", "old_string": "one", "new_string": "two"},
            ],
            context_token=fresh_token,
        )
    )
    assert batch_patch_result["ok"] is True
    assert batch_patch_result["llm_calls"] == 0
    assert batch_patch_result["patch_apply"]["patch_count"] == 2
    assert batch_patch_result["patch_apply"]["file_count"] == 2
    assert batch_patch_result["patch_apply"]["total_replacements"] == 2
    assert batch_patch_result["patch_apply"]["changed"] is True
    assert batch_patch_result["patch_apply"]["audit"] == {
        "tool": "patch_apply",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert [file["path"] for file in batch_patch_result["patch_apply"]["files"]] == [
        "notes.md",
        "other.md",
    ]
    assert notes.read_text(encoding="utf-8") == "gamma\n"
    assert other_notes.read_text(encoding="utf-8") == "two\n"
    assert str(workspace_root) not in json.dumps(batch_patch_result)

    write_result = json.loads(
        file_write(workspace_id, "created.txt", "created\n", context_token=fresh_token)
    )
    assert write_result["ok"] is True
    assert write_result["llm_calls"] == 0
    assert write_result["write"]["path"] == "created.txt"
    assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "created\n"

    move_result = json.loads(
        file_move(workspace_id, "created.txt", "moved.txt", context_token=fresh_token)
    )
    assert move_result["ok"] is True
    assert move_result["llm_calls"] == 0
    assert move_result["move"]["source_path"] == "created.txt"
    assert move_result["move"]["destination_path"] == "moved.txt"
    assert move_result["move"]["moved"] is True
    assert move_result["move"]["bytes_moved"] == len("created\n".encode("utf-8"))
    assert move_result["move"]["audit"] == {
        "tool": "file_move",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert not (workspace_root / "created.txt").exists()
    assert (workspace_root / "moved.txt").read_text(encoding="utf-8") == "created\n"
    assert str(workspace_root) not in json.dumps(move_result)

    delete_result = json.loads(file_delete(workspace_id, "moved.txt", context_token=fresh_token))
    assert delete_result["ok"] is True
    assert delete_result["llm_calls"] == 0
    assert delete_result["delete"]["path"] == "moved.txt"
    assert delete_result["delete"]["deleted"] is True
    assert delete_result["delete"]["bytes_deleted"] == len("created\n".encode("utf-8"))
    assert delete_result["delete"]["audit"] == {
        "tool": "file_delete",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert not (workspace_root / "moved.txt").exists()
    assert str(workspace_root) not in json.dumps(delete_result)

    directory_result = json.loads(
        directory_create(
            workspace_id,
            "created-dir/nested",
            parents=True,
            context_token=fresh_token,
        )
    )
    assert directory_result["ok"] is True
    assert directory_result["llm_calls"] == 0
    assert directory_result["directory"]["path"] == "created-dir/nested"
    assert directory_result["directory"]["created"] is True
    assert directory_result["directory"]["audit"] == {
        "tool": "directory_create",
        "llm_calls": 0,
        "root_exposed": False,
    }
    assert (workspace_root / "created-dir" / "nested").is_dir()
    assert str(workspace_root) not in json.dumps(directory_result)

    terminal_blocked_model = json.loads(
        terminal_run(
            workspace_id,
            "codex exec 'touch SHOULD_NOT_EXIST'",
            context_token=fresh_token,
        )
    )
    assert terminal_blocked_model["ok"] is False
    assert terminal_blocked_model["error"]["code"] == "terminal_command_blocked"
    assert terminal_blocked_model["terminal_command"]["blocked"] is True
    assert "model_command" in {
        reason["code"] for reason in terminal_blocked_model["terminal_command"]["reasons"]
    }
    assert terminal_blocked_model["llm_calls"] == 0

    terminal_disabled = json.loads(
        terminal_run(
            workspace_id,
            "git status --short",
            timeout=MAX_TERMINAL_TIMEOUT_SECONDS + 20,
            max_output_chars=MAX_TERMINAL_OUTPUT_CHARS + 20_000,
            context_token=fresh_token,
        )
    )
    assert terminal_disabled["ok"] is False
    assert terminal_disabled["error"]["code"] == "tool_disabled"
    assert terminal_disabled["terminal_command"]["blocked"] is False
    assert terminal_disabled["terminal_command"]["decision"] == "disabled_pending_execution_policy"
    assert terminal_disabled["terminal_command"]["working_directory"] == "."
    assert terminal_disabled["terminal_command"]["timeout_seconds"] == MAX_TERMINAL_TIMEOUT_SECONDS
    assert terminal_disabled["terminal_command"]["max_output_chars"] == MAX_TERMINAL_OUTPUT_CHARS
    assert terminal_disabled["terminal_command"]["audit"] == {
        "tool": "terminal_run",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "executes": False,
        "execution_policy_enabled": False,
        "allowlist_match": False,
        "execution_plan_available": False,
        "no_shell_compatible": True,
        "public_mcp_exposure": "disabled_pending_http_auth_config_review",
    }
    assert terminal_disabled["terminal_command"]["execution_plan"]["available"] is False
    assert terminal_disabled["terminal_command"]["execution_plan"]["argv"] is None
    assert terminal_disabled["llm_calls"] == 0
    assert not (workspace_root / "SHOULD_NOT_EXIST").exists()


def test_workspace_file_missing_error_does_not_expose_host_root(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    missing = json.loads(
        workspace_file_read(
            workspace_id,
            "tmp/chatgpt-hermes-smoke.txt",
            context_token=token,
        )
    )
    dumped = json.dumps(missing)
    assert missing["ok"] is False
    assert missing["error"]["code"] == "path_not_found"
    assert missing["llm_calls"] == 0
    assert str(workspace_root) not in dumped
    assert str(allowed_root) not in dumped
    assert str(tmp_path) not in dumped
    assert "/Users/" not in dumped
    assert "/home/" not in dumped


def test_chatgpt_safe_status_and_scratch_smokes_use_fixed_actions(
    hermes_home,
    monkeypatch,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    subprocess_calls = []

    def _fake_subprocess_run(argv, **kwargs):
        subprocess_calls.append((tuple(argv), kwargs))
        assert argv == ["pwd"]
        assert kwargs["cwd"] == str(workspace_root)
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 0, stdout=f"{workspace_root}\n", stderr="")

    monkeypatch.setattr(mcp_profile_router.subprocess, "run", _fake_subprocess_run)
    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True, "write": True},
                "terminal": {
                    "enabled": True,
                    "execution": {
                        "enabled": True,
                        "allowed_commands": ["pwd"],
                    },
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    status = json.loads(workspace_status_probe(workspace_id, context_token=token))
    assert status["ok"] is True
    assert status["llm_calls"] == 0
    assert status["status_probe"]["probe"]["stdout_workspace_marker_seen"] is True
    assert status["status_probe"]["probe"]["command_exposed"] is False
    assert status["status_probe"]["audit"] == {
        "tool": "workspace_status_probe",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "executes": True,
        "arbitrary_command_accepted": False,
    }

    preexisting = workspace_root / "tmp" / "chatgpt-hermes-action-smoke.txt"
    preexisting.parent.mkdir(parents=True, exist_ok=True)
    preexisting.write_text("do-not-touch\n", encoding="utf-8")

    scratch = json.loads(workspace_scratch_smoke(workspace_id, context_token=token))
    smoke = scratch["scratch_smoke"]
    assert scratch["ok"] is True
    assert scratch["llm_calls"] == 0
    assert smoke["path"].startswith("tmp/chatgpt-hermes-action-smoke-")
    assert smoke["path"].endswith(".txt")
    assert smoke["server_chosen_path"] is True
    assert smoke["arbitrary_path_accepted"] is False
    assert smoke["arbitrary_content_accepted"] is False
    assert smoke["write"]["ok"] is True
    assert smoke["read_initial"]["ok"] is True
    assert smoke["read_initial"]["content_sha256"] == hashlib.sha256(b"alpha\n").hexdigest()
    assert smoke["patch"]["ok"] is True
    assert smoke["read_patched"]["ok"] is True
    assert smoke["read_patched"]["content_sha256"] == hashlib.sha256(b"beta\n").hexdigest()
    assert smoke["cleanup"]["deleted"] is True
    assert preexisting.read_text(encoding="utf-8") == "do-not-touch\n"
    assert not (workspace_root / smoke["path"]).exists()

    dumped = json.dumps({"status": status, "scratch": scratch})
    assert "pwd" not in dumped
    assert "alpha" not in dumped
    assert "beta" not in dumped
    assert str(workspace_root) not in dumped
    assert str(allowed_root) not in dumped
    assert "/Users/" not in dumped
    assert "/home/" not in dumped
    assert len(subprocess_calls) == 1


def test_terminal_run_preflight_requires_policy_and_workspace_relative_cwd(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    subdir = workspace_root / "subdir"
    subdir.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    (workspace_root / "notes.md").write_text("notes\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    denied_by_policy = json.loads(
        terminal_run(workspace_id, "git status --short", context_token=token)
    )
    assert denied_by_policy["ok"] is False
    assert denied_by_policy["error"]["code"] == "terminal_not_allowed"
    assert "terminal_command" not in denied_by_policy
    assert denied_by_policy["llm_calls"] == 0

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {"enabled": True},
            }
        },
    )
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    outside_cwd = json.loads(
        terminal_run(
            workspace_id,
            "pwd",
            working_directory="../outside",
            context_token=token,
        )
    )
    assert outside_cwd["ok"] is False
    assert outside_cwd["error"]["code"] == "path_outside_workspace"

    file_cwd = json.loads(
        terminal_run(
            workspace_id,
            "pwd",
            working_directory="notes.md",
            context_token=token,
        )
    )
    assert file_cwd["ok"] is False
    assert file_cwd["error"]["code"] == "working_directory_not_directory"

    disabled = json.loads(
        terminal_run(
            workspace_id,
            "pwd",
            timeout=MAX_TERMINAL_TIMEOUT_SECONDS + 999,
            working_directory="subdir",
            max_output_chars=MAX_TERMINAL_OUTPUT_CHARS + 999,
            context_token=token,
        )
    )
    assert disabled["ok"] is False
    assert disabled["error"]["code"] == "tool_disabled"
    preflight = disabled["terminal_command"]
    assert preflight["blocked"] is False
    assert preflight["working_directory"] == "subdir"
    assert preflight["timeout_seconds"] == MAX_TERMINAL_TIMEOUT_SECONDS
    assert preflight["max_output_chars"] == MAX_TERMINAL_OUTPUT_CHARS
    assert preflight["policy"]["terminal_allowed"] is True
    assert preflight["audit"]["root_exposed"] is False
    assert str(workspace_root) not in json.dumps(disabled)


def test_terminal_run_executes_allowlisted_commands_with_sanitized_output(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {
                        "enabled": True,
                        "allowed_commands": ["pwd"],
                        "allowed_command_prefixes": ["git status"],
                    },
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    allowed = json.loads(terminal_run(workspace_id, "pwd", context_token=token))
    assert allowed["ok"] is True
    allowed_preflight = allowed["terminal_command"]
    assert allowed_preflight["blocked"] is False
    assert allowed_preflight["decision"] == "disabled_pending_execution_implementation"
    assert allowed_preflight["risk_level"] == "low_allowlisted_unexecuted"
    assert allowed_preflight["execution_policy"]["enabled"] is True
    assert allowed_preflight["execution_policy"]["allowlist_match"] is True
    assert allowed_preflight["execution_policy"]["allowlist_match_type"] == "exact"
    assert allowed_preflight["audit"]["executes"] is False
    assert allowed_preflight["audit"]["execution_policy_enabled"] is True
    assert allowed_preflight["audit"]["allowlist_match"] is True
    assert allowed_preflight["audit"]["execution_plan_available"] is True
    assert allowed_preflight["audit"]["no_shell_compatible"] is True
    plan = allowed_preflight["execution_plan"]
    assert plan["available"] is True
    assert plan["executes"] is False
    assert plan["shell"] is False
    assert plan["implementation_status"] == "private_no_shell_subprocess_runner_available"
    assert plan["argv"] == {
        "shell": False,
        "argv_redacted": True,
        "argc": 1,
        "argument_count": 0,
        "option_count": 0,
        "path_like_token_count": 0,
        "assignment_prefix_count": 0,
    }
    assert plan["env_policy"]["inherits_parent_env"] is False
    assert plan["env_policy"]["values_redacted"] is True
    assert "PATH" in plan["env_policy"]["allowed_keys"]
    assert "OPENAI_API_KEY" not in plan["env_policy"]["allowed_keys"]
    assert plan["cwd"] == {
        "workspace_relative": ".",
        "root_exposed": False,
        "resolved_host_path_exposed": False,
    }
    assert plan["limits"]["timeout_seconds"] == 30
    readiness = plan["execution_readiness_review"]
    assert readiness["gate"] == "terminal_execution_readiness_review"
    assert readiness["scope"] == "private_non_executing_terminal_scaffold"
    assert readiness["pre_executor_checks_passed"] is True
    assert readiness["current_phase_allows_subprocess_run"] is True
    assert readiness["subprocess_run_allowed"] is True
    assert readiness["public_mcp_subprocess_run_allowed"] is False
    assert readiness["real_executor_status"] == "private_direct_runner_enabled_public_mcp_blocked"
    assert readiness["fresh_context_enforced_before_gate"] is True
    assert readiness["raw_command_exposed"] is False
    assert readiness["argv_values_exposed"] is False
    assert readiness["env_values_exposed"] is False
    assert readiness["root_exposed"] is False
    assert readiness["llm_calls"] == 0
    assert readiness["failed_checks"] == []
    assert all(readiness["checks"].values())
    assert readiness["checks"]["fresh_context_validated_upstream"] is True
    assert readiness["checks"]["terminal_execution_policy_enabled"] is True
    assert readiness["checks"]["allowlist_match"] is True
    assert readiness["checks"]["sanitized_env"] is True
    assert readiness["checks"]["bounded_result_contract"] is True
    assert readiness["checks"]["public_mcp_absent_by_default"] is True
    assert readiness["sanitized_env"] == {
        "inherits_parent_env": False,
        "key_count": 6,
        "values_redacted": True,
    }
    pwd_result = allowed["terminal_result"]
    assert pwd_result["status"] == "success"
    assert pwd_result["returncode"] == 0
    assert pwd_result["stdout"]["text"].strip() == "<workspace>"
    assert pwd_result["audit"]["executes"] is True
    assert pwd_result["audit"]["execution_attempted"] is True
    assert pwd_result["audit"]["subprocess_run_allowed"] is True
    assert pwd_result["audit"]["subprocess_run_called"] is True
    assert pwd_result["audit"]["uses_shell"] is False
    assert str(workspace_root) not in json.dumps(allowed)
    assert "pwd" not in json.dumps(allowed)

    _git(workspace_root, "init")
    prefix_allowed = json.loads(
        terminal_run(workspace_id, "git status --short", context_token=token)
    )
    assert prefix_allowed["ok"] is True
    assert prefix_allowed["terminal_command"]["execution_policy"][
        "allowlist_match_type"
    ] == "prefix"
    prefix_plan = prefix_allowed["terminal_command"]["execution_plan"]
    assert prefix_plan["available"] is True
    assert prefix_plan["argv"]["argc"] == 3
    assert prefix_plan["argv"]["argument_count"] == 2
    assert prefix_plan["argv"]["option_count"] == 1
    assert prefix_allowed["terminal_result"]["status"] == "success"
    assert prefix_allowed["terminal_result"]["audit"]["executes"] is True
    assert "AGENTS.md" in prefix_allowed["terminal_result"]["stdout"]["text"]
    assert "git status --short" not in json.dumps(prefix_allowed)

    blocked_commands = {
        "python -m pytest": "terminal_command_not_allowlisted",
        "pwd && git status": "terminal_shell_control_not_allowed",
        "rm -rf tmp": "destructive_command",
        "git push origin main": "protected_git_command",
        "kubectl apply -f prod.yaml": "deploy_command",
    }
    for command, expected_reason in blocked_commands.items():
        blocked = json.loads(terminal_run(workspace_id, command, context_token=token))
        assert blocked["ok"] is False
        assert blocked["error"]["code"] == "terminal_command_blocked"
        assert blocked["terminal_command"]["execution_plan"]["available"] is False
        assert blocked["terminal_command"]["execution_plan"]["argv"] is None
        assert expected_reason in {
            reason["code"] for reason in blocked["terminal_command"]["reasons"]
        }
        assert "terminal_result" not in blocked


def test_terminal_private_runner_executes_allowlisted_commands_but_not_public(
    hermes_home,
    monkeypatch,
    tmp_path,
):
    subprocess_calls = []

    def _fake_subprocess_run(argv, **kwargs):
        subprocess_calls.append((argv, kwargs))
        assert argv == ["pwd"]
        assert kwargs["cwd"] == str(workspace_root)
        assert kwargs["env"] == _build_terminal_sanitized_env()
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{workspace_root}\n",
            stderr="",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    monkeypatch.setattr(mcp_profile_router.subprocess, "run", _fake_subprocess_run)
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {
                        "enabled": True,
                        "allowed_commands": ["pwd"],
                    },
                },
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    direct = json.loads(terminal_run(workspace_id, "pwd", context_token=token))
    assert direct["ok"] is True
    assert len(subprocess_calls) == 1

    command = direct["terminal_command"]
    assert command["decision"] == "disabled_pending_execution_implementation"
    assert command["audit"]["executes"] is False
    assert command["audit"]["execution_plan_available"] is True
    plan = command["execution_plan"]
    assert plan["available"] is True
    boundary = plan["executor_boundary"]
    assert boundary["adapter"] == "non_executing_terminal_executor_boundary"
    assert boundary["accepts_plan_type"] == "TerminalSubprocessPlan"
    assert boundary["implementation_status"] == (
        "pending_execution_policy_audit_public_exposure_review"
    )
    assert boundary["execution_attempted"] is False
    assert boundary["subprocess_run_allowed"] is False
    assert boundary["subprocess_run_called"] is False
    assert boundary["executes"] is False
    assert boundary["shell"] is False
    assert boundary["argc"] == 1
    assert boundary["env_values_exposed"] is False
    assert boundary["cwd"] == {
        "workspace_relative": ".",
        "root_exposed": False,
        "resolved_host_path_exposed": False,
    }
    assert boundary["result_contract"] == {
        "shape": "terminal_subprocess_result",
        "status_values": ["success", "failed", "timeout"],
        "stdout_stderr_bounded": True,
        "returncode_included": True,
        "timed_out_included": True,
        "max_output_chars": MAX_TERMINAL_OUTPUT_CHARS,
        "working_directory": ".",
        "root_exposed": False,
        "argv_values_exposed": False,
        "env_values_exposed": False,
        "uses_shell": False,
        "llm_calls": 0,
    }
    readiness = plan["execution_readiness_review"]
    assert readiness["pre_executor_checks_passed"] is True
    assert readiness["current_phase_allows_subprocess_run"] is True
    assert readiness["subprocess_run_allowed"] is True
    assert readiness["public_mcp_subprocess_run_allowed"] is False
    assert readiness["checks"]["executor_boundary_non_executing"] is True
    assert readiness["checks"]["tool_metadata_no_model"] is True
    assert readiness["checks"]["public_mcp_absent_by_default"] is True
    assert readiness["checks"]["fresh_context_validated_upstream"] is True
    assert readiness["failed_checks"] == []

    terminal_result = direct["terminal_result"]
    assert terminal_result["status"] == "success"
    assert terminal_result["returncode"] == 0
    assert terminal_result["stdout"]["text"].strip() == "<workspace>"
    assert terminal_result["audit"] == {
        "tool": "terminal_run",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "executes": True,
        "execution_attempted": True,
        "subprocess_run_allowed": True,
        "subprocess_run_called": True,
        "argv_redacted": True,
        "env_values_exposed": False,
        "public_mcp_exposure": "disabled_pending_http_auth_config_review",
    }

    metadata = get_router_tool_metadata()["terminal_run"]
    assert metadata["enabled_by_default"] is False
    assert metadata["requires_context"] is True
    assert metadata["cost_class"] == COST_CLASS_NO_MODEL
    assert metadata["llm_calls"] == 0

    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    server = mcp_serve.create_profile_router_mcp_server()
    assert "terminal_run" in server._tool_manager._tools
    assert "file_patch" in server._tool_manager._tools
    assert "patch_apply" in server._tool_manager._tools
    assert "file_write" in server._tool_manager._tools
    assert "file_move" in server._tool_manager._tools
    assert "file_delete" in server._tool_manager._tools
    assert "directory_create" in server._tool_manager._tools
    assert "workspace_status_probe" in server._tool_manager._tools
    assert "workspace_scratch_smoke" in server._tool_manager._tools
    assert "workspace_diff" in server._tool_manager._tools
    dumped = json.dumps(direct)
    assert "pwd" not in dumped
    assert str(workspace_root) not in dumped
    assert "/usr/bin" not in dumped
    assert "should-not-leak" not in dumped


def test_terminal_private_runner_shapes_failure_and_timeout_without_leaking_values(
    hermes_home,
    monkeypatch,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    calls = []

    def _fake_subprocess_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == str(workspace_root)
        assert kwargs["env"] == _build_terminal_sanitized_env()
        assert "OPENAI_API_KEY" not in kwargs["env"]
        if argv == ["python", "-c", "fail"]:
            return subprocess.CompletedProcess(
                argv,
                7,
                stdout=(
                    f"root={workspace_root}\n"
                    f"path={kwargs['env']['PATH']}\n"
                    "ok-prefix\n"
                ),
                stderr=(
                    f"cwd={workspace_root}\n"
                    f"locale={kwargs['env']['LC_ALL']}\n"
                    + "E" * 200
                ),
            )
        if argv == ["python", "-c", "timeout"]:
            raise subprocess.TimeoutExpired(
                argv,
                kwargs["timeout"],
                output=f"partial={workspace_root}\n{kwargs['env']['PATH']}\n",
                stderr=f"err={workspace_root}\n{kwargs['env']['TERM']}\n" + "T" * 200,
            )
        raise AssertionError(f"unexpected argv: {argv!r}")

    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret-should-not-leak")
    monkeypatch.setattr(mcp_profile_router.subprocess, "run", _fake_subprocess_run)
    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {
                        "enabled": True,
                        "allowed_commands": [
                            "python -c fail",
                            "python -c timeout",
                        ],
                    },
                },
            }
        },
    )

    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    failed = json.loads(
        terminal_run(
            workspace_id,
            "python -c fail",
            context_token=token,
            max_output_chars=90,
        )
    )
    assert failed["ok"] is False
    assert failed["error"]["code"] == "terminal_command_failed"
    failed_result = failed["terminal_result"]
    assert failed_result["status"] == "failed"
    assert failed_result["returncode"] == 7
    assert failed_result["timed_out"] is False
    assert failed_result["output"]["max_output_chars"] == 90
    assert failed_result["output"]["returned_chars"] <= 90
    assert failed_result["output"]["truncated"] is True
    assert failed_result["stdout"]["truncated"] is False
    assert failed_result["stderr"]["truncated"] is True
    assert "<workspace>" in failed_result["stdout"]["text"]
    assert "<redacted_env_value>" in failed_result["stdout"]["text"]
    assert "<workspace>" in failed_result["stderr"]["text"]
    assert failed_result["audit"] == {
        "tool": "terminal_run",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "executes": True,
        "execution_attempted": True,
        "subprocess_run_allowed": True,
        "subprocess_run_called": True,
        "argv_redacted": True,
        "env_values_exposed": False,
        "public_mcp_exposure": "disabled_pending_http_auth_config_review",
    }

    timed_out = json.loads(
        terminal_run(
            workspace_id,
            "python -c timeout",
            context_token=token,
            max_output_chars=70,
        )
    )
    assert timed_out["ok"] is False
    assert timed_out["error"]["code"] == "terminal_command_timeout"
    timeout_result = timed_out["terminal_result"]
    assert timeout_result["status"] == "timeout"
    assert timeout_result["returncode"] is None
    assert timeout_result["timed_out"] is True
    assert timeout_result["output"]["max_output_chars"] == 70
    assert timeout_result["output"]["returned_chars"] <= 70
    assert timeout_result["output"]["truncated"] is True
    assert "<workspace>" in timeout_result["stdout"]["text"]
    assert "<redacted_env_value>" in timeout_result["stdout"]["text"]
    assert "<workspace>" in timeout_result["stderr"]["text"]
    assert timeout_result["audit"]["executes"] is True
    assert timeout_result["audit"]["subprocess_run_allowed"] is True
    assert timeout_result["audit"]["subprocess_run_called"] is True
    assert len(calls) == 2

    for payload, raw_command in (
        (failed, "python -c fail"),
        (timed_out, "python -c timeout"),
    ):
        dumped = json.dumps(payload)
        assert raw_command not in dumped
        assert str(workspace_root) not in dumped
        assert "/usr/bin:/bin:/usr/sbin:/sbin" not in dumped
        assert "C.UTF-8" not in dumped
        assert "dumb" not in dumped
        assert "parent-secret-should-not-leak" not in dumped
        assert "OPENAI_API_KEY" not in dumped

    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    server = mcp_serve.create_profile_router_mcp_server()
    assert "terminal_run" in server._tool_manager._tools
    assert "file_patch" in server._tool_manager._tools
    assert "patch_apply" in server._tool_manager._tools
    assert "file_write" in server._tool_manager._tools
    assert "file_move" in server._tool_manager._tools
    assert "file_delete" in server._tool_manager._tools
    assert "directory_create" in server._tool_manager._tools


def test_process_tools_are_context_gated_and_tracked_only(hermes_home, tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    command = "python -c fake-bg"

    popen_calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            popen_calls.append((list(argv), kwargs))
            assert kwargs["shell"] is False
            assert kwargs["cwd"] == str(workspace_root)
            assert kwargs["env"] == _build_terminal_sanitized_env()
            assert "OPENAI_API_KEY" not in kwargs["env"]
            kwargs["stdout"].write(f"root={workspace_root}\npath={kwargs['env']['PATH']}\n")
            kwargs["stderr"].write(f"cwd={workspace_root}\nterm={kwargs['env']['TERM']}\n")
            kwargs["stdout"].flush()
            kwargs["stderr"].flush()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret-should-not-leak")
    monkeypatch.setattr(mcp_profile_router.subprocess, "Popen", FakePopen)

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {"enabled": True, "allowed_commands": [command]},
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    missing_context = json.loads(process_list(workspace_id))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "context_not_loaded"

    start_missing_context = json.loads(process_start(workspace_id, command))
    assert start_missing_context["ok"] is False
    assert start_missing_context["error"]["code"] == "context_not_loaded"

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    listed = json.loads(process_list(workspace_id, context_token=token, limit=999))
    assert listed["ok"] is True
    assert listed["llm_calls"] == 0
    assert listed["processes"]["processes"] == []
    assert listed["processes"]["process_count"] == 0
    assert listed["processes"]["limit"] == 50
    assert listed["processes"]["registry"] == {
        "enabled": True,
        "tracked_processes_only": True,
        "host_process_listing": False,
        "launch_supported": True,
        "status": "runtime_owned_background_process_registry",
    }
    assert listed["processes"]["audit"]["llm_calls"] == 0
    assert listed["processes"]["audit"]["host_process_listing"] is False
    assert listed["processes"]["audit"]["host_pid_exposed"] is False
    assert str(workspace_root) not in json.dumps(listed)

    started = json.loads(process_start(workspace_id, command, context_token=token, max_output_chars=120))
    dumped_started = json.dumps(started)
    assert started["ok"] is True
    assert started["llm_calls"] == 0
    assert started["process"]["status"] == "running"
    assert started["process"]["tracked_by_runtime"] is True
    assert started["process"]["host_pid_exposed"] is False
    assert started["process"]["raw_command_exposed"] is False
    assert started["process"]["root_exposed"] is False
    assert started["process"]["process_id"].startswith("proc_")
    assert command not in dumped_started
    assert str(workspace_root) not in dumped_started
    assert "parent-secret-should-not-leak" not in dumped_started
    assert "OPENAI_API_KEY" not in dumped_started
    assert popen_calls and popen_calls[0][0] == ["python", "-c", "fake-bg"]

    process_id = started["process"]["process_id"]
    listed_after_start = json.loads(process_list(workspace_id, context_token=token))
    assert listed_after_start["processes"]["process_count"] == 1
    assert listed_after_start["processes"]["processes"][0]["process_id"] == process_id
    assert listed_after_start["processes"]["processes"][0]["host_pid_exposed"] is False

    polled = json.loads(process_poll(workspace_id, process_id, context_token=token))
    assert polled["ok"] is True
    assert polled["process"]["status"] == "running"
    assert polled["process"]["host_pid_exposed"] is False

    logged_running = json.loads(process_log(workspace_id, process_id, context_token=token, max_chars=90))
    dumped_log = json.dumps(logged_running)
    assert logged_running["ok"] is True
    assert logged_running["log"]["available"] is True
    assert logged_running["log"]["max_chars"] == 90
    assert "<workspace>" in dumped_log
    assert "<redacted_env_value>" in dumped_log
    assert command not in dumped_log
    assert str(workspace_root) not in dumped_log
    assert "/usr/bin:/bin:/usr/sbin:/sbin" not in dumped_log
    assert "dumb" not in dumped_log

    killed = json.loads(process_kill(workspace_id, process_id, context_token=token))
    assert killed["ok"] is True
    assert killed["process"]["status"] == "killed"
    assert killed["process"]["host_pid_exposed"] is False

    opaque_id = "proc_should_not_be_echoed"
    for tool_fn in (process_poll, process_log, process_kill):
        result = json.loads(tool_fn(workspace_id, opaque_id, context_token=token))
        assert result["ok"] is False
        assert result["error"]["code"] == "process_not_found"
        assert result["llm_calls"] == 0
        assert result["process"]["tracked_by_runtime"] is False
        assert result["process"]["id_exposed"] is False
        assert result["process"]["host_pid_exposed"] is False
        assert opaque_id not in json.dumps(result)
        assert str(workspace_root) not in json.dumps(result)
    logged = json.loads(process_log(workspace_id, opaque_id, context_token=token, max_chars=999999))
    assert logged["log"] == {
        "available": False,
        "max_chars": 20000,
        "truncated": False,
        "root_exposed": False,
    }

    metadata = get_router_tool_metadata()
    for name in {"process_start", "process_list", "process_poll", "process_log", "process_kill"}:
        assert metadata[name]["enabled_by_default"] is False
        assert metadata[name]["requires_context"] is True
        assert metadata[name]["capability_group"] == "terminal"
        assert metadata[name]["llm_calls"] == 0


def test_process_tools_require_terminal_policy(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    start_denied = json.loads(process_start(workspace_id, "python -c fake-bg", context_token=token))
    assert start_denied["ok"] is False
    assert start_denied["error"]["code"] == "terminal_not_allowed"

    denied = json.loads(process_list(workspace_id, context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "terminal_not_allowed"


def test_workspace_python_run_is_context_gated_no_model_and_blocks_model_imports(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {"enabled": True, "allowed_commands": [sys.executable]},
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    missing_context = json.loads(workspace_python_run(workspace_id, "print(1 + 1)"))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "context_not_loaded"

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    result = json.loads(workspace_python_run(workspace_id, "print(1 + 1)", context_token=token))
    assert result["ok"] is True
    assert result["python"]["status"] == "success"
    assert result["python"]["stdout"]["text"].strip() == "2"
    assert result["llm_calls"] == 0
    assert result["python"]["audit"]["uses_shell"] is False
    assert result["python"]["audit"]["env_values_exposed"] is False
    assert result["python"]["audit"]["allowlist_redacted"] is True
    assert result["code"]["source_returned"] is False
    assert not (workspace_root / ".chatgpt-hermes-python").exists()

    nonzero = json.loads(
        workspace_python_run(
            workspace_id,
            "import sys\nprint('bad')\nsys.exit(3)",
            context_token=token,
        )
    )
    assert nonzero["ok"] is False
    assert nonzero["python"]["status"] == "failed"
    assert nonzero["python"]["returncode"] == 3
    assert nonzero["python"]["audit"]["llm_calls"] == 0

    timed_out = json.loads(
        workspace_python_run(workspace_id, "while True:\n    pass", timeout=1, context_token=token)
    )
    assert timed_out["ok"] is False
    assert timed_out["python"]["status"] == "timeout"
    assert timed_out["python"]["timed_out"] is True
    assert timed_out["python"]["audit"]["uses_shell"] is False

    blocked = json.loads(workspace_python_run(workspace_id, "import openai\nprint('x')", context_token=token))
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "python_model_path_denied"


def test_workspace_python_run_requires_python_executable_allowlist(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "terminal": {
                    "enabled": True,
                    "execution": {"enabled": True, "allowed_commands": ["git status --short"]},
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    denied = json.loads(workspace_python_run(workspace_id, "print('nope')", context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "python_command_not_allowlisted"
    assert denied["llm_calls"] == 0


def test_profile_skill_write_wrappers_require_policy_and_delete_intent(hermes_home):
    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "skills": {"write": True, "delete": True},
            }
        },
    )
    content = "---\nname: test-skill\ndescription: Safe helper.\n---\n\nInitial body.\n"

    created = json.loads(profile_skill_create("local:main-bot", "test-skill", content))
    assert created["ok"] is True
    assert created["skill_management"]["skill"]["id"] == "test-skill"
    assert created["llm_calls"] == 0

    patched = json.loads(
        profile_skill_patch("local:main-bot", "test-skill", "Initial body.", "Updated body.")
    )
    assert patched["ok"] is True
    assert "Updated body" in (hermes_home / "profiles" / "main-bot" / "skills" / "test-skill" / "SKILL.md").read_text(encoding="utf-8")

    support = json.loads(
        profile_skill_write_file("local:main-bot", "test-skill", "references/example.md", "Example text.\n")
    )
    assert support["ok"] is True
    assert support["skill_management"]["file"]["path"] == "references/example.md"

    missing_intent = json.loads(profile_skill_delete("local:main-bot", "test-skill", absorbed_into=""))
    assert missing_intent["ok"] is False
    assert missing_intent["error"]["code"] == "delete_intent_required"

    deleted = json.loads(
        profile_skill_delete("local:main-bot", "test-skill", absorbed_into="", confirm_delete=True)
    )
    assert deleted["ok"] is True
    assert not (hermes_home / "profiles" / "main-bot" / "skills" / "test-skill").exists()


def test_profile_memory_wrappers_use_exact_text_and_redacted_list(hermes_home):
    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "memory": {"write": True},
            }
        },
    )

    added = json.loads(profile_memory_add("local:main-bot", "Remember apples."))
    assert added["ok"] is True
    assert added["memory"]["added"] is True
    assert added["llm_calls"] == 0

    replaced = json.loads(
        profile_memory_replace("local:main-bot", "Remember apples.", "Remember pears.")
    )
    assert replaced["ok"] is True
    assert replaced["memory"]["entries"][0]["text"] == "Remember pears."

    not_exact = json.loads(profile_memory_remove("local:main-bot", "Remember apples."))
    assert not_exact["ok"] is False
    assert not_exact["error"]["code"] == "memory_entry_not_found"

    listed = json.loads(profile_memory_list("local:main-bot"))
    assert listed["ok"] is True
    assert listed["memory"]["entries"][0]["text"] == "Remember pears."

    removed = json.loads(profile_memory_remove("local:main-bot", "Remember pears."))
    assert removed["ok"] is True
    assert removed["memory"]["total_count"] == 0


def test_cron_tools_are_context_gated_script_only_and_no_model(hermes_home, tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    class FakeCronBackend:
        def __init__(self):
            self.jobs = {
                "script1": {
                    "id": "script1",
                    "name": "safe script API_KEY=should_not_leak",
                    "schedule_display": "every 5m",
                    "enabled": True,
                    "state": "scheduled",
                    "script": "safe.py",
                    "no_agent": True,
                    "prompt": "ignored prompt SECRET=hidden",
                },
                "agent1": {
                    "id": "agent1",
                    "name": "agent job",
                    "schedule_display": "every 1m",
                    "enabled": True,
                    "state": "scheduled",
                    "script": None,
                    "no_agent": False,
                    "prompt": "this would call a model SECRET=hidden",
                },
                "unapproved": {
                    "id": "unapproved",
                    "name": "unapproved script",
                    "schedule_display": "every 10m",
                    "enabled": True,
                    "state": "scheduled",
                    "script": "other.py",
                    "no_agent": True,
                },
            }
            self.created = []
            self.parsed_schedules = []

        def list_jobs(self, include_disabled=False):
            return [job for job in self.jobs.values() if include_disabled or job.get("enabled", True)]

        def resolve_job_ref(self, ref):
            return self.jobs.get(ref)

        def pause_job(self, job_id, reason=None):
            job = dict(self.jobs[job_id])
            job.update({"enabled": False, "state": "paused", "paused_reason": reason})
            self.jobs[job_id] = job
            return job

        def resume_job(self, job_id):
            job = dict(self.jobs[job_id])
            job.update({"enabled": True, "state": "scheduled"})
            self.jobs[job_id] = job
            return job

        def trigger_job(self, job_id):
            job = dict(self.jobs[job_id])
            job.update({"enabled": True, "state": "scheduled", "next_run_at": "2026-06-25T00:00:00+00:00"})
            self.jobs[job_id] = job
            return job

        def parse_schedule(self, schedule):
            self.parsed_schedules.append(schedule)
            if schedule == "bad schedule":
                raise ValueError("bad schedule")
            return {"kind": "interval", "minutes": 5, "display": schedule}

        def create_job(self, prompt, schedule, **kwargs):
            self.created.append({"prompt": prompt, "schedule": schedule, **kwargs})
            job = {
                "id": "created1",
                "name": kwargs.get("name") or "created",
                "schedule_display": schedule,
                "enabled": True,
                "state": "scheduled",
                "script": kwargs["script"],
                "no_agent": kwargs["no_agent"],
            }
            self.jobs[job["id"]] = job
            return job

    backend = FakeCronBackend()
    monkeypatch.setattr(mcp_profile_router, "_cron_jobs_backend", lambda: backend)

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "cron": {"enabled": True, "allowed_scripts": ["safe.py"]},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    missing_context = json.loads(cron_list(workspace_id))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "context_not_loaded"

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    listed = json.loads(cron_list(workspace_id, context_token=token, include_disabled=True, limit=999))
    assert listed["ok"] is True
    assert listed["llm_calls"] == 0
    cron_payload = listed["cron"]
    assert cron_payload["job_count"] == 3
    assert cron_payload["limit"] == 50
    assert cron_payload["policy"] == {
        "cron_enabled": True,
        "script_allowlist_count": 1,
        "model_backed_crons_allowed": False,
    }
    assert cron_payload["audit"]["llm_calls"] == 0
    by_id = {job["job_id"]: job for job in cron_payload["jobs"]}
    assert by_id["script1"]["script_only"] is True
    assert by_id["script1"]["script"]["allowed_by_profile_policy"] is True
    assert by_id["agent1"]["model_backed"] is True
    dumped_list = json.dumps(listed)
    assert "safe.py" not in dumped_list
    assert str(workspace_root) not in dumped_list
    assert "should_not_leak" not in dumped_list
    assert "SECRET=hidden" not in dumped_list

    denied_model = json.loads(cron_run(workspace_id, "agent1", context_token=token))
    assert denied_model["ok"] is False
    assert denied_model["error"]["code"] == "cron_model_backed_job_denied"
    assert denied_model["llm_calls"] == 0

    denied_unapproved = json.loads(cron_run(workspace_id, "unapproved", context_token=token))
    assert denied_unapproved["ok"] is False
    assert denied_unapproved["error"]["code"] == "cron_script_not_allowed"

    paused = json.loads(cron_pause(workspace_id, "script1", context_token=token, reason="manual"))
    assert paused["ok"] is True
    assert paused["cron"]["job"]["state"] == "paused"
    assert paused["cron"]["audit"]["model_backed_crons_allowed"] is False

    resumed = json.loads(cron_resume(workspace_id, "script1", context_token=token))
    assert resumed["ok"] is True
    assert resumed["cron"]["job"]["state"] == "scheduled"

    triggered = json.loads(cron_run(workspace_id, "script1", context_token=token))
    assert triggered["ok"] is True
    assert triggered["cron"]["audit"]["action"] == "trigger_next_tick"

    created = json.loads(
        cron_create_script_only(
            workspace_id,
            "every 5m",
            "safe.py",
            context_token=token,
            name="created SECRET=hidden",
            repeat=1,
        )
    )
    assert created["ok"] is True
    assert created["cron"]["job"]["job_id"] == "created1"
    assert backend.created[-1]["prompt"] == ""
    assert backend.created[-1]["script"] == "safe.py"
    assert backend.created[-1]["no_agent"] is True
    assert backend.created[-1]["model"] is None
    assert backend.created[-1]["provider"] is None
    assert backend.created[-1]["deliver"] == "local"
    assert backend.created[-1]["workdir"] == str(workspace_root)
    dumped_created = json.dumps(created)
    assert "safe.py" not in dumped_created
    assert str(workspace_root) not in dumped_created
    assert "hidden" not in dumped_created

    denied_create = json.loads(
        cron_create_script_only(workspace_id, "every 5m", "other.py", context_token=token)
    )
    assert denied_create["ok"] is False
    assert denied_create["error"]["code"] == "cron_script_not_allowed"

    metadata = get_router_tool_metadata()
    for name in {"cron_list", "cron_pause", "cron_resume", "cron_run", "cron_create_script_only"}:
        assert metadata[name]["enabled_by_default"] is False
        assert metadata[name]["requires_context"] is True
        assert metadata[name]["capability_group"] == "cron"
        assert metadata[name]["llm_calls"] == 0

    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    server = mcp_serve.create_profile_router_mcp_server()
    assert "cron_list" in server._tool_manager._tools
    assert "cron_create_script_only" in server._tool_manager._tools


def test_messaging_tools_are_context_gated_allowlisted_dry_run_only(hermes_home, tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "messaging": {
                    "enabled": True,
                    "allowed_recipients": ["telegram:-1001234567890", "email:ops@example.com"],
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    missing_context = json.loads(message_send(workspace_id, "telegram:-1001234567890", "hello"))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "context_not_loaded"
    assert missing_context["llm_calls"] == 0

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    denied_destination = json.loads(
        message_send(workspace_id, "telegram:-1009999999999", "hello", context_token=token)
    )
    assert denied_destination["ok"] is False
    assert denied_destination["error"]["code"] == "messaging_destination_not_allowed"
    assert denied_destination["llm_calls"] == 0

    secret_message = json.loads(
        message_send(workspace_id, "telegram:-1001234567890", "API_KEY=should_not_send", context_token=token)
    )
    assert secret_message["ok"] is False
    assert secret_message["error"]["code"] == "message_content_secret_denied"

    dry_run = json.loads(
        message_send(workspace_id, "telegram:-1001234567890", "safe status", context_token=token)
    )
    assert dry_run["ok"] is True
    assert dry_run["llm_calls"] == 0
    messaging = dry_run["messaging"]
    assert messaging["status"] == "dry_run_ready"
    assert messaging["platform"] == "telegram"
    assert messaging["destination_allowed"] is True
    assert messaging["destination_exposed"] is False
    assert messaging["message_content_logged"] is False
    assert messaging["delivery_attempted"] is False
    assert messaging["external_delivery_enabled"] is False
    assert messaging["policy"] == {
        "messaging_enabled": True,
        "allowed_recipients_count": 2,
        "allowlist_redacted": True,
        "broadcast_allowed": False,
    }
    assert messaging["audit"]["llm_calls"] == 0
    dumped = json.dumps(dry_run)
    assert "safe status" not in dumped
    assert "-1001234567890" not in dumped
    assert str(workspace_root) not in dumped

    delivery_denied = json.loads(
        message_send(workspace_id, "telegram:-1001234567890", "safe status", context_token=token, dry_run=False)
    )
    assert delivery_denied["ok"] is False
    assert delivery_denied["error"]["code"] == "message_delivery_not_enabled"

    telegram = json.loads(telegram_send(workspace_id, "-1001234567890", "telegram safe", context_token=token))
    assert telegram["ok"] is True
    assert telegram["messaging"]["platform"] == "telegram"
    assert telegram["messaging"]["audit"]["tool"] == "telegram_send"
    dumped_telegram = json.dumps(telegram)
    assert "telegram safe" not in dumped_telegram
    assert "-1001234567890" not in dumped_telegram

    metadata = get_router_tool_metadata()
    for name in {"message_send", "telegram_send"}:
        assert metadata[name]["enabled_by_default"] is False
        assert metadata[name]["requires_context"] is True
        assert metadata[name]["capability_group"] == "messaging"
        assert metadata[name]["llm_calls"] == 0

    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    server = mcp_serve.create_profile_router_mcp_server()
    assert "message_send" in server._tool_manager._tools
    assert "telegram_send" in server._tool_manager._tools


def test_messaging_real_send_requires_delivery_policy_and_redacts_payload(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "messaging": {
                    "enabled": True,
                    "allowed_recipients": ["telegram:-1001234567890"],
                    "delivery": {
                        "enabled": True,
                        "command_argv": [
                            sys.executable,
                            "-c",
                            "import sys; print(sys.argv[1]); print(sys.argv[2])",
                            "{destination}",
                            "{message}",
                        ],
                    },
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    sent = json.loads(message_send(workspace_id, "telegram:-1001234567890", "safe status", context_token=token, dry_run=False))
    assert sent["ok"] is True
    messaging = sent["messaging"]
    assert messaging["status"] == "delivery_attempted"
    assert messaging["delivery_attempted"] is True
    assert messaging["external_delivery_enabled"] is True
    assert messaging["delivery_result"]["status"] == "success"
    dumped = json.dumps(sent)
    assert "safe status" not in dumped
    assert "-1001234567890" not in dumped
    assert str(workspace_root) not in dumped
    assert sent["llm_calls"] == 0


def test_production_server_and_web_wrappers_are_policy_gated_and_no_model(
    hermes_home,
    tmp_path,
    monkeypatch,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        extra_profile_router={
            "server_aliases": {
                "local_ops": {
                    "transport": "local",
                    "allowed_ports": [9],
                    "command_groups": {
                        "status": [sys.executable, "-c", "print('status-ok')"],
                        "deploy_echo": [sys.executable, "-c", "import sys; print('command-ok:' + sys.argv[1])", "{project}"],
                    },
                }
            }
        },
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "deploy": {
                    "enabled": True,
                    "actions": {
                        "smoke_deploy": {
                            "argv": [sys.executable, "-c", "print('deploy-ok')"],
                            "rollback_action": "smoke_rollback",
                        },
                        "smoke_ssh_deploy": {
                            "server_alias": "local_ops",
                            "server_command": "deploy_echo",
                            "rollback_action": "smoke_rollback",
                        },
                    },
                },
                "server": {
                    "enabled": True,
                    "allowed_aliases": ["local_ops"],
                    "allow_status": True,
                    "allow_commands": True,
                    "allow_shell": True,
                },
                "web": {
                    "enabled": True,
                    "allowed_domains": ["*"],
                    "max_bytes": 2000,
                },
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    actions = json.loads(workspace_production_action_list(workspace_id, context_token=token))
    assert actions["ok"] is True
    assert actions["production_actions"]["actions"][0]["name"] == "smoke_deploy"
    ran = json.loads(workspace_production_action_run(workspace_id, "smoke_deploy", context_token=token))
    assert ran["ok"] is True
    assert ran["production_action"]["result"]["status"] == "success"
    assert ran["production_action"]["result"]["audit"]["llm_calls"] == 0
    assert ran["production_action"]["action"]["execution_target"] == "workspace_argv"
    assert str(workspace_root) not in json.dumps(ran)

    ssh_ran = json.loads(workspace_production_action_run(workspace_id, "smoke_ssh_deploy", context_token=token))
    assert ssh_ran["ok"] is True
    ssh_action = ssh_ran["production_action"]["action"]
    assert ssh_action["execution_target"] == "server_alias"
    assert ssh_action["server_alias"] == "local_ops"
    assert ssh_action["server_command"] == "deploy_echo"
    assert ssh_action["server_target_exposed"] is False
    assert ssh_ran["production_action"]["result"]["status"] == "success"
    assert "command-ok:project" in ssh_ran["production_action"]["result"]["stdout"]["text"]
    assert ssh_ran["production_action"]["result"]["audit"]["execution_target"] == "server_alias"
    assert ssh_ran["production_action"]["result"]["audit"]["server_target_exposed"] is False
    assert ssh_ran["production_action"]["result"]["audit"]["llm_calls"] == 0
    assert str(workspace_root) not in json.dumps(ssh_ran)

    status = json.loads(workspace_production_action_status(workspace_id, "smoke_deploy", context_token=token))
    assert status["ok"] is True
    assert status["production_action_status"]["action"]["rollback_action"] == "smoke_rollback"

    aliases = json.loads(server_alias_list("local:main-bot"))
    assert aliases["ok"] is True
    assert aliases["servers"]["aliases"][0]["ssh_target_exposed"] is False
    server_status = json.loads(server_status_check("local:main-bot", "local_ops"))
    assert server_status["ok"] is True
    assert server_status["server_status"]["result"]["status"] == "success"
    command = json.loads(server_command_run("local:main-bot", "local_ops", "deploy_echo"))
    assert command["ok"] is True
    assert command["server_command"]["result"]["audit"]["llm_calls"] == 0
    shell = json.loads(server_shell_run("local:main-bot", "local_ops", "printf shell-ok", timeout_seconds=5, max_output_chars=1000))
    assert shell["ok"] is True
    assert shell["server_shell"]["result"]["status"] == "success"
    assert shell["server_shell"]["result"]["stdout"]["text"] == "shell-ok"
    assert shell["server_shell"]["result"]["audit"]["llm_calls"] == 0
    assert shell["server_shell"]["result"]["audit"]["raw_command_exposed"] is False
    assert "printf shell-ok" not in json.dumps(shell)
    blocked_shell = json.loads(server_shell_run("local:main-bot", "local_ops", "cat .env"))
    assert blocked_shell["ok"] is False
    assert blocked_shell["error"]["code"] == "server_shell_secret_path_denied"

    class FakeResponse:
        status = 200
        headers = {"content-type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit):
            return b"TOKEN=*** public-ok"

    monkeypatch.setattr(
        mcp_profile_router.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(mcp_profile_router.socket.AF_INET, mcp_profile_router.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))],
    )
    monkeypatch.setattr(mcp_profile_router, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    fetched = json.loads(workspace_web_fetch(workspace_id, "https://example.com/status?secret=hidden", context_token=token))
    assert fetched["ok"] is True
    assert fetched["web_fetch"]["fetch"]["hostname"] == "example.com"
    dumped_fetch = json.dumps(fetched)
    assert "secret=hidden" not in dumped_fetch
    assert fetched["web_fetch"]["audit"]["llm_calls"] == 0


def test_file_write_is_denied_without_policy_and_for_secret_or_symlink_paths(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    notes = workspace_root / "notes.md"
    notes.write_text("alpha alpha\n", encoding="utf-8")
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_file = outside_root / "leak.txt"
    outside_file.write_text("leak\n", encoding="utf-8")
    outside_dir = outside_root / "dir"
    outside_dir.mkdir()
    link = workspace_root / "link"
    link_dir = workspace_root / "linkdir"
    try:
        link.symlink_to(outside_file)
        link_dir.symlink_to(outside_dir)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    denied = json.loads(file_write(workspace_id, "notes.md", "beta\n", context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "filesystem_write_not_allowed"
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"

    mkdir_denied = json.loads(directory_create(workspace_id, "new-dir", context_token=token))
    assert mkdir_denied["ok"] is False
    assert mkdir_denied["error"]["code"] == "filesystem_write_not_allowed"
    assert not (workspace_root / "new-dir").exists()

    move_denied = json.loads(file_move(workspace_id, "notes.md", "renamed.md", context_token=token))
    assert move_denied["ok"] is False
    assert move_denied["error"]["code"] == "filesystem_write_not_allowed"
    assert notes.exists()
    assert not (workspace_root / "renamed.md").exists()

    delete_denied = json.loads(file_delete(workspace_id, "notes.md", context_token=token))
    assert delete_denied["ok"] is False
    assert delete_denied["error"]["code"] == "filesystem_write_not_allowed"
    assert notes.exists()

    batch_patch_denied = json.loads(
        patch_apply(
            workspace_id,
            [{"path": "notes.md", "old_string": "alpha", "new_string": "beta"}],
            context_token=token,
        )
    )
    assert batch_patch_denied["ok"] is False
    assert batch_patch_denied["error"]["code"] == "filesystem_write_not_allowed"
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True, "write": True},
            }
        },
    )
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    not_unique = json.loads(
        file_patch(workspace_id, "notes.md", "alpha", "beta", context_token=token)
    )
    assert not_unique["ok"] is False
    assert not_unique["error"]["code"] == "patch_match_not_unique"
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"

    batch_missing_file = json.loads(
        patch_apply(
            workspace_id,
            [
                {
                    "path": "notes.md",
                    "old_string": "alpha",
                    "new_string": "beta",
                    "replace_all": True,
                },
                {"path": "missing.md", "old_string": "missing", "new_string": "created"},
            ],
            context_token=token,
        )
    )
    assert batch_missing_file["ok"] is False
    assert batch_missing_file["error"]["code"] == "not_a_file"
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"

    duplicate_batch = json.loads(
        patch_apply(
            workspace_id,
            [
                {"path": "notes.md", "old_string": "alpha", "new_string": "beta", "replace_all": True},
                {"path": "notes.md", "old_string": "alpha", "new_string": "gamma", "replace_all": True},
            ],
            context_token=token,
        )
    )
    assert duplicate_batch["ok"] is False
    assert duplicate_batch["error"]["code"] == "patch_duplicate_path"
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"

    for blocked_path, expected_code in (
        (".env", "secret_path_denied"),
        ("../outside.txt", "path_outside_workspace"),
        ("link", "symlink_traversal_denied"),
    ):
        blocked = json.loads(
            file_write(workspace_id, blocked_path, "blocked\n", context_token=token)
        )
        assert blocked["ok"] is False
        assert blocked["error"]["code"] == expected_code
        batch_blocked = json.loads(
            patch_apply(
                workspace_id,
                [{"path": blocked_path, "old_string": "alpha", "new_string": "beta"}],
                context_token=token,
            )
        )
        assert batch_blocked["ok"] is False
        assert batch_blocked["error"]["code"] == expected_code
    assert outside_file.read_text(encoding="utf-8") == "leak\n"

    for blocked_path, expected_code in (
        (".env.d", "secret_path_denied"),
        ("../outside-dir", "path_outside_workspace"),
        ("linkdir/child", "symlink_traversal_denied"),
    ):
        blocked = json.loads(directory_create(workspace_id, blocked_path, parents=True, context_token=token))
        assert blocked["ok"] is False
        assert blocked["error"]["code"] == expected_code
    assert not (outside_dir / "child").exists()

    for source_path, destination_path, expected_code in (
        (".env", "moved-secret", "secret_path_denied"),
        ("../outside.txt", "moved-outside", "path_outside_workspace"),
        ("link", "moved-link", "symlink_traversal_denied"),
        ("notes.md", "linkdir/moved-notes.md", "symlink_traversal_denied"),
    ):
        blocked = json.loads(
            file_move(
                workspace_id,
                source_path,
                destination_path,
                context_token=token,
            )
        )
        assert blocked["ok"] is False
        assert blocked["error"]["code"] == expected_code
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"
    assert outside_file.read_text(encoding="utf-8") == "leak\n"
    assert not (outside_dir / "moved-notes.md").exists()

    for blocked_path, expected_code in (
        (".env", "secret_path_denied"),
        ("../outside.txt", "path_outside_workspace"),
        ("link", "symlink_traversal_denied"),
        ("linkdir/child", "symlink_traversal_denied"),
    ):
        blocked = json.loads(file_delete(workspace_id, blocked_path, context_token=token))
        assert blocked["ok"] is False
        assert blocked["error"]["code"] == expected_code
    assert notes.read_text(encoding="utf-8") == "alpha alpha\n"
    assert outside_file.read_text(encoding="utf-8") == "leak\n"


def test_workspace_diff_is_context_gated_bounded_and_filters_local_metadata(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    notes = workspace_root / "notes.md"
    notes.write_text("alpha\n", encoding="utf-8")
    funciones = workspace_root / "funciones.txt"
    funciones.write_text("private deployment notes\n", encoding="utf-8")
    plans_dir = workspace_root / ".hermes" / "plans"
    plans_dir.mkdir(parents=True)
    plan_state = plans_dir / "state.json"
    plan_state.write_text("local plan state\n", encoding="utf-8")
    plan_md = plans_dir / "implementation-plan.md"
    plan_md.write_text("# Plan\nalpha plan\n", encoding="utf-8")

    _git(workspace_root, "init")
    _git(workspace_root, "config", "user.email", "router-test@example.invalid")
    _git(workspace_root, "config", "user.name", "Router Test")
    textconv_marker = workspace_root / "textconv-ran.txt"
    textconv_script = workspace_root / "textconv.py"
    textconv_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(textconv_marker)!r}).write_text('ran', encoding='utf-8')\n"
        "print(Path(sys.argv[1]).read_text(encoding='utf-8'), end='')\n",
        encoding="utf-8",
    )
    (workspace_root / ".gitattributes").write_text("*.md diff=routertextconv\n", encoding="utf-8")
    _git(workspace_root, "config", "diff.routertextconv.textconv", f"{sys.executable} {textconv_script}")
    _git(
        workspace_root,
        "add",
        "AGENTS.md",
        "notes.md",
        "funciones.txt",
        ".hermes/plans/state.json",
        ".hermes/plans/implementation-plan.md",
        ".gitattributes",
        "textconv.py",
    )
    _git(workspace_root, "commit", "-m", "initial")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True, "write": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    without_context = json.loads(workspace_diff(workspace_id))
    assert without_context["ok"] is False
    assert without_context["error"]["code"] == "context_not_loaded"
    assert without_context["llm_calls"] == 0

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    notes.write_text("beta\n", encoding="utf-8")
    funciones.write_text("private deployment notes changed\n", encoding="utf-8")
    plan_state.write_text("local plan state changed\n", encoding="utf-8")
    plan_md.write_text("# Plan\nbeta plan\n", encoding="utf-8")
    (workspace_root / "new.txt").write_text("new file\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=should-not-leak\n", encoding="utf-8")

    result = json.loads(workspace_diff(workspace_id, context_token=token))
    assert result["ok"] is True
    assert result["llm_calls"] == 0
    diff = result["workspace_diff"]
    assert diff["tracked_files"] == [".hermes/plans/implementation-plan.md", "notes.md"]
    assert diff["untracked_files"] == ["new.txt"]
    skipped = {(item["path"], item["reason"]) for item in diff["skipped"]}
    assert ("funciones.txt", "protected_local_metadata") in skipped
    assert (".hermes/plans/state.json", "protected_local_metadata") in skipped
    assert (".env", "secret_path_denied") in skipped
    assert diff["audit"] == {
        "tool": "workspace_diff",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "git_read_only": True,
        "public_mcp_exposure": "enabled_read_only_v1",
    }
    assert not textconv_marker.exists()
    assert "-alpha" in diff["diff"]["unified"]
    assert "+beta" in diff["diff"]["unified"]
    assert "-alpha plan" in diff["diff"]["unified"]
    assert "+beta plan" in diff["diff"]["unified"]
    dumped = json.dumps(result)
    assert str(workspace_root) not in dumped
    assert "private deployment" not in dumped
    assert "should-not-leak" not in dumped


def test_git_read_only_wrappers_require_context_and_git_policy_and_filter_secrets(
    hermes_home,
    tmp_path,
):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    notes = workspace_root / "notes.md"
    notes.write_text("alpha\n", encoding="utf-8")

    _git(workspace_root, "init")
    _git(workspace_root, "config", "user.email", "router-test@example.invalid")
    _git(workspace_root, "config", "user.name", "Router Test")
    _git(workspace_root, "add", "AGENTS.md", "notes.md")
    _git(workspace_root, "commit", "-m", "initial")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
            }
        },
    )
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]

    missing_context = json.loads(git_status(workspace_id))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "context_not_loaded"
    assert missing_context["llm_calls"] == 0

    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    denied = json.loads(git_status(workspace_id, context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "git_read_not_allowed"
    assert denied["llm_calls"] == 0

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "git": {"enabled": True, "protected_branches": ["main", "master"]},
            }
        },
    )
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    notes.write_text("beta\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=should-not-leak\n", encoding="utf-8")

    status = json.loads(git_status(workspace_id, context_token=token, limit=999))
    assert status["ok"] is True
    assert status["llm_calls"] == 0
    status_payload = status["git_status"]
    assert status_payload["limit"] == 100
    assert {entry["path"] for entry in status_payload["changes"]} == {"notes.md"}
    assert (".env", "secret_path_denied") in {
        (item.get("path"), item.get("reason")) for item in status_payload["skipped"]
    }
    assert status_payload["audit"] == {
        "tool": "git_status",
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "git_read_only": True,
        "git_mutation_allowed": False,
        "public_mcp_exposure": "disabled_pending_git_parity_policy_review",
        "workspace_id": workspace_id,
    }

    diff = json.loads(git_diff(workspace_id, context_token=token))
    assert diff["ok"] is True
    assert diff["git_diff"]["tracked_files"] == ["notes.md"]
    assert "-alpha" in diff["git_diff"]["diff"]["unified"]
    assert "+beta" in diff["git_diff"]["diff"]["unified"]

    log = json.loads(git_log(workspace_id, context_token=token, limit=1))
    assert log["ok"] is True
    assert log["git_log"]["commit_count"] == 1
    assert log["git_log"]["commits"][0]["subject"] == "initial"
    assert log["git_log"]["audit"]["llm_calls"] == 0

    branches = json.loads(git_branch(workspace_id, context_token=token))
    assert branches["ok"] is True
    branch_payload = branches["git_branch"]
    assert branch_payload["current"] in {"main", "master"}
    current = [branch for branch in branch_payload["branches"] if branch["current"]]
    assert len(current) == 1
    assert current[0]["protected"] is True

    dumped = json.dumps([status, diff, log, branches])
    assert str(workspace_root) not in dumped
    assert "should-not-leak" not in dumped


def test_git_owner_write_wrappers_require_context_and_explicit_policy(hermes_home, tmp_path):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    notes = workspace_root / "notes.md"
    notes.write_text("alpha\n", encoding="utf-8")
    _git(workspace_root, "init")
    _git(workspace_root, "config", "user.email", "router-test@example.invalid")
    _git(workspace_root, "config", "user.name", "Router Test")
    _git(workspace_root, "add", "AGENTS.md", "notes.md")
    _git(workspace_root, "commit", "-m", "initial")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "git": {"enabled": True, "protected_branches": ["production"]},
            }
        },
    )
    workspace_id = json.loads(workspace_open("local:main-bot", str(workspace_root)))["workspace"]["workspace_id"]
    no_context = json.loads(git_add(workspace_id, ["notes.md"]))
    assert no_context["ok"] is False
    assert no_context["error"]["code"] == "context_not_loaded"
    assert no_context["llm_calls"] == 0
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    denied = json.loads(git_add(workspace_id, ["notes.md"], context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "git_write_not_allowed"
    assert denied["llm_calls"] == 0

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "git": {"enabled": True, "write": True, "protected_branches": ["production"]},
            }
        },
    )
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]
    notes.write_text("beta\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=should-not-leak\n", encoding="utf-8")

    secret_denied = json.loads(git_add(workspace_id, [".env"], context_token=token))
    assert secret_denied["ok"] is False
    assert secret_denied["error"]["code"] == "git_path_denied"
    assert secret_denied["llm_calls"] == 0

    add = json.loads(git_add(workspace_id, ["notes.md"], context_token=token))
    assert add["ok"] is True
    assert add["llm_calls"] == 0
    assert add["git_add"]["audit"]["llm_calls"] == 0
    assert add["git_add"]["audit"]["uses_shell"] is False

    commit = json.loads(git_commit(workspace_id, "owner mode git write", context_token=token))
    assert commit["ok"] is True
    assert commit["llm_calls"] == 0
    assert commit["git_commit"]["message_returned"] is False
    assert commit["git_commit"]["audit"]["git_mutation_allowed"] is True

    checkout = json.loads(git_checkout(workspace_id, "feature/router-owner-mode", context_token=token, create=True))
    assert checkout["ok"] is True
    assert checkout["git_checkout"]["created"] is True
    notes.write_text("gamma\n", encoding="utf-8")
    restored = json.loads(git_restore(workspace_id, ["notes.md"], context_token=token))
    assert restored["ok"] is True
    assert notes.read_text(encoding="utf-8") == "beta\n"

    dump = json.dumps([add, commit, checkout, restored])
    assert str(workspace_root) not in dump
    assert "should-not-leak" not in dump
    assert "GITHUB_TOKEN" not in dump
    assert "GH_TOKEN" not in dump


def test_github_pr_wrappers_are_policy_gated_and_hide_tokens(hermes_home, tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    workspace_root = allowed_root / "project"
    workspace_root.mkdir(parents=True)
    (workspace_root / "AGENTS.md").write_text("# Agents\nPolicy.\n", encoding="utf-8")
    _git(workspace_root, "init")
    _git(workspace_root, "config", "user.email", "router-test@example.invalid")
    _git(workspace_root, "config", "user.name", "Router Test")
    _git(workspace_root, "remote", "add", "origin", "https://github.com/example/private-repo.git")
    (workspace_root / "README.md").write_text("readme\n", encoding="utf-8")
    _git(workspace_root, "add", "AGENTS.md", "README.md")
    _git(workspace_root, "commit", "-m", "initial")

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "git": {"enabled": True},
                "github_pr": {"enabled": True, "allow_merge": True, "allow_issue_comment": True},
            }
        },
    )
    workspace_id = json.loads(workspace_open("local:main-bot", str(workspace_root)))["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    calls = []

    def fake_gh(workspace, args, *, timeout=60):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=0,
            stdout=json.dumps({"url": "https://github.com/example/private-repo/pull/1", "number": 1}),
            stderr="",
        )

    monkeypatch.setattr(mcp_profile_router, "_run_workspace_gh", fake_gh)
    payloads = [
        json.loads(github_pr_status(workspace_id, context_token=token)),
        json.loads(github_pr_create(workspace_id, "Add owner mode wrappers", context_token=token, body="Safe body", draft=True)),
        json.loads(github_pr_update(workspace_id, 1, context_token=token, labels=["safe-label"])),
        json.loads(github_pr_ready(workspace_id, 1, context_token=token)),
        json.loads(github_pr_merge(workspace_id, 1, context_token=token, method="merge")),
        json.loads(github_issue_view(workspace_id, 2, context_token=token)),
        json.loads(github_issue_comment(workspace_id, 2, "Safe comment", context_token=token)),
    ]
    assert calls
    assert all(payload["ok"] is True and payload["llm_calls"] == 0 for payload in payloads)
    dumped = json.dumps(payloads)
    assert str(workspace_root) not in dumped
    assert "GH_TOKEN" not in dumped
    assert "GITHUB_TOKEN" not in dumped
    assert "Safe comment" not in dumped

    _write_router_config(
        hermes_home,
        host_roots=[str(allowed_root)],
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(allowed_root)],
                "filesystem": {"read": True},
                "git": {"enabled": True},
                "github_pr": {"enabled": False},
            }
        },
    )
    denied = json.loads(github_pr_status(workspace_id, context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "github_pr_not_allowed"
    assert denied["llm_calls"] == 0


def test_missing_profile_router_policy_exposes_no_profiles_by_default(hermes_home):
    result = json.loads(profiles_list())
    assert result["ok"] is True
    assert result["profiles"] == []
    assert result["cost_class"] == COST_CLASS_NO_MODEL
    assert result["llm_calls"] == 0


def test_profiles_list_returns_policy_enabled_local_profile_refs_without_secret_paths(
    hermes_home,
):
    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "display_name": "Main Bot",
                "allowed_roots": [str(hermes_home)],
            },
            "local:maker": {
                "enabled": False,
                "allowed_roots": [str(hermes_home)],
            },
        },
    )

    result = json.loads(profiles_list())
    assert result["ok"] is True
    assert result["cost_class"] == COST_CLASS_NO_MODEL
    assert result["llm_calls"] == 0

    refs = {profile["profile_ref"] for profile in result["profiles"]}
    assert refs == {"local:main-bot"}
    enabled_profile = result["profiles"][0]
    assert enabled_profile["display_name"] == "Main Bot"
    assert enabled_profile["policy"]["enabled"] is True
    capabilities = enabled_profile["policy"]["capabilities"]
    for name in {
        "filesystem_read",
        "filesystem_write",
        "terminal",
        "messaging",
        "cron",
        "git_push",
        "deploy",
    }:
        assert capabilities[name] is False
    assert enabled_profile["policy"]["capability_groups"] == {
        group: False for group in PROFILE_ROUTER_CAPABILITY_GROUPS
    }
    assert enabled_profile["policy"]["model_policy"] == {
        "allow_model_tools": False,
        "allowed_cost_classes": [COST_CLASS_NO_MODEL],
    }
    assert enabled_profile["policy"]["context"] == {
        "skills": {"read": False},
        "sessions": {"search": False},
    }
    assert all("path" not in profile for profile in result["profiles"])
    assert all("has_env" not in profile for profile in result["profiles"])
    assert all("model" not in profile for profile in result["profiles"])
    assert all("provider" not in profile for profile in result["profiles"])

    all_profiles = json.loads(profiles_list(active_only=False))["profiles"]
    by_ref = {profile["profile_ref"]: profile for profile in all_profiles}
    assert by_ref["local:maker"]["policy"]["enabled"] is False


def test_profile_get_and_health_are_local_only_no_model_wrappers(hermes_home):
    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "display_name": "Main Bot",
                "allowed_roots": [str(hermes_home)],
            },
            "local:missing": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
            },
        },
    )

    one = json.loads(profile_get("local:main-bot"))
    assert one["ok"] is True
    assert one["profile"]["profile_ref"] == "local:main-bot"
    assert one["profile"]["policy"]["enabled"] is True
    assert one["llm_calls"] == 0

    health = json.loads(profile_health("local:main-bot"))
    assert health["ok"] is True
    assert health["profile_ref"] == "local:main-bot"
    assert health["health"]["status"] == "ok"
    assert health["llm_calls"] == 0

    unsupported = json.loads(profile_get("mac:maker"))
    assert unsupported["ok"] is False
    assert unsupported["error"]["code"] == "unsupported_host"
    assert unsupported["llm_calls"] == 0

    disabled = json.loads(profile_get("local:maker"))
    assert disabled["ok"] is False
    assert disabled["error"]["code"] == "profile_not_enabled"
    assert disabled["llm_calls"] == 0

    missing = json.loads(profile_get("local:missing"))
    assert missing["ok"] is False
    assert missing["error"]["code"] == "profile_not_found"
    assert missing["llm_calls"] == 0


def test_phase8_context_skills_are_policy_gated_bounded_and_redacted(
    hermes_home,
    tmp_path,
):
    skill_dir = _write_skill(
        hermes_home,
        "main-bot",
        "software-development/demo-skill",
        """---
name: demo-skill
description: Demo skill for safe router context
tags: [demo, context]
required_environment_variables: [OPENAI_API_KEY]
---
# Demo Skill

Use this safely from /Users/alice/private/project.
OPENAI_API_KEY=sk-tes...7890
""",
        linked_files={
            "references/guide.md": "# Guide\nPASSWORD=\"hunter2\"\nUse bounded context from /home/bob/private/notes.\n",
            "references/binary.bin": b"\x00binary",
            "templates/prompt.txt": "token: should-redact\n",
            "references/large.md": "x" * (MAX_CONTEXT_FILE_BYTES + 1),
        },
    )
    (skill_dir / "references" / ".hidden.md").write_text("hidden", encoding="utf-8")

    _write_router_config(hermes_home)

    denied = {
        "skills_list": json.loads(skills_list("local:main-bot")),
        "skill_view": json.loads(skill_view("local:main-bot", "demo-skill")),
        "session_search": json.loads(session_search("local:main-bot", query="hello")),
    }
    assert denied["skills_list"]["error"]["code"] == "context_skills_read_not_allowed"
    assert denied["skill_view"]["error"]["code"] == "context_skills_read_not_allowed"
    assert denied["session_search"]["error"]["code"] == "context_sessions_search_not_allowed"
    for result in denied.values():
        assert result["ok"] is False
        assert result["llm_calls"] == 0
        assert str(hermes_home) not in json.dumps(result)

    missing = json.loads(skills_list("local:missing"))
    assert missing["ok"] is False
    assert missing["error"]["code"] == "profile_not_found"
    assert missing["llm_calls"] == 0

    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
                "context": {
                    "skills": {"read": True},
                    "sessions": {"search": True},
                },
            }
        },
    )
    policy = load_profile_router_policy()
    route_policy = policy.get_profile_policy(parse_profile_ref("local:main-bot"))
    assert route_policy.allow_context_skills_read is True
    assert route_policy.allow_context_sessions_search is True

    listed = json.loads(skills_list("local:main-bot"))
    listed_dump = json.dumps(listed)
    assert listed["ok"] is True
    assert listed["profile_ref"] == "local:main-bot"
    assert listed["cost_class"] == COST_CLASS_NO_MODEL
    assert listed["llm_calls"] == 0
    assert listed["audit"]["root_exposed"] is False
    assert listed["total_count"] == 1
    assert listed["skills"][0]["id"] == "software-development/demo-skill"
    assert listed["skills"][0]["name"] == "demo-skill"
    assert listed["skills"][0]["category"] == "software-development"
    assert listed["skills"][0]["linked_files"]["references"] == [
        "binary.bin",
        "guide.md",
        "large.md",
    ]
    assert str(hermes_home) not in listed_dump
    assert "skill_dir" not in listed_dump
    assert "OPENAI_API_KEY" not in listed_dump
    assert "hunter2" not in listed_dump

    viewed = json.loads(skill_view("local:main-bot", "demo-skill"))
    viewed_dump = json.dumps(viewed)
    assert viewed["ok"] is True
    assert viewed["skill"]["id"] == "software-development/demo-skill"
    assert viewed["file"]["path"] == "SKILL.md"
    assert "Use this safely" in viewed["file"]["content"]
    assert "sk-tes...7890" not in viewed_dump
    assert "/Users/alice" not in viewed_dump
    assert "private/project" not in viewed_dump
    assert str(hermes_home) not in viewed_dump
    assert "skill_dir" not in viewed_dump
    assert viewed["audit"] == {
        "tool": "skill_view",
        "llm_calls": 0,
        "root_exposed": False,
    }

    support = json.loads(
        skill_view(
            "local:main-bot",
            "software-development/demo-skill",
            file_path="references/guide.md",
        )
    )
    support_dump = json.dumps(support)
    assert support["ok"] is True
    assert support["file"]["path"] == "references/guide.md"
    assert "Use bounded context" in support["file"]["content"]
    assert "hunter2" not in support_dump
    assert "/home/bob" not in support_dump
    assert "private/notes" not in support_dump
    assert str(hermes_home) not in support_dump

    for unsafe_path in (
        "../secret.txt",
        "/etc/passwd",
        "references/../SKILL.md",
        "references/.hidden.md",
        "secrets/token.txt",
    ):
        result = json.loads(
            skill_view("local:main-bot", "demo-skill", file_path=unsafe_path)
        )
        assert result["ok"] is False
        assert result["llm_calls"] == 0
        assert result["error"]["code"] in {
            "invalid_skill_file_path",
            "secret_path_denied",
        }
        assert str(hermes_home) not in json.dumps(result)

    bad_name = json.loads(skill_view("local:main-bot", "software-development/../demo-skill"))
    assert bad_name["ok"] is False
    assert bad_name["error"]["code"] == "invalid_skill_name"
    assert str(hermes_home) not in json.dumps(bad_name)

    binary = json.loads(
        skill_view("local:main-bot", "demo-skill", file_path="references/binary.bin")
    )
    assert binary["ok"] is False
    assert binary["error"]["code"] == "binary_file_not_supported"

    large = json.loads(
        skill_view("local:main-bot", "demo-skill", file_path="references/large.md")
    )
    assert large["ok"] is False
    assert large["error"]["code"] == "file_too_large"

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (skill_dir / "references" / "escape.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pass
    else:
        escaped = json.loads(
            skill_view("local:main-bot", "demo-skill", file_path="references/escape.md")
        )
        assert escaped["ok"] is False
        assert escaped["error"]["code"] == "symlink_traversal_denied"
        assert str(outside) not in json.dumps(escaped)

    outside_skill = tmp_path / "outside-skill"
    outside_skill.mkdir()
    (outside_skill / "SKILL.md").write_text("---\nname: linked-skill\n---\n", encoding="utf-8")
    linked_skill = hermes_home / "profiles" / "main-bot" / "skills" / "linked-skill"
    try:
        linked_skill.symlink_to(outside_skill, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass
    else:
        symlinked = json.loads(skill_view("local:main-bot", "linked-skill"))
        assert symlinked["ok"] is False
        assert symlinked["error"]["code"] == "profile_skill_symlink_denied"

    session = json.loads(session_search("local:main-bot", query="hello"))
    assert session["ok"] is True
    assert session["count"] == 0
    assert session["state_db_present"] is False
    assert session["cost_class"] == COST_CLASS_NO_MODEL
    assert session["llm_calls"] == 0


def test_phase8_session_search_is_policy_gated_read_only_bounded_and_redacted(
    hermes_home,
    tmp_path,
):
    _write_session_db(
        hermes_home,
        "main-bot",
        [
            {
                "id": f"session-alpha-{hermes_home}-TOKEN=supersecret",
                "source": "telegram",
                "model": "openai/test",
                "started_at": 10.0,
                "title": "Deploy debug",
                "messages": [
                    {
                        "role": "user",
                        "content": f"please deploy from {hermes_home} and /Users/alice/private/project with OPENAI_API_KEY=sk-tes...7890",
                        "timestamp": 11.0,
                    },
                    {
                        "role": "tool",
                        "content": "deploy raw tool output should never be returned",
                        "timestamp": 12.0,
                    },
                    {
                        "role": "assistant",
                        "content": "deploy summary PASSWORD=hunter2 and no full dump",
                        "timestamp": 13.0,
                    },
                ],
            },
            {
                "id": "session-beta",
                "source": "cli",
                "model": "anthropic/test",
                "started_at": 20.0,
                "title": "Other work",
                "messages": [
                    {"role": "user", "content": "deploy second match", "timestamp": 21.0},
                    {"role": "assistant", "content": "deploy second answer", "timestamp": 22.0},
                ],
            },
        ],
    )
    _write_session_db(
        hermes_home,
        "maker",
        [
            {
                "id": "maker-session",
                "source": "cli",
                "started_at": 30.0,
                "title": "Wrong profile",
                "messages": [
                    {"role": "user", "content": "deploy maker-only leak", "timestamp": 31.0},
                ],
            }
        ],
    )

    _write_router_config(hermes_home)
    denied = json.loads(session_search("local:main-bot", query="deploy"))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "context_sessions_search_not_allowed"
    assert denied["llm_calls"] == 0
    assert str(hermes_home) not in json.dumps(denied)

    missing = json.loads(session_search("local:missing", query="deploy"))
    assert missing["ok"] is False
    assert missing["error"]["code"] == "profile_not_found"
    assert missing["llm_calls"] == 0

    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
                "context": {"sessions": {"search": True}},
            },
        },
    )
    first_page = json.loads(session_search("local:main-bot", query="deploy", limit=1, sort="newest"))
    first_dump = json.dumps(first_page)
    assert first_page["ok"] is True
    assert first_page["profile_ref"] == "local:main-bot"
    assert first_page["cost_class"] == COST_CLASS_NO_MODEL
    assert first_page["llm_calls"] == 0
    assert first_page["state_db_present"] is True
    assert first_page["query_supplied"] is True
    assert first_page["count"] == 1
    assert first_page["limit"] == 1
    assert first_page["truncated"] is True
    assert first_page["audit"] == {
        "tool": "session_search",
        "llm_calls": 0,
        "root_exposed": False,
        "state_db_read_only": True,
        "roles": ["user", "assistant"],
    }
    assert first_page["results"][0]["role"] in {"user", "assistant"}
    assert first_page["results"][0]["session"]["title"] == "Other work"
    assert "session_id" not in first_page["results"][0]
    assert len(first_page["results"][0]["session_id_hash"]) == 16
    assert len(first_page["results"][0]["snippet"]) <= MAX_SESSION_SNIPPET_CHARS
    assert "content" not in first_page["results"][0]
    assert "messages" not in first_dump
    assert "deploy raw tool output" not in first_dump
    assert "maker-only leak" not in first_dump
    assert "sk-test-secret" not in first_dump
    assert "hunter2" not in first_dump
    assert str(hermes_home) not in first_dump
    assert "TOKEN=supersecret" not in first_dump

    all_results = json.loads(session_search("local:main-bot", query="deploy", limit=5, sort="oldest"))
    all_dump = json.dumps(all_results)
    assert all_results["ok"] is True
    assert all_results["count"] == 4
    assert [item["role"] for item in all_results["results"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "tool" not in {item["role"] for item in all_results["results"]}
    assert {item["session"]["title"] for item in all_results["results"]} == {
        "Deploy debug",
        "Other work",
    }
    assert all(len(item["session_id_hash"]) == 16 for item in all_results["results"])
    assert all("session_id" not in item for item in all_results["results"])
    assert str(hermes_home) not in all_dump
    assert "/Users/alice" not in all_dump
    assert "private/project" not in all_dump
    assert "TOKEN=supersecret" not in all_dump
    assert "OPENAI_API_KEY=***" not in all_dump
    assert "PASSWORD=hunter2" not in all_dump

    recent = json.loads(session_search("local:main-bot", query=None, limit=2))
    assert recent["ok"] is True
    assert recent["query_supplied"] is False
    assert recent["count"] == 2

    invalid = json.loads(session_search("local:main-bot", query="deploy", sort="random"))
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "invalid_session_search_sort"
    assert invalid["llm_calls"] == 0

    outside = tmp_path / "outside-state.db"
    outside.write_text("not sqlite", encoding="utf-8")
    state_db = hermes_home / "profiles" / "main-bot" / "state.db"
    state_db.unlink()
    try:
        state_db.symlink_to(outside)
    except (OSError, NotImplementedError):
        pass
    else:
        symlinked = json.loads(session_search("local:main-bot", query="deploy"))
        assert symlinked["ok"] is False
        assert symlinked["error"]["code"] == "profile_session_db_symlink_denied"
        assert str(outside) not in json.dumps(symlinked)


def test_phase8_session_search_uses_fts_index_when_available(hermes_home):
    db_path = _write_session_db(
        hermes_home,
        "main-bot",
        [
            {
                "id": "fts-session",
                "source": "cli",
                "started_at": 40.0,
                "title": "FTS backed",
                "messages": [
                    {
                        "role": "user",
                        "content": "visible content without the synthetic index marker",
                        "timestamp": 41.0,
                    }
                ],
            }
        ],
    )
    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        except sqlite3.OperationalError as exc:
            pytest.skip(f"sqlite fts5 unavailable: {exc}")
        conn.execute(
            "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
            (1, "ftsmarker-only"),
        )
        conn.commit()
    finally:
        conn.close()

    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
                "context": {"sessions": {"search": True}},
            },
        },
    )

    result = json.loads(session_search("local:main-bot", query="ftsmarker-only"))
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["session"]["title"] == "FTS backed"
    assert "visible content" in result["results"][0]["snippet"]
    assert result["llm_calls"] == 0


def test_phase8_openviking_search_is_policy_gated_local_bounded_and_compatible(
    hermes_home,
    monkeypatch,
):
    _write_router_config(hermes_home)
    denied = json.loads(viking_search("hello SECRET_TOKEN=should-not-log"))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "context_viking_read_not_allowed"
    assert denied["cost_class"] == COST_CLASS_EXTERNAL_API_NO_MODEL
    assert denied["llm_calls"] == 0
    assert "should-not-log" not in json.dumps(denied)

    _write_router_config(
        hermes_home,
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
                "context": {"viking": {"read": True}},
            },
        },
    )
    profile_only = json.loads(viking_search("hello"))
    assert profile_only["ok"] is False
    assert profile_only["error"]["code"] == "context_viking_read_not_allowed"

    _write_router_config(
        hermes_home,
        context={"viking": {"read": True}},
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
            },
        },
    )
    monkeypatch.delenv("OPENVIKING_ENDPOINT", raising=False)
    unconfigured = json.loads(viking_search("hello"))
    assert unconfigured["ok"] is False
    assert unconfigured["error"]["code"] == "openviking_endpoint_unconfigured"

    monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://example.com")
    public_endpoint = json.loads(viking_search("hello"))
    public_dump = json.dumps(public_endpoint)
    assert public_endpoint["ok"] is False
    assert public_endpoint["error"]["code"] == "openviking_endpoint_not_private"
    assert "example.com" not in public_dump

    calls = []

    def fake_request(method, path, *, params=None, payload=None):
        params = params or {}
        payload = payload or {}
        calls.append({"method": method, "path": path, "params": params, "payload": payload})
        assert path == "/api/v1/search/find"
        if "limit" in payload:
            raise ProfileRouterError("openviking_request_failed", "limit not accepted")
        assert payload["top_k"] == 2
        return {
            "result": {
                "memories": [
                    {
                        "uri": "viking://user/hermes/memories/mem_1.md",
                        "score": 0.91,
                        "abstract": "Memory OPENAI_API_KEY=sk-secret-value at /Users/arturo/secrets",
                    }
                ],
                "resources": [
                    {
                        "uri": "viking://resources/.env",
                        "score": 0.99,
                        "abstract": "must be skipped",
                    }
                ],
                "total": 2,
            }
        }

    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933")
    monkeypatch.setattr(mcp_profile_router, "_openviking_request_json", fake_request)
    result = json.loads(
        viking_search(
            "find routing SECRET_TOKEN=should-not-log",
            mode="deep",
            scope="viking://user/hermes",
            limit=2,
        )
    )
    dumped = json.dumps(result)
    assert result["ok"] is True
    assert result["cost_class"] == COST_CLASS_EXTERNAL_API_NO_MODEL
    assert result["llm_calls"] == 0
    assert result["mode"] == "deep"
    assert result["scope"] == "viking://user/hermes"
    assert result["request_shape"] == "top_k_fallback"
    assert result["policy_scope"] == {
        "type": "global",
        "authorized_profiles_count": 1,
        "profile_refs_exposed": False,
    }
    assert result["endpoint"] == {
        "server_configured": True,
        "local_private": True,
        "url_exposed": False,
    }
    assert result["count"] == 1
    assert result["skipped"] >= 1
    assert result["results"][0]["uri"] == "viking://user/hermes/memories/mem_1.md"
    assert result["audit"] == {
        "tool": "viking_search",
        "llm_calls": 0,
        "root_exposed": False,
        "endpoint_local_private": True,
        "raw_query_logged": False,
        "url_exposed": False,
    }
    assert calls[0]["payload"]["limit"] == 2
    assert calls[1]["payload"]["top_k"] == 2
    assert "should-not-log" not in dumped
    assert "sk-secret-value" not in dumped
    assert "/Users/arturo" not in dumped
    assert "127.0.0.1" not in dumped


def test_phase8_openviking_read_validates_uri_and_returns_bounded_redacted_content(
    hermes_home,
    monkeypatch,
):
    _write_router_config(
        hermes_home,
        context={"viking": {"read": True}},
        profiles={
            "local:main-bot": {
                "enabled": True,
                "allowed_roots": [str(hermes_home)],
            },
        },
    )
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
    calls = []

    def fake_request(method, path, *, params=None, payload=None):
        params = params or {}
        payload = payload or {}
        calls.append({"method": method, "path": path, "params": params, "payload": payload})
        if path == "/api/v1/fs/stat":
            return {"result": {"isDir": False}}
        if path == "/api/v1/content/read":
            return {
                "result": {
                    "content": (
                        "TOKEN=supersecret /Users/arturo/root "
                        + "x" * (MAX_VIKING_OVERVIEW_CHARS + 200)
                    )
                }
            }
        raise AssertionError(path)

    monkeypatch.setattr(mcp_profile_router, "_openviking_request_json", fake_request)
    result = json.loads(viking_read("viking://user/hermes/memories/mem_1.md", level="overview"))
    dumped = json.dumps(result)
    assert result["ok"] is True
    assert result["cost_class"] == COST_CLASS_EXTERNAL_API_NO_MODEL
    assert result["llm_calls"] == 0
    assert result["uri"] == "viking://user/hermes/memories/mem_1.md"
    assert result["resolved_uri"] == "viking://user/hermes/memories/mem_1.md"
    assert result["fallback"] == "content/read"
    assert result["truncated"] is True
    assert result["max_chars"] == MAX_VIKING_OVERVIEW_CHARS
    assert len(result["content"]) <= MAX_VIKING_OVERVIEW_CHARS
    assert calls == [
        {
            "method": "GET",
            "path": "/api/v1/fs/stat",
            "params": {"uri": "viking://user/hermes/memories/mem_1.md"},
            "payload": {},
        },
        {
            "method": "GET",
            "path": "/api/v1/content/read",
            "params": {"uri": "viking://user/hermes/memories/mem_1.md"},
            "payload": {},
        },
    ]
    assert "supersecret" not in dumped
    assert "/Users/arturo" not in dumped
    assert "localhost" not in dumped

    call_count = len(calls)
    for bad_uri in (
        "http://example.com/resource",
        "viking://",
        "viking://resources/.env",
        "viking://resources/../secret",
        "viking://resources/%2e%2e/public",
        "viking://resources/%2ehidden/file",
        "viking://resources/foo%2Fbar",
        "viking://resources/Users/arturo",
    ):
        invalid = json.loads(viking_read(bad_uri, level="overview"))
        assert invalid["ok"] is False
        assert invalid["llm_calls"] == 0
    assert len(calls) == call_count


def test_profile_router_mcp_factory_exposes_only_no_model_profile_tools(
    hermes_home,
    monkeypatch,
):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)

    server = mcp_serve.create_profile_router_mcp_server()
    tools = server._tool_manager._tools
    expected_public_tools = {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "skills_list",
        "skill_view",
        "session_search",
        "viking_search",
        "viking_read",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_diff",
    }
    metadata_public_tools = {
        name
        for name, tool in get_router_tool_metadata().items()
        if tool["enabled_by_default"]
    }

    private_action_tools = {"file_patch", "patch_apply", "file_write", "workspace_status_probe", "workspace_scratch_smoke", "file_move", "file_delete", "directory_create", "terminal_run", "workspace_python_run", "process_start", "process_list", "process_poll", "process_log", "process_kill", "git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit", "git_push", "git_checkout", "git_restore", "git_rebase", "git_merge", "github_pr_status", "github_pr_create", "github_pr_update", "github_pr_ready", "github_pr_merge", "github_issue_view", "github_issue_comment", "cron_list", "cron_pause", "cron_resume", "cron_run", "cron_create_script_only", "message_send", "telegram_send", "workspace_production_action_list", "workspace_production_action_status", "workspace_production_action_run", "server_alias_list", "server_status_check", "server_service_logs", "server_docker_ps", "server_docker_logs", "server_port_check", "server_command_run", "server_shell_run", "workspace_web_fetch", "profile_skill_create", "profile_skill_patch", "profile_skill_edit", "profile_skill_write_file", "profile_skill_remove_file", "profile_skill_delete", "profile_memory_add", "profile_memory_replace", "profile_memory_remove", "profile_memory_list"}
    assert set(tools) == expected_public_tools | private_action_tools
    assert expected_public_tools == metadata_public_tools
    assert not (set(tools) & FORBIDDEN_MODEL_LOOP_TOOL_NAMES)
    assert "messages_send" not in tools
    assert "conversations_list" not in tools
    assert "message_send" in tools
    assert "telegram_send" in tools
    assert "terminal_run" in tools
    assert "workspace_diff" in tools
    assert "workspace_file_search" in tools
    assert "workspace_file_stat" in tools
    assert "file_read" not in tools
    assert "file_search" not in tools
    assert "file_patch" in tools
    assert "patch_apply" in tools
    assert "file_write" in tools
    assert "workspace_status_probe" in tools
    assert "workspace_scratch_smoke" in tools
    assert "file_move" in tools
    assert "file_delete" in tools
    assert "directory_create" in tools
    assert "workspace_file_stat" in tools
    assert "skills_list" in tools
    assert "skill_view" in tools
    assert "session_search" in tools
    assert "viking_search" in tools
    assert "viking_read" in tools
    assert "git_status" in tools
    assert "git_diff" in tools
    assert "git_log" in tools
    assert "git_branch" in tools
    assert "cron_list" in tools
    assert "cron_create_script_only" in tools
    assert "cron_run" in tools

    listed = json.loads(tools["profiles_list"].fn())
    assert listed["ok"] is True
    assert listed["cost_class"] == COST_CLASS_NO_MODEL
    assert listed["llm_calls"] == 0


def test_mcp_serve_profile_router_parser_flag_sets_explicit_surface():
    from hermes_cli.subcommands.mcp import build_mcp_parser

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_mcp_parser(sub, cmd_mcp=lambda args: None)

    args = parser.parse_args(["mcp", "serve", "--profile-router"])
    assert args.mcp_action == "serve"
    assert args.profile_router is True
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.public_url is None

    http_args = parser.parse_args(["mcp", "serve", "--profile-router", "--http"])
    assert http_args.profile_router is True
    assert http_args.http is True

    public_args = parser.parse_args([
        "mcp",
        "serve",
        "--profile-router",
        "--http",
        "--public-url",
        "https://mcp.example.com",
    ])
    assert public_args.public_url == "https://mcp.example.com"


def test_mcp_command_routes_profile_router_serve(monkeypatch, capsys):
    mock_run = MagicMock()
    mock_legacy_run = MagicMock()
    monkeypatch.setattr("mcp_serve.run_profile_router_mcp_server", mock_run)
    monkeypatch.setattr("mcp_serve.run_mcp_server", mock_legacy_run)

    from hermes_cli import mcp_config

    monkeypatch.setattr(mcp_config, "load_config", lambda: {})
    mcp_command = mcp_config.mcp_command

    args = argparse.Namespace(mcp_action="serve", verbose=True, profile_router=True)
    mcp_command(args)
    mock_run.assert_called_once_with(
        verbose=True,
        transport="stdio",
        host="127.0.0.1",
        port=8765,
        streamable_http_path="/mcp",
        public_url=None,
    )

    mock_run.reset_mock()
    args = argparse.Namespace(
        mcp_action="serve",
        verbose=False,
        profile_router=True,
        http=True,
        transport="stdio",
        host="127.0.0.1",
        port=9999,
        streamable_http_path="/router",
        public_url="https://mcp.example.com",
    )
    mcp_command(args)
    mock_run.assert_called_once_with(
        verbose=False,
        transport="streamable-http",
        host="127.0.0.1",
        port=9999,
        streamable_http_path="/router",
        public_url="https://mcp.example.com",
    )
    mock_legacy_run.assert_not_called()

    args = argparse.Namespace(
        mcp_action="serve",
        verbose=False,
        profile_router=False,
        http=True,
        transport="stdio",
        host="127.0.0.1",
        port=8765,
        streamable_http_path="/mcp",
        public_url=None,
    )
    mcp_command(args)
    mock_legacy_run.assert_not_called()
    assert "require --profile-router" in capsys.readouterr().out


def test_mcp_command_reads_profile_router_public_url_from_config(monkeypatch):
    import mcp_serve
    from hermes_cli import mcp_config

    mock_run = MagicMock()
    monkeypatch.setattr(mcp_serve, "run_profile_router_mcp_server", mock_run)
    monkeypatch.setattr(
        mcp_config,
        "load_config",
        lambda: {"profile_router": {"public_url": "https://mcp.example.com"}},
    )

    mcp_config.mcp_command(
        argparse.Namespace(mcp_action="serve", profile_router=True, http=True)
    )

    assert mock_run.call_args.kwargs["public_url"] == "https://mcp.example.com"

    mock_run.reset_mock()
    mcp_config.mcp_command(
        argparse.Namespace(
            mcp_action="serve",
            profile_router=True,
            http=True,
            public_url="https://cli-mcp.example.com",
        )
    )
    assert mock_run.call_args.kwargs["public_url"] == "https://cli-mcp.example.com"


def test_profile_router_token_store_hashes_verifies_revokes_and_rotates(tmp_path):
    store = ProfileRouterTokenStore(tmp_path / "tokens.json")
    created = store.create_token(name="chatgpt")
    raw_token = created["token"]
    record = created["record"]

    assert raw_token.startswith("hpr_prt_")
    assert record["token_id"].startswith("prt_")
    assert record["scopes"] == list(DEFAULT_PROFILE_ROUTER_SCOPES)
    assert VIKING_PROFILE_ROUTER_SCOPE not in record["scopes"]
    assert store.verify_token(raw_token, required_scopes=[VIKING_PROFILE_ROUTER_SCOPE]) is None
    viking_token = store.create_token(scopes=[VIKING_PROFILE_ROUTER_SCOPE], name="viking")
    assert viking_token["record"]["scopes"] == [VIKING_PROFILE_ROUTER_SCOPE]
    assert store.verify_token(viking_token["token"], required_scopes=[VIKING_PROFILE_ROUTER_SCOPE]) is not None
    action_token = store.create_token(
        scopes=[
            PROFILE_ROUTER_WRITE_SCOPE,
            PROFILE_ROUTER_TERMINAL_SCOPE,
            PROFILE_ROUTER_CRON_SCOPE,
            PROFILE_ROUTER_MESSAGING_SCOPE,
        ],
        name="action-pilot",
    )
    assert action_token["record"]["scopes"] == [
        PROFILE_ROUTER_WRITE_SCOPE,
        PROFILE_ROUTER_TERMINAL_SCOPE,
        PROFILE_ROUTER_CRON_SCOPE,
        PROFILE_ROUTER_MESSAGING_SCOPE,
    ]
    assert store.verify_token(
        action_token["token"],
        required_scopes=[
            PROFILE_ROUTER_WRITE_SCOPE,
            PROFILE_ROUTER_TERMINAL_SCOPE,
            PROFILE_ROUTER_CRON_SCOPE,
            PROFILE_ROUTER_MESSAGING_SCOPE,
        ],
    ) is not None
    assert raw_token not in (tmp_path / "tokens.json").read_text(encoding="utf-8")
    assert "token_hash_prefix" in record

    verified = store.verify_token(raw_token, required_scopes=["context:read"])
    assert verified is not None
    assert verified.token_id == record["token_id"]
    assert verified.scopes == DEFAULT_PROFILE_ROUTER_SCOPES
    assert store.verify_token(raw_token, required_scopes=["context:read", "diff:read"]) is not None

    listed = store.list_tokens()
    assert listed[0]["last_used_at"] is not None
    assert raw_token not in json.dumps(listed)

    revoked = store.revoke_token(record["token_id"])
    assert revoked["revoked_at"] is not None
    assert store.verify_token(raw_token) is None

    rotated = store.rotate_token(record["token_id"])
    assert rotated["token"] != raw_token
    assert rotated["record"]["token_id"] != record["token_id"]
    assert store.verify_token(rotated["token"]) is not None


def test_profile_router_token_store_lock_serializes_verify_and_revoke(tmp_path):
    token_path = tmp_path / "tokens.json"
    store = ProfileRouterTokenStore(token_path)
    created = store.create_token()
    raw_token = created["token"]
    token_id = created["record"]["token_id"]
    child_code = (
        "import sys\n"
        "from mcp_profile_router_auth import ProfileRouterTokenStore\n"
        "ProfileRouterTokenStore(sys.argv[1]).revoke_token(sys.argv[2])\n"
    )

    with store._locked_store():
        proc = subprocess.Popen(
            [sys.executable, "-c", child_code, str(token_path), token_id],
            cwd=str(__file__).rsplit("/tests/", 1)[0],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.1)
        assert proc.poll() is None
        assert store.verify_token(raw_token) is not None

    stdout, stderr = proc.communicate(timeout=5)
    assert proc.returncode == 0, (stdout, stderr)
    fresh_store = ProfileRouterTokenStore(token_path)
    assert fresh_store.list_tokens()[0]["revoked_at"] is not None
    assert fresh_store.verify_token(raw_token) is None


def test_profile_router_bearer_verifier_and_audit_log_are_secret_safe(tmp_path):
    store = ProfileRouterTokenStore(tmp_path / "tokens.json")
    created = store.create_token(scopes=["context:read"], name="ctx-only")
    verifier = ProfileRouterBearerTokenVerifier(store)

    access_token = asyncio.run(verifier.verify_token(created["token"]))
    assert access_token is not None
    assert access_token.client_id == created["record"]["token_id"]
    assert access_token.scopes == ["context:read"]
    assert access_token.token == created["record"]["token_hash_prefix"]
    assert created["token"] not in json.dumps(access_token.model_dump())

    result = json.dumps(
        {
            "ok": True,
            "llm_calls": 0,
            "file": {"content": "SECRET=should-not-log", "truncated": True},
        }
    )
    audit_fields = extract_result_audit_fields(result)
    assert audit_fields == {
        "ok": True,
        "error": None,
        "llm_calls": 0,
        "bytes": len(result.encode("utf-8")),
        "truncated": True,
    }

    logger = ProfileRouterAuditLogger(tmp_path / "audit.jsonl")
    logger.append(
        {
            "token_id": access_token.client_id,
            "token_hash_prefix": access_token.token,
            "profile": "local:main-bot",
            "workspace_id": "ws_test",
            "tool": "workspace_file_read",
            "scope": "workspace:read",
            **audit_fields,
        }
    )
    audit_line = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert created["token"] not in audit_line
    assert "Authorization" not in audit_line
    assert "should-not-log" not in audit_line
    entry = json.loads(audit_line)
    assert entry["token_id"] == access_token.client_id
    assert entry["tool"] == "workspace_file_read"
    assert entry["llm_calls"] == 0
    assert entry["bytes"] == len(result.encode("utf-8"))


def test_profile_router_public_url_validation_for_remote_metadata(monkeypatch):
    import mcp_serve

    assert mcp_serve._profile_router_http_base_url("0.0.0.0", 8765) == "http://127.0.0.1:8765"
    assert (
        mcp_serve._profile_router_http_base_url(
            "127.0.0.1",
            8765,
            public_url="https://mcp.example.com/",
        )
        == "https://mcp.example.com"
    )

    monkeypatch.setenv("HERMES_PROFILE_ROUTER_PUBLIC_URL", "https://env-mcp.example.com")
    assert mcp_serve._profile_router_http_base_url("127.0.0.1", 8765) == "http://127.0.0.1:8765"

    for invalid_public_url in (
        "mcp.example.com",
        "http://mcp.example.com",
        "ftp://mcp.example.com",
        "https://mcp.example.com/mcp",
        "https://mcp.example.com?token=bad",
    ):
        with pytest.raises(ValueError, match="public URL must be an HTTPS origin"):
            mcp_serve._profile_router_http_base_url(
                "127.0.0.1",
                8765,
                public_url=invalid_public_url,
            )


def test_profile_router_http_factory_uses_bearer_auth_and_exact_public_tools(
    monkeypatch,
    tmp_path,
):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)

    server = mcp_serve.create_profile_router_mcp_server(
        public_http=True,
        http_auth=True,
        host="127.0.0.1",
        port=8765,
        public_url="https://mcp.example.com/",
        token_store_path=str(tmp_path / "tokens.json"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    tools = set(server._tool_manager._tools)
    assert tools == {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "skills_list",
        "skill_view",
        "session_search",
        "viking_search",
        "viking_read",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_diff",
    }
    server_kwargs = getattr(server, "kwargs")
    assert server_kwargs["host"] == "127.0.0.1"
    assert server_kwargs["port"] == 8765
    assert server_kwargs["streamable_http_path"] == "/mcp"
    assert str(server_kwargs["auth"].issuer_url).rstrip("/") == "https://mcp.example.com"
    assert str(server_kwargs["auth"].resource_server_url).rstrip("/") == "https://mcp.example.com/mcp"
    assert server_kwargs["token_verifier"].store.path == tmp_path / "tokens.json"
    assert server_kwargs["auth"].required_scopes == []
    assert {
        "terminal_run",
        "file_patch",
        "cron_run",
        "message_send",
        "git_status",
        "server_shell_run",
    }.isdisjoint(tools)
    assert "workspace_file_stat" in tools
    assert "skills_list" in tools
    assert "skill_view" in tools
    assert "session_search" in tools
    assert "viking_search" in tools
    assert "viking_read" in tools


def test_profile_router_tokens_are_case_sensitive(tmp_path):
    store = ProfileRouterTokenStore(tmp_path / "tokens.json")
    created = store.create_token(scopes=["context:read"], name="ctx-only")

    assert store.verify_token(created["token"]) is not None
    assert store.verify_token(created["token"].upper()) is None


def test_http_owner_mode_tools_are_not_registered(monkeypatch, tmp_path):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)

    server = mcp_serve.create_profile_router_mcp_server(
        public_http=True,
        http_auth=True,
        token_store_path=str(tmp_path / "tokens.json"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    tools = server._tool_manager._tools

    assert {
        "workspace_production_action_run",
        "server_command_run",
        "server_shell_run",
    }.isdisjoint(tools)


def test_public_http_without_auth_still_registers_only_read_tools(monkeypatch):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)

    server = mcp_serve.create_profile_router_mcp_server(public_http=True)
    assert set(server._tool_manager._tools) == {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "skills_list",
        "skill_view",
        "session_search",
        "viking_search",
        "viking_read",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_diff",
    }


def test_profile_router_entrypoint_rejects_native_windows(monkeypatch):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    monkeypatch.setattr(mcp_serve.os, "name", "nt")

    with pytest.raises(RuntimeError, match="requires a POSIX host"):
        mcp_serve.run_profile_router_mcp_server(transport="streamable-http")
