import argparse
import json
import subprocess
from unittest.mock import MagicMock

import pytest

import mcp_profile_router
from mcp_profile_router import (
    COST_CLASS_CALLS_HERMES_AGENT_MODEL,
    COST_CLASS_NO_MODEL,
    MAX_TERMINAL_OUTPUT_CHARS,
    MAX_TERMINAL_TIMEOUT_SECONDS,
    PROFILE_ROUTER_TOOL_GROUP,
    ProfileRouterError,
    RouterToolMetadata,
    _build_terminal_sanitized_env,
    _prepare_terminal_subprocess_plan,
    _shape_terminal_subprocess_result,
    assert_default_tools_are_no_model,
    classify_terminal_command,
    create_workspace_metadata,
    file_patch,
    file_read,
    file_search,
    file_write,
    get_router_tool_metadata,
    load_profile_router_policy,
    parse_profile_ref,
    profile_context_get,
    profile_get,
    profile_health,
    profiles_list,
    require_fresh_workspace_context,
    resolve_workspace_path,
    terminal_run,
    workspace_close,
    workspace_context_status,
    workspace_diff,
    workspace_get,
    workspace_instructions_get,
    workspace_open,
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


def _write_router_config(hermes_home, *, profiles=None, host_roots=None):
    host_roots = host_roots or [str(hermes_home)]
    profiles = profiles or {
        "local:main-bot": {
            "enabled": True,
            "display_name": "Main Bot",
            "allowed_roots": [str(hermes_home)],
        }
    }
    config = {
        "profile_router": {
            "hosts": {
                "local": {
                    "enabled": True,
                    "allowed_roots": host_roots,
                }
            },
            "profiles": profiles,
        }
    }
    (hermes_home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return config


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


def test_router_tool_metadata_is_explicitly_no_model_by_default():
    metadata = get_router_tool_metadata()
    public_tools = {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "file_read",
        "file_search",
    }
    disabled_power_tools = {"file_patch", "file_write", "workspace_diff", "terminal_run"}
    assert set(metadata) == public_tools | disabled_power_tools

    for name in public_tools:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["llm_calls"] == 0
        assert tool["enabled_by_default"] is True
        assert tool["mutates_state"] is False
        assert tool["requires_context"] is False

    for name in disabled_power_tools:
        tool = metadata[name]
        assert tool["cost_class"] == COST_CLASS_NO_MODEL
        assert tool["llm_calls"] == 0
        assert tool["enabled_by_default"] is False
        assert tool["requires_context"] is True
    assert metadata["workspace_diff"]["mutates_state"] is False
    for name in {"file_patch", "file_write", "terminal_run"}:
        assert metadata[name]["mutates_state"] is True

    assert_default_tools_are_no_model()


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
    assert route_policy.allow_filesystem_read is False
    assert route_policy.allow_filesystem_write is False
    assert route_policy.allow_terminal is False
    assert route_policy.allow_messaging is False
    assert route_policy.allow_cron is False
    assert route_policy.allow_git_push is False
    assert route_policy.allow_deploy is False
    assert route_policy.allow_model_tools is False
    assert route_policy.allowed_cost_classes == (COST_CLASS_NO_MODEL,)
    assert route_policy.terminal_execution_policy.enabled is False
    assert route_policy.terminal_execution_policy.allowed_commands == ()
    assert route_policy.terminal_execution_policy.allowed_command_prefixes == ()
    assert route_policy.terminal_execution_policy.require_no_shell is True


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

    read_result = json.loads(file_read(workspace["workspace_id"], "notes.md", offset=2, limit=1))
    assert read_result["ok"] is True
    assert read_result["llm_calls"] == 0
    assert read_result["file"]["content"] == "beta\n"
    assert read_result["file"]["truncated"] is True

    secret = json.loads(file_read(workspace["workspace_id"], ".env.local"))
    assert secret["ok"] is False
    assert secret["error"]["code"] == "secret_path_denied"
    assert secret["llm_calls"] == 0

    binary = json.loads(file_read(workspace["workspace_id"], "binary.bin"))
    assert binary["ok"] is False
    assert binary["error"]["code"] == "binary_file_not_supported"
    assert binary["llm_calls"] == 0

    search_result = json.loads(
        file_search(workspace["workspace_id"], "alpha", file_glob="*.md")
    )
    assert search_result["ok"] is True
    assert search_result["llm_calls"] == 0
    assert [match["line"] for match in search_result["search"]["matches"]] == [1, 3]
    assert search_result["search"]["skipped"]["binary"] == 0

    files_only = json.loads(
        file_search(
            workspace["workspace_id"],
            "alpha",
            file_glob="*.md",
            output_mode="files_only",
        )
    )
    assert files_only["ok"] is True
    assert files_only["search"]["files"] == ["notes.md"]
    assert files_only["llm_calls"] == 0

    missing_workspace = json.loads(file_read("ws_missing", "notes.md"))
    assert missing_workspace["ok"] is False
    assert missing_workspace["error"]["code"] == "workspace_not_found"
    assert missing_workspace["llm_calls"] == 0


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

    write_result = json.loads(
        file_write(workspace_id, "created.txt", "created\n", context_token=fresh_token)
    )
    assert write_result["ok"] is True
    assert write_result["llm_calls"] == 0
    assert write_result["write"]["path"] == "created.txt"
    assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "created\n"

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


def test_terminal_run_reports_allowlist_policy_without_executing(
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
    assert allowed["ok"] is False
    assert allowed["error"]["code"] == "tool_disabled"
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
    assert plan["implementation_status"] == "pending_no_shell_subprocess_executor"
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
    assert readiness["current_phase_allows_subprocess_run"] is False
    assert readiness["subprocess_run_allowed"] is False
    assert readiness["real_executor_status"] == "blocked_pending_auth_public_exposure_review"
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
    assert str(workspace_root) not in json.dumps(allowed)
    assert "pwd" not in json.dumps(allowed)

    prefix_allowed = json.loads(
        terminal_run(workspace_id, "git status --short", context_token=token)
    )
    assert prefix_allowed["ok"] is False
    assert prefix_allowed["error"]["code"] == "tool_disabled"
    assert prefix_allowed["terminal_command"]["execution_policy"][
        "allowlist_match_type"
    ] == "prefix"
    prefix_plan = prefix_allowed["terminal_command"]["execution_plan"]
    assert prefix_plan["available"] is True
    assert prefix_plan["argv"]["argc"] == 3
    assert prefix_plan["argv"]["argument_count"] == 2
    assert prefix_plan["argv"]["option_count"] == 1
    assert "git status --short" not in json.dumps(prefix_allowed)

    unlisted = json.loads(
        terminal_run(workspace_id, "python -m pytest", context_token=token)
    )
    assert unlisted["ok"] is False
    assert unlisted["error"]["code"] == "terminal_command_blocked"
    assert unlisted["terminal_command"]["execution_plan"]["available"] is False
    assert unlisted["terminal_command"]["execution_plan"]["argv"] is None
    assert "terminal_command_not_allowlisted" in {
        reason["code"] for reason in unlisted["terminal_command"]["reasons"]
    }

    shell_control = json.loads(
        terminal_run(workspace_id, "pwd && git status", context_token=token)
    )
    assert shell_control["ok"] is False
    assert shell_control["error"]["code"] == "terminal_command_blocked"
    assert "terminal_shell_control_not_allowed" in {
        reason["code"] for reason in shell_control["terminal_command"]["reasons"]
    }


def test_terminal_executor_boundary_is_non_executing_and_not_public(
    hermes_home,
    monkeypatch,
    tmp_path,
):
    subprocess_calls = []

    def _forbidden_subprocess_run(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        raise AssertionError("terminal_run must not call subprocess.run yet")

    monkeypatch.setattr(mcp_profile_router.subprocess, "run", _forbidden_subprocess_run)
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
    assert direct["ok"] is False
    assert direct["error"]["code"] == "tool_disabled"
    assert subprocess_calls == []

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
    assert readiness["current_phase_allows_subprocess_run"] is False
    assert readiness["subprocess_run_allowed"] is False
    assert readiness["checks"]["executor_boundary_non_executing"] is True
    assert readiness["checks"]["tool_metadata_no_model"] is True
    assert readiness["checks"]["public_mcp_absent_by_default"] is True
    assert readiness["checks"]["fresh_context_validated_upstream"] is True
    assert readiness["failed_checks"] == []

    metadata = get_router_tool_metadata()["terminal_run"]
    assert metadata["enabled_by_default"] is False
    assert metadata["requires_context"] is True
    assert metadata["cost_class"] == COST_CLASS_NO_MODEL
    assert metadata["llm_calls"] == 0

    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)
    server = mcp_serve.create_profile_router_mcp_server()
    assert "terminal_run" not in server._tool_manager._tools
    assert "workspace_diff" not in server._tool_manager._tools
    dumped = json.dumps(direct)
    assert "pwd" not in dumped
    assert str(workspace_root) not in dumped
    assert "/usr/bin" not in dumped


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
    opened = json.loads(workspace_open("local:main-bot", str(workspace_root)))
    workspace_id = opened["workspace"]["workspace_id"]
    token = json.loads(workspace_instructions_get(workspace_id))["context"]["context_token"]

    denied = json.loads(file_write(workspace_id, "notes.md", "beta\n", context_token=token))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "filesystem_write_not_allowed"
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

    _git(workspace_root, "init")
    _git(workspace_root, "config", "user.email", "router-test@example.invalid")
    _git(workspace_root, "config", "user.name", "Router Test")
    _git(workspace_root, "add", "AGENTS.md", "notes.md", "funciones.txt", ".hermes/plans/state.json")
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
    (workspace_root / "new.txt").write_text("new file\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=should-not-leak\n", encoding="utf-8")

    result = json.loads(workspace_diff(workspace_id, context_token=token))
    assert result["ok"] is True
    assert result["llm_calls"] == 0
    diff = result["workspace_diff"]
    assert diff["tracked_files"] == ["notes.md"]
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
        "public_mcp_exposure": "disabled_pending_http_auth_config_review",
    }
    assert "-alpha" in diff["diff"]["unified"]
    assert "+beta" in diff["diff"]["unified"]
    dumped = json.dumps(result)
    assert str(workspace_root) not in dumped
    assert "private deployment" not in dumped
    assert "should-not-leak" not in dumped


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
    assert enabled_profile["policy"]["capabilities"] == {
        "filesystem_read": False,
        "filesystem_write": False,
        "terminal": False,
        "messaging": False,
        "cron": False,
        "git_push": False,
        "deploy": False,
    }
    assert enabled_profile["policy"]["model_policy"] == {
        "allow_model_tools": False,
        "allowed_cost_classes": [COST_CLASS_NO_MODEL],
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


def test_profile_router_mcp_factory_exposes_only_no_model_profile_tools(
    hermes_home,
    monkeypatch,
):
    import mcp_serve

    monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", True)
    monkeypatch.setattr(mcp_serve, "FastMCP", _FakeFastMCP)

    server = mcp_serve.create_profile_router_mcp_server()
    tools = server._tool_manager._tools

    assert set(tools) == {
        "profiles_list",
        "profile_get",
        "profile_health",
        "profile_context_get",
        "workspace_open",
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "file_read",
        "file_search",
    }
    assert "messages_send" not in tools
    assert "conversations_list" not in tools
    assert "terminal_run" not in tools
    assert "workspace_diff" not in tools
    assert "file_write" not in tools

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


def test_mcp_command_routes_profile_router_serve(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("mcp_serve.run_profile_router_mcp_server", mock_run)

    from hermes_cli.mcp_config import mcp_command

    args = argparse.Namespace(mcp_action="serve", verbose=True, profile_router=True)
    mcp_command(args)
    mock_run.assert_called_once_with(verbose=True)
