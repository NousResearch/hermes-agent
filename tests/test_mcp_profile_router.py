import argparse
import json
from unittest.mock import MagicMock

import pytest

from mcp_profile_router import (
    COST_CLASS_CALLS_HERMES_AGENT_MODEL,
    COST_CLASS_NO_MODEL,
    PROFILE_ROUTER_TOOL_GROUP,
    ProfileRouterError,
    RouterToolMetadata,
    assert_default_tools_are_no_model,
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
    disabled_power_tools = {"file_patch", "file_write", "terminal_run"}
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
        assert tool["mutates_state"] is True
        assert tool["requires_context"] is True

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

    terminal_disabled = json.loads(
        terminal_run(
            workspace_id,
            "touch SHOULD_NOT_EXIST",
            context_token=fresh_token,
        )
    )
    assert terminal_disabled["ok"] is False
    assert terminal_disabled["error"]["code"] == "tool_disabled"
    assert terminal_disabled["llm_calls"] == 0
    assert not (workspace_root / "SHOULD_NOT_EXIST").exists()


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
