from __future__ import annotations

import importlib.util
import json
import subprocess
from types import SimpleNamespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mcp" / "quinn_ops_server.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quinn_ops_server_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_and_tool_functions_json_serializable():
    mod = load_module()
    expected = {
        "get_overview",
        "get_snapshot_status",
        "get_overview_diff",
        "save_overview_snapshot",
        "get_gateway_status",
        "get_platform_status",
        "get_mcp_status",
        "get_toolsets_status",
        "get_cron_status",
        "get_sessions_summary",
        "get_recent_errors",
        "get_config_summary",
        "get_repo_status",
        "get_runtime_files_status",
        "healthcheck",
    }
    assert expected <= set(mod.TOOL_FUNCTIONS)
    for name in expected - {"get_overview"}:
        result = mod.TOOL_FUNCTIONS[name]()
        assert isinstance(result, dict)
        json.dumps(result)
        assert set(result) == {"ok", "data", "errors", "warnings"}


def test_redaction_representative_secrets():
    mod = load_module()
    payload = {
        "api_key": "sk-test123456789",
        "github": "ghp_fakeSECRET123456",
        "pat": "github_pat_fakeSECRET123456",
        "header": "Authorization: Bearer abc123SECRET",
        "discord": "DISCORD_TOKEN=abc.def.ghi01234567890123456789",
        "telegram": "TELEGRAM_BOT_TOKEN=123:ABCSECRETXYZ",
        "nested": [{"token": "should-not-appear"}],
    }
    redacted = json.dumps(mod.sanitize(payload))
    assert "sk-test" not in redacted
    assert "ghp_fake" not in redacted
    assert "github_pat_fake" not in redacted
    assert "abc123SECRET" not in redacted
    assert "abc.def" not in redacted
    assert "ABCSECRET" not in redacted
    assert "should-not-appear" not in redacted


def test_safe_metadata_keys_are_not_redacted():
    mod = load_module()
    payload = {
        "auth_file": {"exists": True, "metadata_only": True},
        "sessions": {"count": 5},
        "session_count": 5,
        "sessions_dir_exists": True,
        "mcp_servers": ["quinn_ops"],
        "session_metadata": [{"platform": "telegram", "chat_type": "dm"}],
        "platform_status": {"telegram": {"configured": True}},
        "access_token": "abc123SECRET",
        "headers": {"Authorization": "Bearer abc123SECRET"},
    }
    sanitized = mod.sanitize(payload)
    dumped = json.dumps(sanitized)
    assert sanitized["auth_file"]["exists"] is True
    assert sanitized["sessions"]["count"] == 5
    assert sanitized["session_count"] == 5
    assert sanitized["sessions_dir_exists"] is True
    assert sanitized["mcp_servers"] == ["quinn_ops"]
    assert sanitized["session_metadata"][0]["platform"] == "telegram"
    assert sanitized["platform_status"]["telegram"]["configured"] is True
    assert "abc123SECRET" not in dumped


def test_config_summary_redacts_sensitive_keys(tmp_path, monkeypatch):
    mod = load_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  provider: openai
  api_key: sk-test123456789
gateway:
  discord:
    token: abc.def.ghi01234567890123456789
  telegram:
    bot_token: 123:ABCSECRETXYZ
mcp_servers:
  sample:
    command: python
tools:
  web:
    enabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "CONFIG_PATH", config)
    monkeypatch.setattr(mod, "AUTH_PATH", tmp_path / "auth.json")
    result = mod.get_config_summary()
    dumped = json.dumps(result)
    assert result["ok"]
    assert result["data"]["model"]["provider_present"] is True
    assert result["data"]["mcp_servers"] == ["sample"]
    assert result["data"]["platforms"]["configured_sections"] == ["discord", "telegram"]
    assert "discord_configured" not in result["data"]["platforms"]
    assert "sk-test" not in dumped
    assert "abc.def" not in dumped
    assert "ABCSECRET" not in dumped


def test_subprocess_failure_structured(monkeypatch):
    mod = load_module()

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["hermes"], timeout=1)

    monkeypatch.setattr(mod.subprocess, "run", boom)
    result = mod.get_gateway_status()
    assert result["ok"] is False
    assert result["errors"]
    assert result["errors"][0]["kind"] == "timeout"
    json.dumps(result)


def test_gateway_status_does_not_return_raw_multiline_service_text(monkeypatch):
    mod = load_module()

    def fake_run_cmd(argv, source, timeout=10, cwd=None):
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return {"stdout": "active", "stderr": "", "returncode": 0}, None
        if argv[:3] == ["systemctl", "--user", "show"]:
            return {
                "stdout": "MainPID=123\nActiveState=active\nSubState=running\nLoadState=loaded\nUnitFileState=enabled\n",
                "stderr": "",
                "returncode": 0,
            }, None
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)
    result = mod.get_gateway_status()
    assert result["ok"]
    data = result["data"]
    assert data["pid"] == "123"
    assert data["booleans"]["running"] is True
    assert "hermes_gateway_status" not in data
    assert "summary" not in json.dumps(data).lower()


def test_recent_errors_limit_and_redaction(tmp_path, monkeypatch):
    mod = load_module()
    log = tmp_path / "gateway.log"
    log.write_text("ok\n2026-05-13T04:00:00Z ERROR Authorization: Bearer abc123SECRET user: private text\nWARN sk-test123456789\n", encoding="utf-8")
    monkeypatch.setattr(mod, "LOG_PATHS", [log])
    result = mod.get_recent_errors(500)
    dumped = json.dumps(result)
    assert result["data"]["limit"] == 200
    assert result["data"]["matched_total"] == 2
    assert result["data"]["grouped"]["gateway.log"]["categories"]["error"] == 1
    assert result["data"]["snippets"] == []
    assert "abc123SECRET" not in dumped
    assert "sk-test" not in dumped
    assert "private text" not in dumped


def test_recent_errors_snippets_are_env_gated(tmp_path, monkeypatch):
    mod = load_module()
    log = tmp_path / "gateway.log"
    log.write_text("ERROR safe operational failure\nERROR user: do not return this\n", encoding="utf-8")
    monkeypatch.setattr(mod, "LOG_PATHS", [log])
    no_snippets = mod.get_recent_errors(50, include_snippets=True)
    assert no_snippets["data"]["snippets"] == []
    monkeypatch.setenv("QUINN_OPS_ALLOW_LOG_SNIPPETS", "1")
    snippets = mod.get_recent_errors(50, include_snippets=True)
    dumped = json.dumps(snippets)
    assert "safe operational failure" in dumped
    assert "do not return this" not in dumped


def test_git_porcelain_parsing_preserves_first_filename(monkeypatch):
    mod = load_module()

    def fake_run_cmd(argv, source, timeout=10, cwd=None):
        if argv[:3] == ["git", "status", "--porcelain=v1"]:
            return {"stdout": " M gateway/platforms/discord.py\n?? docs/foo.md\n", "stderr": "", "returncode": 0}, None
        if argv[:2] == ["git", "rev-parse"]:
            return {"stdout": "abcdef1234567890\n", "stderr": "", "returncode": 0}, None
        if argv[:2] == ["git", "describe"]:
            return {"stdout": "v1-dirty\n", "stderr": "", "returncode": 0}, None
        if argv[:2] == ["git", "branch"]:
            return {"stdout": "main\n", "stderr": "", "returncode": 0}, None
        raise AssertionError(argv)

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)
    result = mod.get_repo_status()
    assert result["data"]["status_short"]["files"][0] == "gateway/platforms/discord.py"
    assert result["data"]["status_short"]["files"][1] == "docs/foo.md"


def test_platform_status_parses_messaging_section(monkeypatch):
    mod = load_module()
    sample = """
Hermes status

Messaging Platforms:
  Discord: connected
  Telegram: configured, connection unknown
  Slack: not configured
  Matrix: disabled
  WhatsApp: not configured

Other:
  Discord mention elsewhere should not matter
"""

    def fake_run_cmd(argv, source, timeout=10, cwd=None):
        return {"stdout": sample, "stderr": "", "returncode": 0}, None

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(mod, "get_config_summary", lambda: {"ok": True, "data": {"platforms": {}}, "errors": [], "warnings": []})
    result = mod.get_platform_status()
    platforms = result["data"]["platform_status"]
    assert platforms["discord"]["connected"] is True
    assert platforms["telegram"]["configured"] is True
    assert platforms["telegram"]["connected"] is None
    assert platforms["slack"]["configured"] is False
    assert platforms["whatsapp"]["configured"] is False
    assert platforms["matrix"]["status"] == "not_configured"


def test_cron_status_no_scheduled_jobs(monkeypatch):
    mod = load_module()

    def fake_run_cmd(argv, source, timeout=10, cwd=None):
        return {"stdout": "No scheduled jobs.\nCreate one with `hermes cron add`.\n", "stderr": "", "returncode": 0}, None

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)
    result = mod.get_cron_status()
    assert result["ok"]
    assert result["data"] == {"active": 0, "total": 0, "jobs": []}


def test_platform_status_realish_connected_configured(monkeypatch):
    mod = load_module()
    sample = """
Status:
  Model: gpt-5.5

Messaging Platforms:
  discord configured connected
  telegram configured connected
  slack not configured
  whatsapp not configured
  signal not configured
"""

    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: ({"stdout": sample, "stderr": "", "returncode": 0}, None))
    monkeypatch.setattr(mod, "get_config_summary", lambda: {"ok": True, "data": {"platforms": {"configured_sections": []}}, "errors": [], "warnings": []})
    result = mod.get_platform_status()
    platforms = result["data"]["platform_status"]
    assert platforms["discord"]["configured"] is True
    assert platforms["discord"]["connected"] is True
    assert platforms["telegram"]["configured"] is True
    assert platforms["telegram"]["connected"] is True
    assert platforms["slack"]["configured"] is False
    assert platforms["whatsapp"]["configured"] is False
    assert platforms["signal"]["configured"] is False


def test_platform_status_live_configured_unknown_connection(monkeypatch):
    mod = load_module()
    sample = """
◆ Messaging Platforms
  Telegram      ✓ configured (home: 8230754176)
  Discord       ✓ configured (home: 1498774848493584394)
  WhatsApp      ✗ not configured
"""

    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: ({"stdout": sample, "stderr": "", "returncode": 0}, None))
    monkeypatch.setattr(mod, "get_config_summary", lambda: {"ok": True, "data": {"platforms": {"configured_sections": []}}, "errors": [], "warnings": []})
    result = mod.get_platform_status()
    platforms = result["data"]["platform_status"]
    assert platforms["telegram"]["configured"] is True
    assert platforms["telegram"]["connected"] is None
    assert platforms["telegram"]["status"] == "configured"
    assert platforms["discord"]["configured"] is True
    assert platforms["discord"]["connected"] is None
    assert platforms["discord"]["status"] == "configured"
    assert platforms["whatsapp"]["configured"] is False
    assert platforms["whatsapp"]["connected"] is False
    assert platforms["whatsapp"]["status"] == "not_configured"


def test_platform_status_adds_passive_delivery_probe_without_history_or_send(monkeypatch):
    mod = load_module()
    sample = """
Messaging Platforms:
  Discord: connected
  Telegram: configured, connection unknown
  Slack: not configured
  Signal: configured, not connected
"""

    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: ({"stdout": sample, "stderr": "", "returncode": 0}, None))
    monkeypatch.setattr(mod, "get_config_summary", lambda: {"ok": True, "data": {"platforms": {"configured_sections": []}}, "errors": [], "warnings": []})
    result = mod.get_platform_status()
    platforms = result["data"]["platform_status"]

    assert platforms["discord"]["delivery_capable"] is True
    assert platforms["discord"]["delivery_status"] == "delivery_capable"
    assert platforms["telegram"]["delivery_capable"] is None
    assert platforms["telegram"]["delivery_status"] == "configured_delivery_unknown"
    assert platforms["signal"]["delivery_capable"] is False
    assert platforms["signal"]["delivery_status"] == "not_connected"
    assert platforms["slack"]["delivery_capable"] is False
    assert platforms["slack"]["delivery_status"] == "not_configured"
    assert all(item["probe_method"] == "passive_status_parse" for item in platforms.values())
    assert all(item["history_read"] is False for item in platforms.values())
    assert all(item["delivery_attempted"] is False for item in platforms.values())


def test_overview_aggregates_partial_errors(monkeypatch):
    mod = load_module()

    def ok_data(name):
        return {"ok": True, "data": {name: True}, "errors": [], "warnings": []}

    monkeypatch.setattr(mod, "get_recent_errors", lambda limit=50, include_snippets=False: {"ok": True, "data": {"matched_total": 0, "grouped": {}}, "errors": [], "warnings": []})
    monkeypatch.setattr(mod, "get_gateway_status", lambda: {"ok": False, "data": {"partial": True}, "errors": [{"source": "gateway", "message": "bad", "kind": "test"}], "warnings": ["gateway warning"]})
    monkeypatch.setattr(mod, "get_platform_status", lambda: ok_data("platforms"))
    monkeypatch.setattr(mod, "get_mcp_status", lambda: ok_data("mcp"))
    monkeypatch.setattr(mod, "get_cron_status", lambda: ok_data("cron"))
    monkeypatch.setattr(mod, "get_sessions_summary", lambda: ok_data("sessions"))
    monkeypatch.setattr(mod, "get_toolsets_status", lambda: ok_data("toolsets"))
    monkeypatch.setattr(mod, "get_repo_status", lambda: ok_data("repo"))
    monkeypatch.setattr(mod, "get_runtime_files_status", lambda: ok_data("runtime"))
    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: ({"stdout": "1.0", "stderr": "", "returncode": 0}, None))
    result = mod.get_overview()
    assert result["ok"] is False
    assert result["data"]["gateway"]["partial"] is True
    assert result["errors"][0]["source"] == "gateway"
    assert "gateway warning" in result["warnings"]


def overview_payload(**overrides):
    data = {
        "timestamp_utc": "2026-05-13T00:00:00Z",
        "hermes_version": "hermes 1.0",
        "gateway": {
            "systemd_active": "active",
            "pid": "100",
            "systemd_properties": {
                "ActiveState": "active",
                "SubState": "running",
                "UnitFileState": "enabled",
            },
        },
        "platforms": {
            "platform_status": {
                "discord": {"configured": True, "connected": True, "status": "connected"},
                "telegram": {"configured": True, "connected": None, "status": "configured"},
            }
        },
        "mcp": {
            "configured_servers": ["quinn_ops"],
            "mcp_list": {"active_like": 1, "total": 1},
        },
        "cron": {"active": 0, "total": 0, "jobs": []},
        "sessions": {
            "count": 1,
            "file_count": 2,
            "json_file_count": 1,
            "recent_metadata": [{"platform": "discord", "chat_type": "group", "updated_at": "ignored"}],
        },
        "repo": {
            "head": "abc123",
            "branch": "main",
            "describe": "abc123",
            "status_short": {"dirty_count": 0, "files": []},
        },
        "recent_errors": {
            "count": 1,
            "grouped": {"gateway.log": {"total": 1, "categories": {"error": 1}, "last_seen": "ignored"}},
        },
        "runtime_files": {
            "files": [
                {"path": "/home/quinn/quinn/runtime/quinn_loader_order.json", "exists": True, "size_bytes": 10, "mtime_utc": "ignored"}
            ]
        },
        "toolsets": {
            "toolsets": ["web enabled"],
            "summary": {"total": 1, "active_like": 1},
        },
    }
    for key, value in overrides.items():
        data[key] = value
    return {"ok": True, "data": data, "errors": [], "warnings": []}


def use_snapshot_path(mod, tmp_path, monkeypatch):
    snapshot = tmp_path / "mcp" / "quinn_ops_state" / "overview_snapshot.json"
    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot)
    return snapshot


def save_snapshot_for(mod, snapshot_path, overview):
    payload = {
        "schema_version": mod.SNAPSHOT_SCHEMA_VERSION,
        "created_by": "quinn_ops",
        "updated_at_utc": "2026-05-13T00:00:00Z",
        "overview": mod.sanitize(overview["data"]),
        "overview_errors": [],
        "overview_warnings": [],
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")


def diff_paths(result, group):
    return {item["path"]: item for item in result["data"]["changes"][group]}


def test_snapshot_status_missing(tmp_path, monkeypatch):
    mod = load_module()
    use_snapshot_path(mod, tmp_path, monkeypatch)
    result = mod.get_snapshot_status()
    assert result["ok"]
    assert result["data"]["exists"] is False
    assert result["data"]["schema_version"] is None


def test_save_snapshot_creates_sanitized_schema_versioned_file(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    private = overview_payload(
        gateway={"systemd_active": "active", "pid": "sk-test123456789", "systemd_properties": {"ActiveState": "active"}}
    )
    monkeypatch.setattr(mod, "get_overview", lambda: private)
    result = mod.save_overview_snapshot()
    dumped = snapshot.read_text(encoding="utf-8")
    obj = json.loads(dumped)
    assert result["ok"]
    assert obj["schema_version"] == 1
    assert obj["created_by"] == "quinn_ops"
    assert "sk-test" not in dumped
    assert snapshot.stat().st_mode & 0o777 == 0o600


def test_diff_without_update_baseline_does_not_write(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    before = snapshot.read_text(encoding="utf-8")
    current = overview_payload(gateway={"systemd_active": "active", "pid": "200", "systemd_properties": {"ActiveState": "active", "SubState": "running", "UnitFileState": "enabled"}})
    monkeypatch.setattr(mod, "get_overview", lambda: current)
    result = mod.get_overview_diff(update_baseline=False)
    assert result["ok"]
    assert result["data"]["updated_baseline"] is False
    assert snapshot.read_text(encoding="utf-8") == before


def test_first_diff_no_baseline_behavior(tmp_path, monkeypatch):
    mod = load_module()
    use_snapshot_path(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload())
    result = mod.get_overview_diff()
    assert result["ok"]
    assert result["data"]["has_previous"] is False
    assert result["data"]["summary"]["changed_count"] == 0
    assert all(not changes for changes in result["data"]["changes"].values())
    assert result["warnings"]
    assert "No previous" in result["data"]["summary"]["headlines"][0]


def test_gateway_pid_change_info(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current = overview_payload(gateway={"systemd_active": "active", "pid": "200", "systemd_properties": {"ActiveState": "active", "SubState": "running", "UnitFileState": "enabled"}})
    monkeypatch.setattr(mod, "get_overview", lambda: current)
    result = mod.get_overview_diff()
    change = diff_paths(result, "gateway")["gateway.pid"]
    assert change["before"] == "100"
    assert change["after"] == "200"
    assert change["severity"] == "info"


def test_gateway_active_to_inactive_critical(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current = overview_payload(gateway={"systemd_active": "inactive", "pid": "0", "systemd_properties": {"ActiveState": "inactive", "SubState": "dead", "UnitFileState": "enabled"}})
    monkeypatch.setattr(mod, "get_overview", lambda: current)
    result = mod.get_overview_diff()
    assert result["data"]["summary"]["severity"] == "critical"
    assert diff_paths(result, "gateway")["gateway.systemd_active"]["severity"] == "critical"


def test_platform_configured_false_warning_and_connected_false_critical(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current_platforms = {
        "platform_status": {
            "discord": {"configured": False, "connected": False, "status": "not_configured"},
            "telegram": {"configured": True, "connected": False, "status": "not_connected"},
        }
    }
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(platforms=current_platforms))
    result = mod.get_overview_diff()
    changes = diff_paths(result, "platforms")
    assert changes["platforms.discord.configured"]["severity"] == "warning"
    assert changes["platforms.discord.connected"]["severity"] == "critical"


def test_platform_configured_to_unknown_warning(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current_platforms = {
        "platform_status": {
            "discord": {"configured": None, "connected": None, "status": "unknown"},
            "telegram": {"configured": True, "connected": None, "status": "configured"},
        }
    }
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(platforms=current_platforms))
    result = mod.get_overview_diff()
    changes = diff_paths(result, "platforms")
    assert changes["platforms.discord.configured"]["severity"] == "warning"
    assert changes["platforms.discord.connected"]["severity"] == "warning"


def test_cron_count_changes(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current = overview_payload(cron={"active": 1, "total": 2, "jobs": [{"id_or_name": "job-1"}, {"id_or_name": "job-2"}]})
    monkeypatch.setattr(mod, "get_overview", lambda: current)
    result = mod.get_overview_diff()
    changes = diff_paths(result, "cron")
    assert changes["cron.total"]["after"] == 2
    assert changes["cron.active"]["after"] == 1
    assert "cron.jobs" in changes


def test_repo_dirty_and_file_list_changes(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current_repo = {
        "head": "abc123",
        "branch": "main",
        "describe": "abc123-dirty",
        "status_short": {"dirty_count": 2, "files": ["scripts/mcp/quinn_ops_server.py", "docs/foo.md"]},
    }
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(repo=current_repo))
    result = mod.get_overview_diff()
    changes = diff_paths(result, "repo")
    assert changes["repo.status_short.dirty_count"]["type"] == "increased"
    assert "scripts/mcp/quinn_ops_server.py" in changes["repo.status_short.files.added"]["after"]


def test_mcp_quinn_ops_configured_but_inactive_is_critical(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current_mcp = {"configured_servers": ["quinn_ops"], "mcp_list": {"active_like": 0, "total": 1}}
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(mcp=current_mcp))
    result = mod.get_overview_diff()
    changes = diff_paths(result, "mcp")
    assert changes["mcp.mcp_list.active_like"]["severity"] == "critical"


def test_recent_error_count_category_increase_without_snippets(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    save_snapshot_for(mod, snapshot, overview_payload())
    current_errors = {
        "count": 4,
        "grouped": {
            "gateway.log": {
                "total": 4,
                "categories": {"error": 2, "exception": 2},
                "last_seen": "2026-05-13T00:01:00Z",
                "snippets": ["private message should not matter"],
            }
        },
    }
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(recent_errors=current_errors))
    result = mod.get_overview_diff()
    dumped = json.dumps(result)
    changes = diff_paths(result, "recent_errors")
    assert changes["recent_errors.count"]["type"] == "increased"
    assert changes["recent_errors.grouped.gateway.log.categories.exception"]["severity"] == "warning"
    assert "private message should not matter" not in dumped


def test_recent_error_delta_summary_tracks_counts_repeats_and_last_seen_without_snippets(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    previous_errors = {
        "count": 3,
        "grouped": {
            "gateway.log": {
                "total": 3,
                "categories": {"error": 2, "warning": 1},
                "last_seen": "2026-05-13T00:01:00Z",
                "snippets": ["previous private text must stay out"],
            }
        },
    }
    save_snapshot_for(mod, snapshot, overview_payload(recent_errors=previous_errors))
    current_errors = {
        "count": 6,
        "grouped": {
            "gateway.log": {
                "total": 5,
                "categories": {"error": 3, "warning": 1, "traceback": 1},
                "last_seen": "2026-05-13T00:03:00Z",
                "snippets": ["current private text must stay out"],
            },
            "agent.log": {
                "total": 1,
                "categories": {"exception": 1},
                "last_seen": "2026-05-13T00:02:00Z",
                "snippets": ["agent private text must stay out"],
            },
        },
    }
    monkeypatch.setattr(mod, "get_overview", lambda: overview_payload(recent_errors=current_errors))
    result = mod.get_overview_diff()
    dumped = json.dumps(result)
    summary = result["data"]["summary"]["error_delta"]
    changes = diff_paths(result, "recent_errors")
    assert summary["total_before"] == 3
    assert summary["total_after"] == 6
    assert summary["total_delta"] == 3
    assert summary["sources"]["gateway.log"]["categories"]["error"]["delta"] == 1
    assert summary["sources"]["gateway.log"]["categories"]["traceback"]["is_new"] is True
    assert summary["sources"]["agent.log"]["is_new_source"] is True
    assert {item["category"] for item in summary["new_categories"]} == {"traceback", "exception"}
    assert {item["category"] for item in summary["repeated_categories"]} >= {"error", "warning"}
    assert summary["last_seen_moved"][0]["source"] == "gateway.log"
    assert changes["recent_errors.grouped.gateway.log.last_seen"]["severity"] == "info"
    assert "previous private text" not in dumped
    assert "current private text" not in dumped
    assert "agent private text" not in dumped


def test_private_strings_absent_from_snapshot_and_diff(tmp_path, monkeypatch):
    mod = load_module()
    snapshot = use_snapshot_path(mod, tmp_path, monkeypatch)
    previous = overview_payload(repo={"head": "abc123", "branch": "main", "describe": "ghp_fakeSECRET123456", "status_short": {"dirty_count": 1, "files": ["safe.py"]}})
    save_snapshot_for(mod, snapshot, previous)
    current = overview_payload(repo={"head": "def456", "branch": "main", "describe": "sk-test123456789", "status_short": {"dirty_count": 1, "files": ["safe.py"]}})
    monkeypatch.setattr(mod, "get_overview", lambda: current)
    result = mod.get_overview_diff(update_baseline=True)
    snapshot_text = snapshot.read_text(encoding="utf-8")
    dumped = json.dumps(result)
    assert "ghp_fake" not in snapshot_text
    assert "sk-test" not in snapshot_text
    assert "ghp_fake" not in dumped
    assert "sk-test" not in dumped
    assert result["data"]["updated_baseline"] is True
