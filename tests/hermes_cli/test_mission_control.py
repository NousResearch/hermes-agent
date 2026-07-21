from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient


def _seed_state_db(home: Path) -> None:
    con = sqlite3.connect(home / "state.db")
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            cwd TEXT,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT,
            rewind_count INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT,
            platform_message_id TEXT,
            observed INTEGER DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages_fts (content TEXT);
        CREATE TABLE summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, content TEXT, updated_at REAL);
        """
    )
    now = time.time()
    con.execute(
        """
        INSERT INTO sessions (
            id, source, model, started_at, ended_at, message_count,
            tool_call_count, input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens, reasoning_tokens, estimated_cost_usd, actual_cost_usd,
            title, api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "session-1",
            "telegram",
            "gpt-5.5",
            now - 3600,
            now - 60,
            2,
            3,
            1200,
            800,
            333,
            44,
            111,
            0.42,
            0.31,
            "Seed session",
            5,
        ),
    )
    con.executemany(
        "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
        [
            ("session-1", "user", "hello", None, now - 3500),
            ("session-1", "assistant", "done", None, now - 3400),
            ("session-1", "tool", "<secret should not surface>", "terminal", now - 3300),
        ],
    )
    con.execute("INSERT INTO summaries (session_id, content, updated_at) VALUES (?, ?, ?)", ("session-1", "private summary", now - 100))
    con.commit()
    con.close()


def test_blueprint_static_source_covers_all_steps_and_feature_picker():
    from hermes_cli.mission_control import BLUEPRINT_STEPS, HERMES_FEATURES, OPENCLAW_FEATURES

    assert [step["id"] for step in BLUEPRINT_STEPS] == [
        *[f"step-{i}" for i in range(1, 23)],
        "step-22-5",
        *[f"step-{i}" for i in range(23, 27)],
    ]
    assert [feature["id"] for feature in HERMES_FEATURES] == [f"H{i}" for i in range(1, 12)]
    assert [feature["id"] for feature in OPENCLAW_FEATURES] == [f"O{i}" for i in range(1, 11)]


def test_snapshot_uses_real_runtime_counts_and_compacts_sensitive_state(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "skills" / "example").mkdir(parents=True)
    (home / "skills" / "example" / "SKILL.md").write_text(
        "---\nname: example\ndescription: demo\n---\nmetadata:\n  created_by: agent\n\n# Example\n",
        encoding="utf-8",
    )
    (home / "skills" / "acme-client-lawsuit").mkdir(parents=True)
    (home / "skills" / "acme-client-lawsuit" / "SKILL.md").write_text(
        "---\nname: acme-client-lawsuit\ndescription: private workflow\n---\n\n# Private\n",
        encoding="utf-8",
    )
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "job-1", "enabled": True, "schedule": "0 9 * * *"}]}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        """
model: gpt-5.5
provider: openai-codex
agent:
  reasoning_effort: xhigh
approvals:
  mode: smart
security:
  redact_secrets: true
gateway:
  platforms:
    telegram:
      enabled: true
stt:
  enabled: true
  provider: local
tts:
  provider: edge
mcp_servers:
  gmail:
    command: npx
    args: [server]
""".strip(),
        encoding="utf-8",
    )
    (home / ".env").write_text(
        "OPENROUTER_API_KEY=placeholder-openrouter-key\n"
        "DASHBOARD_TOKEN=placeholder-dashboard-token\n"
        "TELEGRAM_BOT_TOKEN=placeholder-telegram-token\n"
        "ALLOWED_USER_IDS=123456,789012\n"
        "OPENAI_API_KEY=placeholder-openai-key\n"
        "PINECONE_API_KEY=placeholder-pinecone-key\n"
        "PINECONE_INDEX=private-index-name\n",
        encoding="utf-8",
    )
    (home / "soul.md").write_text("# Identity\nA real soul file\n", encoding="utf-8")
    _seed_state_db(home)

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["blueprint"]["stepCount"] == 27
    assert snapshot["blueprint"]["hermesFeatureCount"] == 11
    assert snapshot["blueprint"]["openclawFeatureCount"] == 10
    assert snapshot["runtime"]["sessions"]["total"] == 1
    assert snapshot["runtime"]["sessions"]["messages"] == 3
    assert snapshot["runtime"]["sessions"]["toolCalls"] == 3
    assert snapshot["runtime"]["skills"]["total"] >= 1
    assert snapshot["runtime"]["cron"]["total"] == 1
    assert snapshot["runtime"]["mcp"]["configured"] == 1
    assert snapshot["runtime"]["model"]["provider"] == "openai-codex"
    assert snapshot["runtime"]["model"]["model"] == "gpt-5.5"
    assert snapshot["runtime"]["model"]["reasoning"] == "xhigh"
    assert snapshot["runtime"]["env"]["telegram"]["allowedUserCount"] == 2
    assert snapshot["runtime"]["env"]["dashboard"]["tokenPresent"] is True
    assert snapshot["runtime"]["semantic"]["configured"] is True
    assert snapshot["runtime"]["semantic"]["indexConfigured"] is True
    assert snapshot["runtime"]["sessions"]["cacheReadTokens"] == 333
    assert snapshot["runtime"]["sessions"]["cacheWriteTokens"] == 44
    assert snapshot["runtime"]["sessions"]["actualCostUsd"] == 0.31
    assert snapshot["runtime"]["sessions"]["apiCalls"] == 5
    assert snapshot["runtime"]["sessions"]["ftsPresent"] is True
    assert snapshot["runtime"]["memory"]["sqlite"]["summaries"] == 1
    assert len(snapshot["runtime"]["preflight"]) == 11
    assert len(snapshot["runtime"]["customization"]) == 9
    assert len(snapshot["runtime"]["dataFlow"]) == 5
    assert snapshot["runtime"]["production"]["score"] >= 0
    assert len(snapshot["blueprint"]["architecturePieces"]) == 7
    assert len(snapshot["blueprint"]["prerequisites"]) == 5
    assert len(snapshot["blueprint"]["nextTools"]) == 8
    assert len(snapshot["blueprint"]["troubleshooting"]) == 10
    assert all("id" not in item and "title" not in item for item in snapshot["runtime"]["sessions"]["recent"])
    assert any(item["id"] == "step-24" and item["missionControl"] for item in snapshot["coverage"]["steps"])

    assert str(home) not in encoded
    assert "***" not in encoded
    assert "placeholder-openrouter-key" not in encoded
    assert "placeholder-dashboard-token" not in encoded
    assert "placeholder-telegram-token" not in encoded
    assert "placeholder-openai-key" not in encoded
    assert "placeholder-pinecone-key" not in encoded
    assert "123456" not in encoded
    assert "789012" not in encoded
    assert "private-index-name" not in encoded
    assert "acme-client-lawsuit" not in encoded
    assert "private summary" not in encoded
    assert "Seed session" not in encoded
    assert "session-1" not in encoded
    assert "<secret should not surface>" not in encoded


def test_snapshot_counts_gateway_telegram_allowlist_env_without_exposing_ids(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        """
model: gpt-5.5
provider: openai-codex
gateway:
  platforms:
    telegram:
      enabled: true
""".strip(),
        encoding="utf-8",
    )
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=placeholder-telegram-token\n"
        "TELEGRAM_ALLOWED_USERS=123456,789012\n",
        encoding="utf-8",
    )
    _seed_state_db(home)

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["runtime"]["env"]["telegram"]["allowedUserCount"] == 2
    allowed_key = next(item for item in snapshot["runtime"]["env"]["requiredKeys"] if item["key"] == "ALLOWED_USER_IDS")
    assert allowed_key["isSet"] is True
    assert allowed_key["count"] == 2
    allowed_preflight = next(item for item in snapshot["runtime"]["preflight"] if item["id"] == "allowed_users")
    assert allowed_preflight["status"] == "pass"
    assert "123456" not in encoded
    assert "789012" not in encoded


def test_snapshot_treats_systemd_gateway_service_as_managed_hosting(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        """
model: gpt-5.5
provider: openai-codex
approvals:
  mode: smart
""".strip(),
        encoding="utf-8",
    )
    (home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=placeholder-telegram-token\n"
        "TELEGRAM_ALLOWED_USERS=123456\n",
        encoding="utf-8",
    )
    _seed_state_db(home)

    import hermes_cli.gateway as gateway_cli

    class FakeGatewaySnapshot:
        manager = "systemd (user)"
        service_installed = True
        service_running = True
        gateway_pids = (4242,)
        service_scope = "user"

        @property
        def running(self):
            return self.service_running or bool(self.gateway_pids)

        @property
        def has_process_service_mismatch(self):
            return False

    monkeypatch.setattr(gateway_cli, "get_gateway_runtime_snapshot", lambda system=False: FakeGatewaySnapshot())

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    hosting = snapshot["runtime"]["hosting"]

    assert snapshot["runtime"]["gateway"]["serviceManager"] == "systemd"
    assert snapshot["runtime"]["gateway"]["serviceInstalled"] is True
    assert snapshot["runtime"]["gateway"]["serviceRunning"] is True
    assert hosting["installMethod"] == "systemd"
    assert hosting["managedService"] is True
    hosting_signal = next(item for item in snapshot["runtime"]["production"]["signals"] if item["id"] == "hosting")
    assert hosting_signal["status"] == "pass"
    step_25 = next(item for item in snapshot["coverage"]["steps"] if item["id"] == "step-25")
    assert step_25["status"] == "active"


def test_snapshot_treats_uppercase_soul_md_as_identity_file(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("model: gpt-5.5\nprovider: openai-codex\n", encoding="utf-8")
    (home / "SOUL.md").write_text("You are direct and technically precise.\n", encoding="utf-8")
    _seed_state_db(home)

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()

    assert snapshot["runtime"]["identity"]["soulPresent"] is True
    assert "~/.hermes/SOUL.md" in snapshot["runtime"]["identity"]["files"]
    step_5 = next(item for item in snapshot["coverage"]["steps"] if item["id"] == "step-5")
    assert step_5["status"] == "active"


def test_snapshot_redacts_runtime_labels_that_can_embed_private_names(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "skills" / "acme-client-lawsuit" / "internal").mkdir(parents=True)
    (home / "skills" / "acme-client-lawsuit" / "internal" / "SKILL.md").write_text(
        "---\nname: acme-private-skill\ndescription: private workflow\n---\n\n# Private\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        """
model:
  provider: local
  default: /private/models/acme-client-local.gguf
mcp_servers:
  acme-prod-db:
    command: npx
    args: [server]
""".strip(),
        encoding="utf-8",
    )
    _seed_state_db(home)
    con = sqlite3.connect(home / "state.db")
    con.execute(
        "UPDATE sessions SET source=?, model=? WHERE id=?",
        ("telegram:123456789:secret-topic", "/private/models/session-secret.gguf", "session-1"),
    )
    con.commit()
    con.close()

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["runtime"]["model"]["model"] == "local-model"
    assert snapshot["runtime"]["sessions"]["sources"] == {"telegram": 1}
    assert snapshot["runtime"]["sessions"]["recent"][0]["source"] == "telegram"
    assert snapshot["runtime"]["sessions"]["recent"][0]["model"] == "local-model"
    assert snapshot["runtime"]["mcp"]["servers"] == ["server-1"]
    assert snapshot["runtime"]["mcp"]["serverNamesRedacted"] is True
    assert set(snapshot["runtime"]["skills"]["categories"]) <= {"direct", "grouped"}

    for forbidden in [
        "/private/models",
        "acme-client-local",
        "session-secret",
        "telegram:123456789",
        "123456789",
        "secret-topic",
        "acme-prod-db",
        "acme-client-lawsuit",
        "acme-private-skill",
    ]:
        assert forbidden not in encoded


def test_snapshot_redacts_toolset_gateway_and_provider_labels(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        """
model:
  provider: acme-private-provider
  default: file:///private/models/acme-client-model.gguf
delegation:
  provider: acme-delegation-private
semantic:
  provider: acme-vector-client
  index: private-index-name
stt:
  enabled: true
  provider: acme-whisper
tts:
  provider: acme-voice
toolsets:
  - terminal
  - mcp-acme-prod-db
  - private-client-plugin
agent:
  reasoning_effort: acme-private-reasoning
  disabled_toolsets:
    - mcp-secret-calendar
approvals:
  mode: acme-secret-approval
  cron_mode: acme-secret-cron
terminal:
  backend: acme-secret-backend
tools:
  max_output_chars: 9000
gateway:
  platforms:
    telegram:
      enabled: true
    telegram:123456789:secret-topic:
      enabled: true
    acme-private-gateway:
      enabled: true
mcp_servers:
  acme-prod-db:
    command: npx
    args: [server]
""".strip(),
        encoding="utf-8",
    )
    _seed_state_db(home)

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["runtime"]["tools"]["configuredToolsetCount"] == 3
    assert snapshot["runtime"]["tools"]["disabledToolsetCount"] == 1
    assert snapshot["runtime"]["gateway"]["configuredCount"] == 3
    assert snapshot["runtime"]["gateway"]["configuredPlatforms"] == ["other", "telegram"]
    assert snapshot["runtime"]["model"]["provider"] == "custom"
    assert snapshot["runtime"]["model"]["delegationProvider"] == "custom"
    assert snapshot["runtime"]["model"]["reasoning"] == "custom"
    assert snapshot["runtime"]["semantic"]["provider"] == "custom"
    assert snapshot["runtime"]["voice"]["sttProvider"] == "custom"
    assert snapshot["runtime"]["voice"]["ttsProvider"] == "custom"
    assert snapshot["runtime"]["safety"]["approvalsMode"] == "custom"
    assert snapshot["runtime"]["safety"]["cronApprovalsMode"] == "custom"
    assert snapshot["runtime"]["safety"]["terminalBackend"] == "custom"
    assert snapshot["runtime"]["safety"]["toolOutputLimits"]["configured"] is True
    assert snapshot["runtime"]["safety"]["toolOutputLimits"]["minChars"] == 9000
    assert snapshot["runtime"]["safety"]["promptInjection"]["toolOutputLimitConfigured"] is True

    for forbidden in [
        "acme-private-provider",
        "acme-delegation-private",
        "acme-vector-client",
        "acme-private-reasoning",
        "acme-secret-approval",
        "acme-secret-cron",
        "acme-secret-backend",
        "acme-whisper",
        "acme-voice",
        "file:///private",
        "acme-client-model",
        "private-index-name",
        "mcp-acme-prod-db",
        "private-client-plugin",
        "mcp-secret-calendar",
        "telegram:123456789",
        "123456789",
        "secret-topic",
        "acme-private-gateway",
        "acme-prod-db",
    ]:
        assert forbidden not in encoded


def test_snapshot_redacts_sensitive_session_db_columns(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("model: gpt-5.5\nprovider: openai-codex\n", encoding="utf-8")
    _seed_state_db(home)
    con = sqlite3.connect(home / "state.db")
    con.execute(
        """
        UPDATE sessions SET
            user_id=?, cwd=?, model_config=?, billing_base_url=?, handoff_platform=?, handoff_error=?, end_reason=?, parent_session_id=?, rewind_count=?
        WHERE id=?
        """,
        (
            "user-canary-987654",
            "/private/workspaces/acme-client",
            '{"deployment":"acme-private-deploy"}',
            "https://acme-private-provider.example/v1",
            "telegram:987654:secret-thread",
            "private handoff error canary",
            "max_iterations",
            "parent-secret-session-id",
            2,
            "session-1",
        ),
    )
    con.execute(
        """
        UPDATE messages SET
            platform_message_id=?, tool_calls=?, tool_name=?, reasoning=?, reasoning_content=?, reasoning_details=?, codex_reasoning_items=?, codex_message_items=?
        WHERE id=1
        """,
        (
            "platform-message-canary",
            '[{"name":"private_tool","arguments":{"cmd":"rm -rf /private/acme"}}]',
            "mcp_acme_private_tool",
            "private reasoning canary",
            "private reasoning content canary",
            "private reasoning details canary",
            "private codex reasoning item canary",
            "private codex message item canary",
        ),
    )
    con.commit()
    con.close()

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["runtime"]["sessions"]["endReasonCounts"]["max_iterations"] == 1
    assert snapshot["runtime"]["sessions"]["childSessionCount"] == 1
    assert snapshot["runtime"]["sessions"]["rewindTotal"] == 2

    for forbidden in [
        "user-canary-987654",
        "/private/workspaces/acme-client",
        "acme-private-deploy",
        "acme-private-provider.example",
        "telegram:987654",
        "secret-thread",
        "private handoff error canary",
        "parent-secret-session-id",
        "platform-message-canary",
        "private_tool",
        "rm -rf",
        "/private/acme",
        "mcp_acme_private_tool",
        "private reasoning canary",
        "private reasoning content canary",
        "private reasoning details canary",
        "private codex reasoning item canary",
        "private codex message item canary",
    ]:
        assert forbidden not in encoded


def test_snapshot_redacts_cron_mcp_commands_and_delivery_targets(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "cron").mkdir(parents=True, exist_ok=True)
    now = time.time()
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "private-job-id",
                        "enabled": True,
                        "name": "acme private reflection heartbeat",
                        "prompt": "read /private/acme/secret and send it",
                        "script": "/private/acme/cron.py",
                        "schedule": "every 30m",
                        "deliver": "telegram:123456789:secret-topic",
                        "last_error": "private cron error canary",
                        "last_run_at": now - 120,
                        "next_run_at": now - 30,
                        "last_status": "error",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        """
mcp_servers:
  acme-private-server:
    command: /private/bin/mcp-acme
    args: [--token, private-arg-canary]
    url: https://acme-private-mcp.example/sse
    headers:
      Authorization: private-header-canary
""".strip(),
        encoding="utf-8",
    )
    _seed_state_db(home)

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["runtime"]["cron"]["total"] == 1
    assert snapshot["runtime"]["cron"]["overdueCount"] == 1
    assert snapshot["runtime"]["cron"]["lastStatusCounts"]["error"] == 1
    assert snapshot["runtime"]["cron"]["lastRunAgeBuckets"]["under_1h"] == 1
    assert snapshot["runtime"]["cron"]["reflectionFreshness"] == "fresh"
    assert snapshot["runtime"]["cron"]["reflectionLastRunAgeSeconds"] is not None
    assert snapshot["runtime"]["mcp"]["configured"] == 1
    assert snapshot["runtime"]["mcp"]["statusCounts"]["enabled"] == 1
    assert snapshot["runtime"]["mcp"]["serverNamesRedacted"] is True

    for forbidden in [
        "private-job-id",
        "acme private heartbeat",
        "acme private reflection",
        "/private/acme",
        "telegram:123456789",
        "123456789",
        "secret-topic",
        "private cron error canary",
        "acme-private-server",
        "/private/bin/mcp-acme",
        "private-arg-canary",
        "acme-private-mcp.example",
        "private-header-canary",
    ]:
        assert forbidden not in encoded


def test_model_label_sanitizer_preserves_public_families_only():
    from hermes_cli.mission_control import _safe_model_label

    assert _safe_model_label("gpt-5.5") == "gpt-5.5"
    assert _safe_model_label("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
    assert _safe_model_label("openrouter/google/gemini-2.5-pro-preview") == "openrouter/google/gemini-2.5-pro-preview"
    assert _safe_model_label("gpt-4o") == "gpt-4o"
    assert _safe_model_label("llama-7b") == "llama-7b"
    assert _safe_model_label("qwen2.5-coder") == "qwen2.5-coder"
    assert _safe_model_label("deepseek-r1") == "deepseek-r1"
    assert _safe_model_label("llama3.1-instruct") == "llama3.1-instruct"
    assert _safe_model_label("acme123") == "custom-model"
    assert _safe_model_label("client42") == "custom-model"
    assert _safe_model_label("prod2") == "custom-model"
    assert _safe_model_label("openai/acme2") == "custom-model"
    assert _safe_model_label("123client") == "custom-model"
    assert _safe_model_label("42prod") == "custom-model"
    assert _safe_model_label("v2acme") == "custom-model"
    assert _safe_model_label("openai/123client") == "custom-model"
    assert _safe_model_label("openai/o3acme") == "custom-model"
    assert _safe_model_label("llama3client") == "custom-model"
    assert _safe_model_label("qwen2prod") == "custom-model"
    assert _safe_model_label("gemma2private") == "custom-model"
    assert _safe_model_label("glm4acme") == "custom-model"
    assert _safe_model_label("r1client") == "custom-model"
    assert _safe_model_label("gpt-4o-客户") == "custom-model"
    assert _safe_model_label("openai/acme-private-client-deploy") == "custom-model"
    assert _safe_model_label("gpt-5.5-acme-private") == "custom-model"
    assert _safe_model_label("https://models.example/acme-private-model") == "custom-model"


def test_mcp_metrics_count_all_servers_while_sampling_redacted_details(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    server_lines = ["mcp_servers:"]
    for idx in range(1, 14):
        server_lines.extend([
            f"  acme-private-server-{idx}:",
            "    command: npx",
            "    args: [server]",
        ])
        if idx == 13:
            server_lines.append("    enabled: false")
    (home / "config.yaml").write_text("\n".join(server_lines), encoding="utf-8")

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot()
    encoded = json.dumps(snapshot, ensure_ascii=False)
    mcp = snapshot["runtime"]["mcp"]

    assert mcp["configured"] == 13
    assert mcp["enabled"] == 12
    assert mcp["disabled"] == 1
    assert mcp["statusCounts"] == {"enabled": 12, "disabled": 1}
    assert mcp["transportCounts"]["stdio"] == 13
    assert len(mcp["servers"]) == 12
    assert len(mcp["serverDetails"]) == 12
    assert mcp["serverNamesRedacted"] is True
    assert "acme-private-server" not in encoded


def test_snapshot_uses_live_dashboard_bind_and_auth_state(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("dashboard:\n  host: 127.0.0.1\n", encoding="utf-8")

    from hermes_cli.mission_control import build_mission_control_snapshot

    snapshot = build_mission_control_snapshot({"bound_host": "0.0.0.0", "auth_required": False})
    dashboard = snapshot["runtime"]["dashboard"]

    assert dashboard["bindExposure"] == "public"
    assert dashboard["runtimeHostKnown"] is True
    assert dashboard["authGated"] is False
    assert dashboard["authMode"] == "not-gated"


def test_dashboard_endpoint_serves_mission_control_snapshot():
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    unauth = client.get("/api/mission-control/blueprint")
    assert unauth.status_code == 401

    res = client.get(
        "/api/mission-control/blueprint",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["blueprint"]["stepCount"] == 27
    assert data["coverage"]["summary"]["total"] >= 48


def test_dashboard_endpoint_works_with_oauth_cookie_gate():
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import DashboardAuthProvider, LoginStart, Session, clear_providers, register_provider

    class DummyProvider(DashboardAuthProvider):
        name = "dummy"
        display_name = "Dummy"

        def start_login(self, *, redirect_uri: str) -> LoginStart:
            return LoginStart(redirect_url=redirect_uri, cookie_payload={})

        def complete_login(self, *, code: str, state: str, code_verifier: str, redirect_uri: str) -> Session:
            raise NotImplementedError

        def verify_session(self, *, access_token: str):
            if access_token != "valid-access-token":
                return None
            return Session(
                user_id="user-1",
                email="",
                display_name="",
                org_id="",
                provider=self.name,
                expires_at=int(time.time()) + 3600,
                access_token=access_token,
                refresh_token="refresh-token",
            )

        def refresh_session(self, *, refresh_token: str) -> Session:
            raise NotImplementedError

        def revoke_session(self, *, refresh_token: str) -> None:
            return None

    previous_auth_required = getattr(web_server.app.state, "auth_required", False)
    clear_providers()
    register_provider(DummyProvider())
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app)
        client.cookies.set("hermes_session_at", "valid-access-token")
        res = client.get("/api/mission-control/blueprint")
        assert res.status_code == 200
        assert res.json()["blueprint"]["stepCount"] == 27
    finally:
        web_server.app.state.auth_required = previous_auth_required
        clear_providers()
