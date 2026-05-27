import json
import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient

from hermes_cli import web_server as ws


def _write_dashboard_agents_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
version: "2026-05-21"
agents:
  - agent_id: pirlo
    name: Pirlo 商业策划师
    role: planning_strategist
    role_summary: 商业方案。
    model_ref: opencode_go_kimi25
    model_strategy:
      mode: fallback
      primary: opencode_go_kimi25
      chain: [opencode_go_kimi25, opencode_go_kimi26]
    tools: [file]
    permission: read_only
    can_delegate: false
    capabilities: [content_writing]
    risk_allowed: [R0, R1, R2]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_dashboard_models_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
models:
  opencode_go_kimi25:
    provider: opencode-go
    base_url: https://opencode.ai/zen/go/v1
    api_key_env: OPENCODE_GO_API_KEY
    model: kimi-k2.5
    status: experimental
  deepseek_pro:
    provider: deepseek
    base_url: https://deepseek.test
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-pro
    status: active
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _client() -> TestClient:
    return TestClient(ws.app)


def _headers() -> dict[str, str]:
    return {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}


def test_update_agent_model_preserves_fallback_strategy(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    resp = _client().put(
        "/api/agents/pirlo/model",
        headers=_headers(),
        json={"model_ref": "deepseek_pro"},
    )

    assert resp.status_code == 200, resp.text
    saved = ws.yaml.safe_load(agents_path.read_text(encoding="utf-8"))
    pirlo = saved["agents"][0]
    assert pirlo["model_ref"] == "deepseek_pro"
    assert pirlo["model_strategy"] == {
        "mode": "fallback",
        "primary": "deepseek_pro",
        "chain": ["deepseek_pro", "opencode_go_kimi25"],
        "fallback_on": [],
    }
    mirror = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert mirror["agents"]["pirlo"]["subagent_profile"]["model_strategy"]["mode"] == "fallback"


def test_agent_console_records_managed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    monkeypatch.setattr(ws, "_valid_agent_ids", None)
    monkeypatch.setattr(ws, "_external_runtime_agent_ids", None)

    resp = _client().post(
        "/api/agents/pirlo/runs",
        headers=_headers(),
        json={"prompt": "写一个商业方案", "workspace": str(tmp_path), "risk_level": "R1"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent_id"] == "pirlo"
    assert body["display_name"] == "Pirlo 商业策划师"
    assert body["prompt"] == "写一个商业方案"
    assert body["risk_level"] == "R1"
    assert body["model_ref"]
    assert body["status"] == "completed"
    assert "does not invoke" in body["result_summary"]

    stored = json.loads((tmp_path / "agent-runs.json").read_text(encoding="utf-8"))
    assert stored["runs"][0]["run_id"] == body["run_id"]


def test_agent_console_rejects_unknown_and_external_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_AGENT_RUNS_PATH", tmp_path / "agent-runs.json")
    monkeypatch.setattr(ws, "_valid_agent_ids", None)
    monkeypatch.setattr(ws, "_external_runtime_agent_ids", None)
    client = _client()

    unknown = client.post("/api/agents/nope/runs", headers=_headers(), json={"prompt": "x"})
    assert unknown.status_code == 400

    external = client.post("/api/agents/claude/runs", headers=_headers(), json={"prompt": "x"})
    assert external.status_code == 400
    assert "external CLI runtime" in external.text


def test_agent_console_session_sends_message_and_stores_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_AGENT_CONSOLE_SESSIONS_PATH", tmp_path / "agent-console-sessions.json")
    monkeypatch.setattr(ws, "_valid_agent_ids", None)
    monkeypatch.setattr(ws, "_external_runtime_agent_ids", None)

    def fake_run(session, prompt):
        assert session["agent_id"] == "pirlo"
        assert prompt == "hello"
        return {
            "content": "hi from pirlo",
            "status": "completed",
            "duration_seconds": 0.1,
            "api_calls": 1,
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "model": "fake-model",
        }

    monkeypatch.setattr(ws, "_run_agent_console_turn", fake_run)
    client = _client()

    created = client.post(
        "/api/agents/pirlo/console/sessions",
        headers=_headers(),
        json={"workspace": str(tmp_path), "risk_level": "R0"},
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    sent = client.post(
        f"/api/agents/console/sessions/{session_id}/messages",
        headers=_headers(),
        json={"prompt": "hello", "workspace": str(tmp_path), "risk_level": "R0"},
    )
    assert sent.status_code == 200, sent.text
    body = sent.json()
    assert body["status"] == "idle"
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["content"] == "hi from pirlo"
    assert body["messages"][1]["api_calls"] == 1

    stored = json.loads((tmp_path / "agent-console-sessions.json").read_text(encoding="utf-8"))
    assert stored["sessions"][0]["messages"][1]["content"] == "hi from pirlo"

    deleted = client.delete(f"/api/agents/console/sessions/{session_id}", headers=_headers())
    assert deleted.status_code == 200, deleted.text
    stored_after_delete = json.loads((tmp_path / "agent-console-sessions.json").read_text(encoding="utf-8"))
    assert stored_after_delete["sessions"] == []


def test_delegations_api_groups_subagent_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            source TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    now = time.time()
    conn.execute(
        "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
        (
            "e1",
            "s1",
            "task-a",
            "subagent.started",
            now,
            "system",
            json.dumps({"subagent_id": "sa1", "agent_id": "pirlo", "goal_preview": "plan"}),
        ),
    )
    conn.execute(
        "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
        (
            "e2",
            "s1",
            "task-a",
            "subagent.completed",
            now + 1,
            "system",
            json.dumps({
                "subagent_id": "sa1",
                "agent_id": "pirlo",
                "status": "completed",
                "fallback_activations": [
                    {
                        "from_model": "kimi-k2.5",
                        "to_model": "kimi-k2.6",
                        "reason": "rate_limit",
                    }
                ],
                "fallback_continuation": {
                    "risk": "retry_after_model_switch",
                    "continuation_guarantee": "conversation_retry_not_tool_checkpoint",
                },
            }),
        ),
    )
    conn.commit()
    conn.close()

    client = _client()
    listing = client.get("/api/delegations?days=1&agent_id=pirlo", headers=_headers())
    assert listing.status_code == 200, listing.text
    assert listing.json()["total"] == 1
    delegation = listing.json()["delegations"][0]
    assert delegation["task_id"] == "task-a"
    assert delegation["fallback_activation_count"] == 1
    assert delegation["fallback_continuation_risk"] == "retry_after_model_switch"

    trace = client.get("/api/delegations/task-a", headers=_headers())
    assert trace.status_code == 200, trace.text
    assert [event["type"] for event in trace.json()["events"]] == [
        "subagent.started",
        "subagent.completed",
    ]
    assert trace.json()["events"][0]["agent_id"] == "pirlo"
    completed = trace.json()["events"][1]
    assert completed["fallback_activations"][0]["to_model"] == "kimi-k2.6"
    assert completed["fallback_continuation"]["continuation_guarantee"] == "conversation_retry_not_tool_checkpoint"

    limited = client.get("/api/delegations?days=1&limit=1", headers=_headers())
    assert limited.status_code == 200, limited.text
    assert limited.json()["total"] == 1
    assert len(limited.json()["delegations"]) == 1
