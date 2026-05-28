import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

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
  - agent_id: claude
    name: Claude 主程执行官
    role: lead_implementer
    role_summary: 外部 Claude Code CLI。
    model_ref: deepseek_pro
    model_strategy:
      mode: external
      primary: deepseek_pro
      chain: [deepseek_pro]
    runtime: claude_code_cli
    tools: [file, terminal]
    permission: ask
    can_delegate: false
    capabilities: [code_edit]
    risk_allowed: [R1, R2, R3]
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


def test_update_agent_model_strategy_reorders_fallback_chain(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    resp = _client().put(
        "/api/agents/pirlo/model-strategy",
        headers=_headers(),
        json={
            "mode": "fallback",
            "primary": "deepseek_pro",
            "chain": ["deepseek_pro", "opencode_go_kimi25"],
            "fallback_on": ["rate_limited", "timeout"],
        },
    )

    assert resp.status_code == 200, resp.text
    saved = ws.yaml.safe_load(agents_path.read_text(encoding="utf-8"))
    pirlo = saved["agents"][0]
    assert pirlo["model_ref"] == "deepseek_pro"
    assert pirlo["model_strategy"] == {
        "mode": "fallback",
        "primary": "deepseek_pro",
        "chain": ["deepseek_pro", "opencode_go_kimi25"],
        "fallback_on": ["rate_limited", "timeout"],
    }
    mirror = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert mirror["agents"]["pirlo"]["subagent_profile"]["model_strategy"]["primary"] == "deepseek_pro"


def test_update_agent_model_strategy_rejects_invalid_requests(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    one_model = _client().put(
        "/api/agents/pirlo/model-strategy",
        headers=_headers(),
        json={"mode": "fallback", "primary": "deepseek_pro", "chain": ["deepseek_pro"]},
    )
    assert one_model.status_code == 400

    unknown = _client().put(
        "/api/agents/pirlo/model-strategy",
        headers=_headers(),
        json={"mode": "fixed", "primary": "missing_model", "chain": ["missing_model"]},
    )
    assert unknown.status_code == 400


def test_update_agent_model_strategy_switches_to_fixed(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    resp = _client().put(
        "/api/agents/pirlo/model-strategy",
        headers=_headers(),
        json={
            "mode": "fixed",
            "primary": "deepseek_pro",
            "chain": ["deepseek_pro", "opencode_go_kimi25"],
            "fallback_on": ["rate_limited"],
        },
    )

    assert resp.status_code == 200, resp.text
    saved = ws.yaml.safe_load(agents_path.read_text(encoding="utf-8"))
    pirlo = saved["agents"][0]
    assert pirlo["model_ref"] == "deepseek_pro"
    assert pirlo["model_strategy"] == {
        "mode": "fixed",
        "primary": "deepseek_pro",
        "chain": ["deepseek_pro"],
        "fallback_on": [],
    }


def test_update_agent_model_strategy_defaults_and_deduplicates(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    resp = _client().put(
        "/api/agents/pirlo/model-strategy",
        headers=_headers(),
        json={
            "mode": "fallback",
            "primary": "deepseek_pro",
            "chain": ["deepseek_pro", "deepseek_pro", "opencode_go_kimi25", "deepseek_pro"],
        },
    )

    assert resp.status_code == 200, resp.text
    strategy = resp.json()["model_strategy"]
    assert strategy["chain"] == ["deepseek_pro", "opencode_go_kimi25"]
    assert strategy["fallback_on"] == [
        "quota_exceeded",
        "rate_limited",
        "timeout",
        "server_error",
        "empty_final_content",
    ]


def test_update_agent_model_strategy_rejects_external_cli_agent(tmp_path, monkeypatch):
    agents_path = tmp_path / "agents.yaml"
    models_path = tmp_path / "models.yaml"
    runtime_path = tmp_path / "agent-registry.json"
    _write_dashboard_agents_yaml(agents_path)
    _write_dashboard_models_yaml(models_path)
    monkeypatch.setattr(ws, "_AGENTS_CONFIG_PATH", agents_path)
    monkeypatch.setattr(ws, "_MODELS_CONFIG_PATH", models_path)
    monkeypatch.setattr(ws, "_RUNTIME_AGENT_REGISTRY_PATH", runtime_path)

    resp = _client().put(
        "/api/agents/claude/model-strategy",
        headers=_headers(),
        json={"mode": "fixed", "primary": "deepseek_pro", "chain": ["deepseek_pro"]},
    )

    assert resp.status_code == 400
    assert "external CLI runtime" in resp.text


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
        json={"workspace": str(tmp_path), "risk_level": "R0", "mode": "task"},
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


def test_agent_console_light_mode_uses_direct_chat_not_delegate(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_AGENT_CONSOLE_SESSIONS_PATH", tmp_path / "agent-console-sessions.json")
    monkeypatch.setattr(ws, "_valid_agent_ids", None)
    monkeypatch.setattr(ws, "_external_runtime_agent_ids", None)

    def forbidden_delegate(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("light console chat must not call delegate_task")

    def fake_light(session, prompt):
        assert session["mode"] == "light"
        assert prompt == "hello"
        return {
            "content": "quick reply",
            "status": "completed",
            "duration_seconds": 0.02,
            "api_calls": 1,
            "usage": {"input_tokens": 3, "output_tokens": 4},
            "model": "kimi-k2.5",
            "model_ref": "opencode_go_kimi25",
            "mode": "light",
        }

    monkeypatch.setattr(ws, "_run_agent_console_turn", forbidden_delegate)
    monkeypatch.setattr(ws, "_run_agent_console_light_turn", fake_light)
    client = _client()

    created = client.post(
        "/api/agents/pirlo/console/sessions",
        headers=_headers(),
        json={"workspace": str(tmp_path), "risk_level": "R0", "mode": "light"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["mode"] == "light"

    sent = client.post(
        f"/api/agents/console/sessions/{created.json()['session_id']}/messages",
        headers=_headers(),
        json={"prompt": "hello", "mode": "light"},
    )
    assert sent.status_code == 200, sent.text
    body = sent.json()
    assert body["status"] == "idle"
    assert body["mode"] == "light"
    assert body["messages"][0]["mode"] == "light"
    assert body["messages"][1]["content"] == "quick reply"
    assert body["messages"][1]["mode"] == "light"
    assert body["messages"][1]["model_ref"] == "opencode_go_kimi25"


def test_agent_console_task_mode_keeps_delegate_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_AGENT_CONSOLE_SESSIONS_PATH", tmp_path / "agent-console-sessions.json")
    monkeypatch.setattr(ws, "_valid_agent_ids", None)
    monkeypatch.setattr(ws, "_external_runtime_agent_ids", None)

    called = {"delegate": False}

    def fake_delegate(session, prompt):
        called["delegate"] = True
        assert session["mode"] == "task"
        assert prompt == "make a plan"
        return {
            "content": "delegated reply",
            "status": "completed",
            "duration_seconds": 1.2,
            "api_calls": 2,
            "usage": {},
            "model": "task-model",
            "model_ref": "opencode_go_kimi25",
            "mode": "task",
        }

    monkeypatch.setattr(ws, "_run_agent_console_turn", fake_delegate)
    client = _client()
    created = client.post(
        "/api/agents/pirlo/console/sessions",
        headers=_headers(),
        json={"workspace": str(tmp_path), "risk_level": "R1", "mode": "task"},
    )
    assert created.status_code == 200, created.text

    sent = client.post(
        f"/api/agents/console/sessions/{created.json()['session_id']}/messages",
        headers=_headers(),
        json={"prompt": "make a plan", "mode": "task"},
    )
    assert sent.status_code == 200, sent.text
    assert called["delegate"] is True
    assert sent.json()["messages"][1]["mode"] == "task"


def test_agent_console_light_turn_resolves_model_and_extracts_text(monkeypatch):
    agent = SimpleNamespace(
        name="Pirlo 商业策划师",
        role_summary="商业方案。",
        model_ref="opencode_go_kimi25",
        model_strategy={},
    )
    monkeypatch.setattr(ws, "_resolve_console_agent", lambda _agent_id: ("pirlo", agent))
    monkeypatch.setattr(ws, "_resolve_profile_model_ref", lambda ref: {
        "model_ref": ref,
        "provider": "opencode-go",
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "model": "kimi-k2.5",
    })

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from model"))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(ws, "OpenAI", FakeOpenAI)
    session = {
        "session_id": "acs-test",
        "agent_id": "pirlo",
        "display_name": "Pirlo 商业策划师",
        "workspace": "/tmp",
        "model_ref": "opencode_go_kimi25",
        "messages": [],
    }

    result = ws._run_agent_console_light_turn(session, "say hi")

    assert result["content"] == "hello from model"
    assert result["mode"] == "light"
    assert result["model"] == "kimi-k2.5"
    assert result["model_ref"] == "opencode_go_kimi25"
    assert captured["model"] == "kimi-k2.5"
    assert captured["client_kwargs"]["base_url"] == "https://opencode.ai/zen/go/v1"


def test_console_chat_messages_do_not_duplicate_current_prompt():
    agent = SimpleNamespace(name="Pirlo 商业策划师", role_summary="商业方案。")
    session = {
        "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ]
    }

    messages = ws._console_chat_messages(session, agent, "current question")

    user_contents = [msg["content"] for msg in messages if msg["role"] == "user"]
    assert user_contents == ["old question", "current question"]


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
