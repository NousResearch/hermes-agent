"""Tests for the Workflows dashboard plugin backend."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "workflows" / "dashboard"

PASS_SPEC = {
    "id": "dashboard_demo",
    "name": "Dashboard Demo",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
}

UNSUPPORTED_SPEC = {
    "id": "unsupported_dashboard_demo",
    "name": "Unsupported Dashboard Demo",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {"start": {"type": "send_message", "output": {"text": "hi"}}},
}

WAIT_SPEC = {
    "id": "dashboard_wait",
    "name": "Dashboard Wait",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {
        "start": {"type": "pass", "output": {"seen": "${ input.value }"}},
        "pause": {"type": "wait", "seconds": 60},
    },
    "edges": [{"from": "start", "to": "pause"}],
}

REQUIRED_RUN_SPEC = {
    "id": "dashboard_required_run",
    "name": "Dashboard Required Run",
    "version": 1,
    "triggers": [{
        "type": "manual",
        "id": "manual",
        "input_schema": {"brief": {"kind": "long_text", "required": True, "min_length": 3}},
        "intake": {"ready_when": {"op": "exists", "path": "$.input.brief"}},
    }],
    "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
}


def _load_plugin_module():
    plugin_file = PLUGIN_DIR / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_workflows_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_router():
    return _load_plugin_module().router


def test_dashboard_api_error_helpers_redact_secret_values():
    mod = _load_plugin_module()
    bad_secret = "sk-dashboardsecret1234567890"
    runtime_secret = "sk-dashboardruntime1234567890"

    bad_request = mod._http_400(ValueError(f"provider failed api_key={bad_secret}"))
    runtime = mod._assistant_runtime_http(f"provider failed {runtime_secret}")

    bad_text = json.dumps(bad_request.detail)
    runtime_text = json.dumps(runtime.detail)
    assert bad_secret not in bad_text
    assert runtime_secret not in runtime_text
    assert "provider failed" in bad_text
    assert "provider failed" in runtime_text


@pytest.fixture
def workflows_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    wfdb.init_db()
    return home


@pytest.fixture
def client(workflows_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/workflows")
    return TestClient(app)


def _deploy(client: TestClient, spec: dict = PASS_SPEC) -> dict:
    r = client.post("/api/plugins/workflows/definitions/deploy", json={"spec": spec})
    assert r.status_code == 200, r.text
    return r.json()["definition"]


CONTINUOUS_SPEC = {
    "id": "continuous_demo",
    "name": "Continuous Demo",
    "version": 1,
    "triggers": [
        {
            "type": "manual",
            "id": "kickoff",
            "input_schema": {
                "repo_path": {"kind": "repo_path", "required": True},
                "prompt": {"kind": "prompt", "required": True, "min_length": 10},
            },
            "intake": {"mode": "continuous", "dedupe_key": "$.input.repo_path"},
        }
    ],
    "nodes": {"start": {"type": "pass", "output": {"repo": "${ input.repo_path }"}}},
}


def test_dashboard_input_feed_api_opens_enqueues_updates_and_lists_items(client):
    _deploy(client, CONTINUOUS_SPEC)

    feed_res = client.post(
        "/api/plugins/workflows/definitions/continuous_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    )
    assert feed_res.status_code == 200, feed_res.text
    feed = feed_res.json()["feed"]
    assert feed["workflow_id"] == "continuous_demo"
    assert feed["status"] == "open"

    item_res = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/repo", "prompt": "short"}},
    )
    assert item_res.status_code == 200, item_res.text
    item = item_res.json()["item"]
    assert item["status"] == "needs_input"
    assert "at least 10 characters" in item["criteria"]["messages"][0]

    update_res = client.patch(
        f"/api/plugins/workflows/input-items/{item['item_id']}",
        json={"input": {"repo_path": "/repo", "prompt": "Review README drift"}},
    )
    assert update_res.status_code == 200, update_res.text
    assert update_res.json()["item"]["status"] == "queued"

    list_res = client.get(f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items")
    assert list_res.status_code == 200, list_res.text
    assert [row["item_id"] for row in list_res.json()["items"]] == [item["item_id"]]


def test_dashboard_input_feed_api_redacts_or_omits_dedupe_value(client):
    secret_spec = {
        "id": "secret_feed_demo",
        "name": "Secret Feed Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "kickoff",
            "input_schema": {"api_key": {"kind": "text", "required": True}},
            "intake": {"mode": "continuous", "dedupe_key": "$.input.api_key"},
        }],
        "nodes": {"start": {"type": "pass"}},
    }
    _deploy(client, secret_spec)
    feed = client.post(
        "/api/plugins/workflows/definitions/secret_feed_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    ).json()["feed"]

    item_res = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"api_key": "super-secret-key"}},
    )
    assert item_res.status_code == 200, item_res.text
    item = item_res.json()["item"]
    assert "dedupe_value" not in item
    assert item["input"]["api_key"] == "[REDACTED]"
    assert "super-secret-key" not in item_res.text

    list_res = client.get(f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items")
    assert list_res.status_code == 200, list_res.text
    assert "dedupe_value" not in list_res.json()["items"][0]
    assert "super-secret-key" not in list_res.text


def test_dashboard_input_feed_api_rejects_non_continuous_trigger(client):
    _deploy(client, PASS_SPEC)

    feed_res = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/input-feeds",
        json={"trigger_id": "manual"},
    )

    assert feed_res.status_code == 400
    assert "continuous manual input trigger" in feed_res.text


def test_dashboard_tick_endpoint_advances_workflows(client, monkeypatch):
    calls: list[int] = []

    def fake_tick(*, limit: int = 1) -> int:
        calls.append(limit)
        return 3

    monkeypatch.setattr(workflows_dispatcher, "tick", fake_tick)

    r = client.post("/api/plugins/workflows/tick", json={"limit": 2})

    assert r.status_code == 200, r.text
    assert r.json() == {"processed": 3}
    assert calls == [2]


def test_dashboard_input_feed_api_can_pause_resume_and_start_execution(client):
    _deploy(client, CONTINUOUS_SPEC)
    feed = client.post(
        "/api/plugins/workflows/definitions/continuous_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    ).json()["feed"]

    # 1) enqueue while open.
    ready = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/repo", "prompt": "Review README drift"}},
    ).json()["item"]
    assert ready["status"] == "queued"

    # 2) pause -> prove no further writes / admission.
    pause = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "paused"},
    )
    assert pause.status_code == 200, pause.text
    assert pause.json()["feed"]["status"] == "paused"
    blocked = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/repo2", "prompt": "second attempt"}},
    )
    assert blocked.status_code == 409, blocked.text
    assert workflows_dispatcher.tick(limit=1) == 0

    # 3) resume -> admission resumes.
    resume = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "open"},
    )
    assert resume.status_code == 200, resume.text
    assert workflows_dispatcher.tick(limit=1) == 1

    items = client.get(f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items").json()["items"]
    assert items[0]["status"] in {"running", "succeeded"}
    assert items[0]["execution_id"]

    # 4) close -> terminal; subsequent transitions and writes are rejected.
    close = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "closed"},
    )
    assert close.status_code == 200, close.text
    terminal = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "open"},
    )
    assert terminal.status_code == 409, terminal.text
    write_after_close = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/after", "prompt": "after close"}},
    )
    assert write_after_close.status_code == 409, write_after_close.text

    # 5) open a new feed; new feed_id.
    next_feed = client.post(
        "/api/plugins/workflows/definitions/continuous_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    ).json()["feed"]
    assert next_feed["feed_id"] != feed["feed_id"]
    assert next_feed["status"] == "open"


def test_dashboard_delete_definition_removes_workflow_and_related_runs(client):
    definition = _deploy(client, PASS_SPEC)
    run = client.post(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}/run",
        json={"input": {"message": "delete me"}},
    )
    assert run.status_code == 200, run.text

    blocked = client.delete(f"/api/plugins/workflows/definitions/{definition['workflow_id']}")
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["code"] == "workflow_history_exists"

    r = client.delete(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}?purge=true"
    )
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert client.get(f"/api/plugins/workflows/definitions/{definition['workflow_id']}").status_code == 404
    assert client.get("/api/plugins/workflows/definitions").json()["definitions"] == []
    assert client.get("/api/plugins/workflows/executions").json()["executions"] == []


def test_dashboard_run_rejects_missing_required_input_before_creating_execution(client):
    _deploy(client, REQUIRED_RUN_SPEC)

    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_required_run/run",
        json={"input": {}},
    )

    assert r.status_code == 400
    assert "brief is required" in r.json()["detail"]
    assert client.get("/api/plugins/workflows/executions").json()["executions"] == []


def test_dashboard_run_accepts_valid_required_input(client):
    _deploy(client, REQUIRED_RUN_SPEC)

    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_required_run/run",
        json={"input": {"brief": "ship it"}},
    )

    assert r.status_code == 200, r.text
    execution = r.json()["execution"]
    assert execution["status"] == "succeeded"
    assert execution["input"] == {"brief": "ship it"}


def test_dashboard_delete_definition_returns_404_for_missing_workflow(client):
    r = client.delete("/api/plugins/workflows/definitions/missing-workflow")

    assert r.status_code == 404
    assert "workflow definition not found" in r.json()["detail"]


def test_dashboard_deploy_auto_bumps_changed_same_version_specs(client):
    first = _deploy(client, PASS_SPEC)
    assert first["version"] == 1

    changed = copy.deepcopy(PASS_SPEC)
    changed["name"] = "Dashboard Demo Updated"
    changed["nodes"]["start"]["output"] = {"message": "updated"}
    r = client.post("/api/plugins/workflows/definitions/deploy", json={"spec": changed})

    assert r.status_code == 200, r.text
    definition = r.json()["definition"]
    assert definition["workflow_id"] == PASS_SPEC["id"]
    assert definition["version"] == 2
    assert definition["spec"]["version"] == 2
    assert definition["spec"]["nodes"]["start"]["output"] == {"message": "updated"}


def _assert_pass_spec(spec: dict) -> None:
    assert spec["id"] == PASS_SPEC["id"]
    assert spec["name"] == PASS_SPEC["name"]
    assert spec["version"] == PASS_SPEC["version"]
    assert spec["nodes"]["start"]["type"] == "pass"


def test_validate_rejects_unknown_spec_fields_with_suggestion(client):
    typo = copy.deepcopy(PASS_SPEC)
    typo["nodes"]["start"]["outputt"] = {"oops": True}

    r = client.post("/api/plugins/workflows/definitions/validate", json={"spec": typo})

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "unknown field 'outputt' on node 'start'" in detail
    assert "output" in detail


def test_executions_list_supports_limit_and_newest_first(client):
    _deploy(client, PASS_SPEC)
    ids = []
    for _ in range(3):
        r = client.post(
            f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/run", json={"input": {}}
        )
        assert r.status_code == 200, r.text
        ids.append(r.json()["execution"]["execution_id"])

    r = client.get("/api/plugins/workflows/executions?limit=2")

    assert r.status_code == 200, r.text
    executions = r.json()["executions"]
    assert len(executions) == 2
    listed = [e["execution_id"] for e in executions]
    assert set(listed).issubset(set(ids))
    # Newest-first: created_at is non-increasing down the list.
    times = [e["created_at"] for e in executions]
    assert times == sorted(times, reverse=True)


def test_prompt_assistant_drafts_text_prompt(client):
    r = client.post(
        "/api/plugins/workflows/prompt-assistant/draft",
        json={
            "workflow_goal": "Review code changes before merge",
            "node_id": "review",
            "profile": "reviewer",
            "cell_objective": "Review implementation output and decide whether it is approved",
            "available_context": ["${ input.repo }", "${ node.implement.output }"],
            "expected_output": {
                "verdict": "approved|changes_requested",
                "reason": "string",
            },
            "constraints": ["Return JSON only", "Mention required changes if any"],
            "provider": "openai-codex",
            "model": "gpt-5.5",
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_text"]
    assert "Review code changes before merge" in body["prompt_text"]
    assert "${ input.repo }" in body["prompt_text"]
    assert "verdict" in body["prompt_text"]
    assert "openai-codex" in body["prompt_text"]
    assert "gpt-5.5" in body["prompt_text"]
    assert body["result_contract"]["verdict"] == "approved|changes_requested"


def test_agent_routing_options_endpoint_lists_profiles_and_models(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    class Profile:
        def __init__(self, name, provider, model):
            self.name = name
            self.provider = provider
            self.model = model
            self.description = ""
            self.is_default = name == "default"

    profiles_mod = SimpleNamespace(
        list_profiles=lambda: [
            Profile("default", "xiaomi-token-plan", "mimo-vl-7b"),
            Profile("reviewer", "openai-codex", "gpt-5.5"),
        ]
    )
    picker_context = object()

    def build_models_payload(*args, **kwargs):
        assert args == (picker_context,)
        assert kwargs == {
            "include_unconfigured": True,
            "picker_hints": True,
            "canonical_order": True,
            "probe_custom_providers": False,
            "max_models": 500,
        }
        return {
            "provider": "xiaomi-token-plan",
            "model": "mimo-vl-7b",
            "providers": [
                {"slug": "xiaomi-token-plan", "label": "Xiaomi", "models": ["mimo-vl-7b"]},
                {
                    "slug": "openai-codex",
                    "label": "OpenAI Codex",
                    "models": ["gpt-5.5"],
                    "api_key": "SHOULD_NOT_LEAK",
                    "authorization": "SHOULD_NOT_LEAK",
                    "token": "SHOULD_NOT_LEAK",
                    "warning": "paste OPENAI_API_KEY to activate",
                },
            ],
        }

    inventory_mod = SimpleNamespace(
        load_picker_context=lambda: picker_context,
        build_models_payload=build_models_payload,
    )
    monkeypatch.setattr(plugin, "profiles_mod", profiles_mod, raising=False)
    monkeypatch.setattr(plugin, "inventory_mod", inventory_mod, raising=False)

    r = client.get("/api/plugins/workflows/agent-routing-options")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["profiles"][0]["name"] == "default"
    assert body["profiles"][1]["name"] == "reviewer"
    assert body["providers"][1]["slug"] == "openai-codex"
    assert body["providers"][1]["models"] == ["gpt-5.5"]
    text = r.text.lower()
    assert "api_key" not in text
    assert "authorization" not in text
    assert '"token"' not in text
    assert "should_not_leak" not in text


def test_capabilities_endpoint_lists_implemented_and_unsupported_primitives(client):
    r = client.get("/api/plugins/workflows/capabilities")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triggers"]["implemented"] == ["manual", "schedule"]
    assert "webhook" in body["triggers"]["unsupported"]
    assert "agent_task" in body["nodes"]["implemented"]
    assert "send_message" in body["nodes"]["unsupported"]


def test_status_endpoint_reports_dispatcher_config(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(
        plugin,
        "load_config",
        lambda: {"workflow": {"dispatch_in_gateway": False, "tick_interval_seconds": 30}},
    )

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"]["dispatch_in_gateway"] is False
    assert body["dispatcher"]["warning"]
    assert "workflow.dispatch_in_gateway" in body["dispatcher"]["warning"]
    assert "hermes workflow tick" in body["dispatcher"]["warning"]


def test_status_endpoint_defaults_when_workflow_config_missing(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(plugin, "load_config", lambda: {})

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    # dispatch_in_gateway defaults on — a missing workflow section behaves
    # like the shipped default, not like an opt-out.
    assert body["dispatcher"]["dispatch_in_gateway"] is True
    assert body["dispatcher"]["tick_interval_seconds"] == 30.0
    assert body["dispatcher"]["warning"] is None


def test_status_endpoint_handles_non_dict_workflow_config(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(plugin, "load_config", lambda: {"workflow": "not-a-dict"})

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"]["dispatch_in_gateway"] is True
    assert body["dispatcher"]["tick_interval_seconds"] == 30.0
    assert body["dispatcher"]["warning"] is None


def test_status_endpoint_treats_string_false_as_disabled(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(
        plugin,
        "load_config",
        lambda: {"workflow": {"dispatch_in_gateway": "false", "tick_interval_seconds": 9}},
    )

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"]["dispatch_in_gateway"] is False
    assert body["dispatcher"]["tick_interval_seconds"] == 9.0
    assert body["dispatcher"]["warning"]


@pytest.mark.parametrize(
    ("raw_interval", "expected_interval"),
    [
        ("nan", 30.0),
        (["bad"], 30.0),
        (0.25, 1.0),
        ("9", 9.0),
    ],
)
def test_status_endpoint_normalizes_dispatcher_interval(
    client, monkeypatch, raw_interval, expected_interval
):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(
        plugin,
        "load_config",
        lambda: {
            "workflow": {
                "dispatch_in_gateway": True,
                "tick_interval_seconds": raw_interval,
            }
        },
    )

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"]["tick_interval_seconds"] == expected_interval


def test_status_endpoint_omits_warning_when_dispatcher_enabled(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(
        plugin,
        "load_config",
        lambda: {"workflow": {"dispatch_in_gateway": True, "tick_interval_seconds": 12}},
    )

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"] == {
        "dispatch_in_gateway": True,
        "tick_interval_seconds": 12.0,
        "warning": None,
    }


def test_definition_draft_endpoint_returns_validated_spec(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_draft(goal):
        assert goal == "Build demo"
        from hermes_cli.workflows_assistant import parse_assistant_payload

        return parse_assistant_payload(
            {
                "spec": PASS_SPEC,
                "summary": "Drafted dashboard demo.",
                "assumptions": ["Manual trigger."],
                "questions": [],
                "warnings": [],
                "unsupported_requests": [],
            }
        )

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post("/api/plugins/workflows/definitions/draft", json={"goal": "Build demo"})

    assert r.status_code == 200, r.text
    body = r.json()["draft"]
    assert body["spec"]["id"] == PASS_SPEC["id"]
    assert body["summary"] == "Drafted dashboard demo."


def test_dashboard_plain_language_draft_deploy_run_e2e(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin
    from hermes_cli.workflows_assistant import parse_assistant_payload

    def fake_draft(goal):
        assert "plain language" in goal
        return parse_assistant_payload(
            {
                "spec": {
                    "id": "plain_language_demo",
                    "name": "Plain Language Demo",
                    "version": 1,
                    "triggers": [
                        {"type": "manual", "id": "manual", "input": {"topic": ""}}
                    ],
                    "nodes": {
                        "start": {
                            "type": "pass",
                            "output": {"topic": "${ input.topic }"},
                        }
                    },
                },
                "summary": "Drafted from plain language.",
                "assumptions": [],
                "questions": [],
                "warnings": [],
                "unsupported_requests": [],
            }
        )

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post(
        "/api/plugins/workflows/definitions/draft", json={"goal": "plain language demo"}
    )
    assert r.status_code == 200, r.text
    spec = r.json()["draft"]["spec"]

    r = client.post("/api/plugins/workflows/definitions/deploy", json={"spec": spec})
    assert r.status_code == 200, r.text
    assert r.json()["definition"]["workflow_id"] == "plain_language_demo"

    r = client.post(
        "/api/plugins/workflows/definitions/plain_language_demo/run",
        json={"input": {"topic": "ai-first workflows"}},
    )
    assert r.status_code == 200, r.text
    execution_id = r.json()["execution"]["execution_id"]

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}")
    assert r.status_code == 200, r.text
    assert r.json()["execution"]["status"] in {"queued", "succeeded"}

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["node_runs"], list)


def test_definition_draft_endpoint_redacts_unexpected_runtime_errors(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_draft(goal):
        raise RuntimeError("secret provider token abc123")

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post("/api/plugins/workflows/definitions/draft", json={"goal": "Build demo"})

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_runtime_error"
    assert "Check workflow assistant provider/model configuration" in detail["hint"]
    assert "secret" not in r.text
    assert "abc123" not in r.text


def test_definition_draft_endpoint_redacts_common_credential_terms(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_draft(goal):
        raise RuntimeError("auth key jwt credential private value")

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post("/api/plugins/workflows/definitions/draft", json={"goal": "Build demo"})

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_runtime_error"
    assert "auth key" not in detail["message"]
    assert "credential" not in r.text


def test_definition_draft_endpoint_surfaces_non_secret_errors(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_draft(goal):
        raise RuntimeError("No inference provider configured. Run 'hermes model' to choose a provider.")

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post("/api/plugins/workflows/definitions/draft", json={"goal": "Build demo"})

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_runtime_error"
    assert "No inference provider configured" in detail["message"]
    assert "secret" not in r.text


def test_definition_draft_endpoint_returns_validation_hint(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_draft(goal):
        raise plugin.workflows_assistant.AssistantValidationError(
            "assistant draft failed validation (invalid JSON)"
        )

    monkeypatch.setattr(plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft)

    r = client.post("/api/plugins/workflows/definitions/draft", json={"goal": "draft this"})

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_validation_error"
    assert "Advanced YAML" in detail["hint"]
    assert "draft this" not in r.text


def test_definition_refine_endpoint_uses_existing_spec(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    calls = []

    def fake_refine(spec, instruction):
        calls.append((spec.id, instruction))
        from hermes_cli.workflows_assistant import parse_assistant_payload

        return parse_assistant_payload(
            {
                "spec": PASS_SPEC,
                "summary": "Refined spec.",
                "assumptions": [],
                "questions": [],
                "warnings": [],
                "unsupported_requests": [],
            }
        )

    monkeypatch.setattr(plugin.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    r = client.post(
        "/api/plugins/workflows/definitions/refine",
        json={"spec": PASS_SPEC, "instruction": "Rename it"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["draft"]["summary"] == "Refined spec."
    assert calls == [(PASS_SPEC["id"], "Rename it")]


def test_definition_refine_endpoint_uses_deployed_workflow_id(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    _deploy(client, PASS_SPEC)
    calls = []

    def fake_refine(spec, instruction):
        calls.append((spec.id, instruction))
        from hermes_cli.workflows_assistant import parse_assistant_payload

        return parse_assistant_payload(
            {
                "spec": PASS_SPEC,
                "summary": "Refined deployed spec.",
                "assumptions": [],
                "questions": [],
                "warnings": [],
                "unsupported_requests": [],
            }
        )

    monkeypatch.setattr(plugin.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    r = client.post(
        "/api/plugins/workflows/definitions/refine",
        json={
            "workflow_id": PASS_SPEC["id"],
            "version": PASS_SPEC["version"],
            "instruction": "Rename it",
        },
    )

    assert r.status_code == 200, r.text
    assert r.json()["draft"]["summary"] == "Refined deployed spec."
    assert calls == [(PASS_SPEC["id"], "Rename it")]


def test_definition_refine_endpoint_redacts_unexpected_runtime_errors(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_refine(spec, instruction):
        raise RuntimeError("secret provider token abc123")

    monkeypatch.setattr(plugin.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    r = client.post(
        "/api/plugins/workflows/definitions/refine",
        json={"spec": PASS_SPEC, "instruction": "Rename it"},
    )

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_runtime_error"
    assert "Check workflow assistant provider/model configuration" in detail["hint"]
    assert "secret" not in r.text
    assert "abc123" not in r.text


def test_definition_refine_endpoint_returns_validation_hint(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    def fake_refine(spec, instruction):
        raise plugin.workflows_assistant.AssistantValidationError(
            "assistant draft failed validation (unsupported workflow primitive)"
        )

    monkeypatch.setattr(plugin.workflows_assistant, "refine_workflow_with_default_runner", fake_refine)

    r = client.post(
        "/api/plugins/workflows/definitions/refine",
        json={"spec": PASS_SPEC, "instruction": "Rename it"},
    )

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_validation_error"
    assert "Advanced YAML" in detail["hint"]
    assert "Rename it" not in r.text


def test_manifest_points_to_plugin_api():
    manifest_file = PLUGIN_DIR / "manifest.json"
    assert manifest_file.exists(), f"manifest missing: {manifest_file}"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["name"] == "workflows"
    assert manifest["label"] == "Workflows"
    assert manifest["tab"] == {"path": "/workflows", "position": "after:kanban"}
    assert manifest["entry"] == "dist/index.js"
    assert (PLUGIN_DIR / manifest["entry"]).exists()
    assert manifest["css"] == "dist/style.css"
    assert (PLUGIN_DIR / manifest["css"]).exists()
    assert manifest["api"] == "plugin_api.py"


def test_dashboard_bundle_is_syntax_valid_when_node_is_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    bundle = PLUGIN_DIR / "src" / "app.js"
    result = subprocess.run(
        [node, "--check", str(bundle)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_dashboard_bundle_renders_node_runs_and_linked_worker_tasks():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "node-runs" in bundle
    assert "Linked worker task" in bundle
    assert "waiting on agent" in bundle or "waiting_on_agent" in bundle
    assert "renderNodeRuns" in bundle


def test_dashboard_bundle_wires_node_runs_as_execution_drilldown():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    load_node_runs = bundle[
        bundle.index("function loadNodeRuns") : bundle.index("function loadExecution")
    ]
    load_execution = bundle[
        bundle.index("function loadExecution") : bundle.index("function loadDefinitions")
    ]
    render_timeline = bundle[
        bundle.index("function renderTimeline") : bundle.index("function renderSimpleGraph")
    ]

    assert "stateNodeRuns" in bundle
    assert "setNodeRuns" in bundle
    assert 'api("/executions/" + encodeURIComponent(executionId) + "/node-runs")' in load_node_runs
    assert "if (!executionId)" in load_node_runs
    assert "setNodeRuns([])" in load_node_runs
    assert "loadNodeRuns(executionId)" in load_execution
    assert "Promise.all([loadEvents(executionId), loadNodeRuns(executionId)])" in load_execution
    assert "setNodeRuns([])" in load_execution
    assert "renderNodeRuns()" in render_timeline


def test_dashboard_bundle_contains_validation_checklist_and_dispatcher_banner():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "Validation checklist" in bundle
    assert "No unsupported nodes (implemented today)" in bundle
    assert "No unsupported triggers (implemented today)" in bundle
    assert "implemented dashboard/dispatcher readiness" in bundle
    assert "Implemented triggers today" in bundle
    assert "Implemented node types today" in bundle
    assert "renderValidationChecklist" in bundle
    assert "loadWorkflowStatus" in bundle


def test_dashboard_run_inputs_prefer_typed_trigger_input_schema():
    fields = _run_node_input_fields_for_spec(
        {
            "id": "typed_input_demo",
            "name": "Typed Input Demo",
            "version": 1,
            "triggers": [
                {
                    "id": "kickoff",
                    "type": "manual",
                    "input": {"legacy": ""},
                    "input_schema": {
                        "repo_path": {"kind": "repo_path", "label": "Repository path", "required": True},
                        "instructions": {"kind": "prompt", "required": True},
                        "criteria": {"kind": "criteria"},
                        "document": {"kind": "document", "accepts": [".md", ".txt"]},
                    },
                    "intake": {"mode": "continuous", "dedupe_key": "$.input.repo_path"},
                }
            ],
            "nodes": {"start": {"type": "pass"}},
        }
    )

    assert fields == [
        {"name": "repo_path", "kind": "repo_path", "label": "Repository path", "required": True, "description": ""},
        {"name": "instructions", "kind": "prompt", "label": "instructions", "required": True, "description": ""},
        {"name": "criteria", "kind": "criteria", "label": "criteria", "required": False, "description": ""},
        {"name": "document", "kind": "document", "label": "document", "required": False, "description": ""},
    ]


def test_dashboard_feed_input_fields_can_target_continuous_trigger():
    spec = {
        "id": "multi_trigger_demo",
        "name": "Multi Trigger Demo",
        "version": 1,
        "triggers": [
            {
                "id": "manual_once",
                "type": "manual",
                "input_schema": {"manual_only": {"kind": "text"}},
            },
            {
                "id": "feed",
                "type": "manual",
                "input_schema": {"repo_path": {"kind": "repo_path", "required": True}},
                "intake": {"mode": "continuous"},
            },
        ],
        "nodes": {"start": {"type": "pass"}},
    }

    fields = _run_node_input_fields_for_spec(spec, spec["triggers"][1])

    assert fields == [
        {"name": "repo_path", "kind": "repo_path", "label": "repo_path", "required": True, "description": ""},
    ]


def test_dashboard_bundle_contains_feed_controls_and_honest_scalar_scope():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "renderInputFeedPanel" in bundle
    assert "Open Continuous Feed" in bundle
    assert "Add Item To Feed" in bundle
    assert "Phase 1 supports scalar manual and continuous input items" in bundle
    assert "Batch splitting and document uploads are not supported in this release" in bundle
    assert "input.documents" not in bundle
    assert "This workflow has no manual trigger with intake.mode: continuous." in bundle
    assert 'api("/definitions/" + encodeURIComponent(workflowId) + "/input-feeds"' in bundle
    assert 'api("/input-feeds/" + encodeURIComponent(feedId) + "/items"' in bundle


def test_dashboard_validation_checklist_waits_for_parsed_spec_before_showing_failures():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    render_body = bundle[
        bundle.index("function renderValidationChecklist") : bundle.index(
            "function renderAdvancedYaml"
        )
    ]

    assert "Select or validate a workflow to see the checklist." in render_body
    assert "if (!spec)" in render_body


def test_dashboard_initial_load_refreshes_workflow_status_without_waiting_for_it():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    load_status_start = bundle.index("function loadWorkflowStatus")
    load_status_end = bundle.index("function loadWorkflowCapabilities", load_status_start)
    load_status = bundle[load_status_start:load_status_end]
    load_capabilities_start = bundle.index("function loadWorkflowCapabilities")
    load_capabilities_end = bundle.index("function refresh", load_capabilities_start)
    load_capabilities = bundle[load_capabilities_start:load_capabilities_end]
    refresh_start = bundle.index("function refresh(")
    refresh_end = bundle.index("useEffect(function () {", refresh_start)
    refresh_body = bundle[refresh_start:refresh_end]
    promise_all_start = refresh_body.index("Promise.all(")
    promise_all_body = refresh_body[
        promise_all_start : refresh_body.index("])", promise_all_start) + 2
    ]
    effect_end = bundle.index("}, []);", refresh_end)
    effect_body = bundle[refresh_end:effect_end]

    assert "setWorkflowStatus(null)" in load_status
    assert 'api("/capabilities")' in load_capabilities
    assert "setWorkflowCapabilities(null)" in load_capabilities
    assert "loadWorkflowStatus();" in refresh_body
    assert "loadWorkflowCapabilities();" in refresh_body
    assert "Promise.all([loadDefinitions(), loadExecutions(preferExecutionId)])" in refresh_body
    assert "loadWorkflowStatus" not in promise_all_body
    assert "loadWorkflowCapabilities" not in promise_all_body
    assert "refresh(initialExecutionId);" in effect_body


def test_dashboard_bundle_preserves_selected_definition_version():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    helpers = bundle[
        bundle.index("function versionQuery") : bundle.index("function runInputSpec")
    ]
    load_definition = bundle[
        bundle.index("function loadDefinition") : bundle.index("function loadEvents")
    ]
    load_definitions = bundle[
        bundle.index("function loadDefinitions") : bundle.index("function loadExecutions")
    ]
    deploy = bundle[
        bundle.index("function deployDefinition") : bundle.index("function selectedRunVersion")
    ]
    definition_list = bundle[
        bundle.index("function renderSidebar") : bundle.index("function renderBuilderToolbar")
    ]

    assert '"?version=" + encodeURIComponent(value)' in helpers
    assert "function loadDefinition(workflowId, version)" in load_definition
    assert "const previousSelectionKey = definitionSelectionKey(selectedDefinition)" in load_definition
    assert "const nextSelectionKey = definitionSelectionKey(definition)" in load_definition
    assert "if (nextSelectionKey !== previousSelectionKey)" in load_definition
    assert 'api("/definitions/" + encodeURIComponent(workflowId) + versionQuery(version))' in load_definition
    assert "function loadDefinitions(preferId, preferVersion)" in load_definitions
    assert "const currentVersion = selectedDefinition && selectedDefinition.version" in load_definitions
    assert "const matches = rows.filter(function (definition)" in load_definitions
    assert "const match = matches[matches.length - 1]" in load_definitions
    assert "return loadDefinition(nextId, nextVersion)" in load_definitions
    assert "const version = definition.version" in deploy
    assert "return loadDefinitions(id, version)" in deploy
    assert "definitionSelectionKey(definition)" in definition_list
    assert "definitionSelectionKey(selectedDefinition)" in definition_list
    assert "loadDefinition" in definition_list


def test_dashboard_validate_keeps_draft_unrunnable_until_deploy():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    validate = bundle[
        bundle.index("function validateDefinition") : bundle.index("function deployDefinition")
    ]
    topbar = bundle[
        bundle.index("function renderTopBar") : bundle.index("function renderSidebar")
    ]

    assert "setSelectedDefinition" not in validate
    assert "updateEditorText(specToEditorText(definition.spec))" in validate
    assert "var persisted = !!(selectedDefinition && workflowIdForDefinition(selectedDefinition)" in topbar
    assert "setRunPanelOpen(true)" in topbar
    assert "onClick: runWorkflow" not in topbar


def test_dashboard_bundle_runs_selected_or_active_definition_version():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    selected_run_version = bundle[
        bundle.index("function selectedRunVersion") : bundle.index("function runWorkflow")
    ]
    run_workflow = bundle[
        bundle.index("function runWorkflow") : bundle.index("function draftFromGoal")
    ]
    run_form = bundle[
        bundle.index("function runWorkflow") : bundle.index("function draftFromGoal")
    ]

    assert "function selectedRunVersion(workflowId)" in selected_run_version
    assert "return selectedDefinition.version" in selected_run_version
    assert "return versionForSpec(spec)" in selected_run_version
    assert "const runVersion = selectedRunVersion(workflowId)" in run_workflow
    assert '"/run" + versionQuery(runVersion)' in run_workflow
        

def test_dashboard_execution_timeline_warns_when_stalled():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    warn_start = bundle.index("function renderExecutionStallWarning(")
    warn_end = bundle.index("function renderTimeline(", warn_start)
    warn_body = bundle[warn_start:warn_end]

    # Fires only for non-terminal executions with dispatch off.
    assert '"queued"' in warn_body and '"waiting"' in warn_body
    assert "dispatch_in_gateway === true" in warn_body
    assert "will not advance automatically" in warn_body
    # And it is rendered inside the execution timeline, not only the
    # dispatcher tab.
    timeline_start = bundle.index("function renderTimeline(")
    timeline_end = bundle.index("function renderNodeRunPreview(", timeline_start)
    assert "renderExecutionStallWarning()" in bundle[timeline_start:timeline_end]


def test_dashboard_live_refreshes_non_terminal_executions():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "Live-refresh a non-terminal execution" in bundle
    assert "setInterval" in bundle
    assert "clearInterval(timer)" in bundle


def test_dashboard_event_status_maps_only_emitted_kinds():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    status_start = bundle.index("function eventStatus(")
    status_end = bundle.index("function statusByNode(", status_start)
    status_body = bundle[status_start:status_end]

    # These kinds are actually emitted by the dispatcher today.
    assert "node_succeeded" in status_body
    assert "node_failed" in status_body
    assert "execution_blocked" in status_body
    assert "execution_waiting" in status_body
    # Dead mappings for never-emitted kinds must not come back.
    assert "node_started" not in status_body
    assert "node_running" not in status_body


def _dashboard_helper_js() -> str:
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    kind_start = bundle.index("const NODE_KIND_LIST")
    kind_end = bundle.index("const EXAMPLE_DEFINITION", kind_start)
    start = bundle.index("function asArray")
    end = bundle.index("function statusClass", start)
    # Include editor helper functions used by UI-only builder tests.
    editor_start = bundle.index("function cleanedNodeForSpec")
    editor_end = bundle.index("function WorkflowsPage", editor_start)
    return (
        bundle[kind_start:kind_end]
        + "\n"
        + bundle[start:end]
        + "\n"
        + bundle[editor_start:editor_end]
    )


def _run_dashboard_function(function_name: str, args):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = (
        _dashboard_helper_js()
        + "\nconst args = "
        + json.dumps(args)
        + ";\nconsole.log(JSON.stringify("
        + function_name
        + ".apply(null, args)));\n"
    )
    result = subprocess.run(
        [node, "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _run_dashboard_function_error(function_name: str, args) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = (
        _dashboard_helper_js()
        + "\nconst args = "
        + json.dumps(args)
        + ";\ntry {\n"
        + "  console.log(JSON.stringify({ ok: true, value: "
        + function_name
        + ".apply(null, args) }));\n"
        + "} catch (err) {\n"
        + "  console.log(JSON.stringify({ ok: false, error: String((err && err.message) || err) }));\n"
        + "  process.exitCode = 1;\n"
        + "}\n"
    )
    result = subprocess.run(
        [node, "-e", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    body = json.loads(result.stdout)
    return body["error"]


def _run_node_summary_rows(spec):
    return _run_dashboard_function("nodeSummaryRows", [spec])


def _run_node_input_fields_for_spec(spec, preferred_trigger=None):
    args = [spec]
    if preferred_trigger is not None:
        args.append(preferred_trigger)
    return _run_dashboard_function("inputFieldsForSpec", args)


def _run_validation_checklist(spec, capabilities=None):
    args = [spec]
    if capabilities is not None:
        args.append(capabilities)
    return _run_dashboard_function("validationChecklist", args)


def test_dashboard_ui_builder_helpers_create_add_connect_delete_cells_without_json():
    spec = _run_dashboard_function("newWorkflowSpec", ["Blank Slate Demo"])
    assert spec["id"] == "blank_slate_demo"
    assert spec["triggers"] == [{"id": "manual", "type": "manual"}]
    assert spec["nodes"] == {}

    spec = _run_dashboard_function("addSpecNodeAfter", [spec, "review", "agent_task", ""])
    assert spec["nodes"]["review"]["type"] == "agent_task"
    assert spec["nodes"]["review"]["profile"] == "default"
    assert spec["nodes"]["review"]["prompt"]
    assert spec["nodes"]["review"]["result_contract"]
    assert spec["edges"] == []

    spec = _run_dashboard_function("addSpecNodeAfter", [spec, "done", "pass", "review"])
    assert spec["nodes"]["done"]["type"] == "pass"
    assert {"from": "review", "to": "done"} in spec["edges"]

    renamed = _run_dashboard_function("upsertSpecNode", [spec, "done", {"id": "finished", "type": "pass"}])
    assert "finished" in renamed["nodes"]
    assert "done" not in renamed["nodes"]
    assert {"from": "review", "to": "finished"} in renamed["edges"]
    assert all(item["ok"] for item in _run_validation_checklist(renamed))

    spec = _run_dashboard_function("removeSpecNode", [spec, "review"])
    assert "review" not in spec["nodes"]
    assert spec["edges"] == []


def test_dashboard_ui_builder_helpers_add_schedule_trigger_and_switch_branch_edges():
    spec = _run_dashboard_function("newWorkflowSpec", ["Branch Demo"])
    spec = _run_dashboard_function("addSpecTrigger", [spec, "weekday", "schedule", "0 9 * * *"])
    assert {"id": "weekday", "type": "schedule", "schedule": "0 9 * * *"} in spec["triggers"]

    spec = _run_dashboard_function("addSpecNodeAfter", [spec, "route", "switch", ""])
    spec = _run_dashboard_function("addSpecNodeAfter", [spec, "approved", "pass", "route.default"])
    spec = _run_dashboard_function("addSwitchCaseToSpec", [spec, "route", "approved", "$.input.status", "approved"])
    spec = _run_dashboard_function("upsertSpecEdge", [spec, "route.approved", "approved"])
    assert {"from": "route.default", "to": "approved"} in spec["edges"]
    assert {"from": "route.approved", "to": "approved"} in spec["edges"]
    assert {"name": "approved", "when": {"op": "eq", "left": {"path": "$.input.status"}, "right": "approved"}} in spec["nodes"]["route"]["cases"]
    checklist = _run_validation_checklist(spec)
    assert all(item["ok"] for item in checklist), checklist


def test_dashboard_cleaned_node_removes_agent_only_fields_when_switching_types():
    clean = _run_dashboard_function(
        "cleanedNodeForSpec",
        [
            {
                "id": "review",
                "type": "pass",
                "profile": "reviewer",
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "result_contract": {"ok": "boolean"},
                "prompt": "notes",
            }
        ],
    )
    assert clean == {"id": "review", "type": "pass", "prompt": "notes"}


def test_dashboard_validation_checklist_accepts_minimal_valid_spec():
    checklist = _run_validation_checklist(
        {"id": "ok", "name": "OK", "version": 1, "nodes": {"start": {"type": "pass"}}}
    )

    assert all(item["ok"] for item in checklist)


def test_dashboard_validation_checklist_rejects_invalid_workflow_id_and_non_string_types():
    bad_specs = [
        {"id": "Bad ID", "name": "Bad", "version": 1, "nodes": {"start": {"type": "pass"}}},
        {"id": "bad", "name": {}, "version": 1, "nodes": {"start": {"type": "pass"}}},
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "triggers": [{"type": ["manual"]}],
            "nodes": {"start": {"type": "pass"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "triggers": [{"type": "manual "}],
            "nodes": {"start": {"type": "pass"}},
        },
        {"id": "bad", "name": "Bad", "version": 1, "nodes": {"start": {"type": ["pass"]}}},
        {"id": "bad", "name": "Bad", "version": 1, "nodes": {"start": {"type": "pass "}}},
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"review": {"type": ["agent_task"], "profile": "worker", "prompt": "Do it"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"review": {"type": "agent_task ", "profile": "worker", "prompt": "Do it"}},
        },
    ]

    for spec in bad_specs:
        checklist = _run_validation_checklist(spec)
        assert not all(item["ok"] for item in checklist)

    checklist = _run_validation_checklist(bad_specs[0])
    assert next(item for item in checklist if item["label"] == "Workflow id set")["ok"] is False

    checklist = _run_validation_checklist(bad_specs[1])
    assert next(item for item in checklist if item["label"] == "Workflow name set")["ok"] is False

    for spec in bad_specs[2:4]:
        checklist = _run_validation_checklist(spec)
        assert next(
            item
            for item in checklist
            if item["label"] == "No unsupported triggers (implemented today)"
        )["ok"] is False

    for spec in bad_specs[4:]:
        checklist = _run_validation_checklist(spec)
        assert next(
            item
            for item in checklist
            if item["label"] == "No unsupported nodes (implemented today)"
        )["ok"] is False


def test_dashboard_validation_checklist_requires_name_version_and_known_node_type():
    checklist = _run_validation_checklist({"id": "bad", "nodes": {"x": {"type": "bogus"}}})

    assert next(item for item in checklist if item["label"] == "Workflow name set")["ok"] is False
    assert next(item for item in checklist if item["label"] == "Version set")["ok"] is False
    assert next(
        item for item in checklist if item["label"] == "No unsupported nodes (implemented today)"
    )["ok"] is False


def test_dashboard_validation_checklist_flags_version_trigger_and_edges():
    checklist = _run_validation_checklist(
        {
            "id": "bad",
            "name": "Bad",
            "version": 0,
            "triggers": [{"type": "webhook"}],
            "nodes": {"start": {"type": "pass"}},
            "edges": [{"from": "missing", "to": "start"}],
        }
    )

    assert next(item for item in checklist if item["label"] == "Version set")["ok"] is False
    assert next(
        item
        for item in checklist
        if item["label"] == "No unsupported triggers (implemented today)"
    )["ok"] is False
    assert next(item for item in checklist if item["label"] == "Edges refer to known nodes")["ok"] is False


def test_dashboard_validation_checklist_flags_backend_graph_rule_failures():
    bad_specs = [
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "triggers": [{"type": "schedule"}],
            "nodes": {"start": {"type": "pass"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "triggers": [{"type": "schedule", "cron": ["* * * * *"]}],
            "nodes": {"start": {"type": "pass"}},
        },
        {"id": "bad", "name": "Bad", "version": 1, "nodes": {"route": {"type": "switch"}}},
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"route": {"type": "switch", "default": "missing"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"start": {"type": "pass", "catch": "start"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"start": {"type": "pass", "catch": "missing"}},
        },
    ]
    for spec in bad_specs:
        checklist = _run_validation_checklist(spec)
        assert next(item for item in checklist if item["label"] == "Graph rules pass")["ok"] is False


def test_dashboard_validation_checklist_accepts_backend_graph_rules():
    good_specs = [
        {
            "id": "ok",
            "name": "OK",
            "version": 1,
            "triggers": [{"type": "schedule", "cron": "* * * * *"}],
            "nodes": {"start": {"type": "pass"}},
        },
        {
            "id": "ok",
            "name": "OK",
            "version": 1,
            "nodes": {"route": {"type": "switch"}, "done": {"type": "pass"}},
            "edges": [{"from": "route.case1", "to": "done"}],
        },
    ]
    for spec in good_specs:
        checklist = _run_validation_checklist(spec)
        assert next(item for item in checklist if item["label"] == "Graph rules pass")["ok"] is True


def test_dashboard_validation_checklist_rejects_invalid_spec_shapes_not_aliases():
    invalid_specs = [
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "triggers": [{"trigger_type": "manual"}],
            "nodes": {"start": {"type": "pass"}},
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"start": {"type": "pass"}, "done": {"type": "pass"}},
            "edges": [{"source": "start", "target": "done"}],
        },
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {"start": {"type": "pass"}, "done": {"type": "pass"}},
            "edges": [{"from": "start", "to": "done.port"}],
        },
        {"id": "bad", "name": "Bad", "version": 1, "nodes": {"Bad ID": {"type": "pass"}}},
    ]
    for spec in invalid_specs:
        checklist = _run_validation_checklist(spec)
        assert not all(item["ok"] for item in checklist)

    checklist = _run_validation_checklist(invalid_specs[0])
    assert next(
        item
        for item in checklist
        if item["label"] == "No unsupported triggers (implemented today)"
    )["ok"] is False

    checklist = _run_validation_checklist(invalid_specs[1])
    assert next(item for item in checklist if item["label"] == "Edges refer to known nodes")["ok"] is False

    checklist = _run_validation_checklist(invalid_specs[3])
    assert next(item for item in checklist if item["label"] == "Node ids are valid")["ok"] is False


def test_dashboard_validation_checklist_accepts_valid_switch_branch_edge():
    checklist = _run_validation_checklist(
        {
            "id": "ok",
            "name": "OK",
            "version": 1,
            "nodes": {
                "route": {"type": "switch", "default": "done"},
                "done": {"type": "pass"},
            },
            "edges": [{"from": "route.case1", "to": "done"}],
        }
    )

    assert next(item for item in checklist if item["label"] == "Edges refer to known nodes")["ok"] is True


def test_dashboard_validation_checklist_uses_loaded_capabilities():
    caps = {
        "triggers": {"implemented": ["manual", "webhook"]},
        "nodes": {"implemented": ["pass", "send_message"]},
    }
    checklist = _run_dashboard_function(
        "validationChecklist",
        [
            {
                "id": "ok",
                "name": "OK",
                "version": 1,
                "triggers": [{"type": "webhook"}],
                "nodes": {"send": {"type": "send_message"}},
            },
            caps,
        ],
    )

    assert next(
        item
        for item in checklist
        if item["label"] == "No unsupported triggers (implemented today)"
    )["ok"] is True
    assert next(
        item
        for item in checklist
        if item["label"] == "No unsupported nodes (implemented today)"
    )["ok"] is True


def test_dashboard_validation_checklist_flags_unsupported_nodes():
    for node_type in ["send_message", "subworkflow"]:
        checklist = _run_validation_checklist(
            {"id": "bad", "nodes": {"send": {"type": node_type}}}
        )

        unsupported = next(
            item
            for item in checklist
            if item["label"] == "No unsupported nodes (implemented today)"
        )
        assert unsupported["ok"] is False


def test_dashboard_validation_checklist_rejects_invalid_node_shapes_and_types():
    checklist = _run_validation_checklist(
        {"id": "bad", "name": "Bad", "version": 1, "nodes": [{"type": "pass"}]}
    )
    assert next(item for item in checklist if item["label"] == "Node definitions are objects")["ok"] is False

    for node in [{}, {"type": "trigger"}, {"type": "bogus"}]:
        checklist = _run_validation_checklist(
            {"id": "bad", "name": "Bad", "version": 1, "nodes": {"x": node}}
        )
        assert next(
            item
            for item in checklist
            if item["label"] == "No unsupported nodes (implemented today)"
        )["ok"] is False


def test_dashboard_validation_checklist_flags_agent_cells_missing_profile_or_prompt():
    checklist = _run_validation_checklist(
        {"id": "bad", "nodes": {"review": {"type": "agent_task", "profile": ""}}}
    )

    agent = next(item for item in checklist if item["label"] == "Agent cells have profile and prompt")
    assert agent["ok"] is False


def test_dashboard_validation_checklist_rejects_non_string_agent_profile():
    checklist = _run_validation_checklist(
        {
            "id": "bad",
            "name": "Bad",
            "version": 1,
            "nodes": {
                "a": {"type": "agent_task", "profile": {}, "prompt": "Do it"}
            },
        }
    )

    assert next(item for item in checklist if "Agent cells" in item["label"])["ok"] is False


def test_dashboard_validation_checklist_rejects_blank_or_empty_agent_prompts():
    for prompt in ["   ", [], {}]:
        checklist = _run_validation_checklist(
            {
                "id": "bad",
                "name": "Bad",
                "version": 1,
                "nodes": {
                    "a": {"type": "agent_task", "profile": "worker", "prompt": prompt}
                },
            }
        )

        agent = next(item for item in checklist if item["label"] == "Agent cells have profile and prompt")
        assert agent["ok"] is False


def test_dashboard_node_summary_rows_handles_real_workflow_shapes():
    rows = _run_node_summary_rows(
        {
            "nodes": {
                "route": {
                    "type": "switch",
                    "prompt": {"task": "Choose path"},
                    "default": "low",
                    "catch": "pause",
                },
                "high": {
                    "type": "agent_task",
                    "profile": "reviewer",
                    "prompt": [{"objective": "Review output"}],
                },
                "low": {"type": "pass"},
                "pause": {"type": "wait"},
            },
            "edges": [
                {"from_": "route.high", "to": "high"},
                {"from": "route.low", "to": "low"},
            ],
        }
    )

    route = next(row for row in rows if row["id"] == "route")
    assert route["objective"] == "Choose path"
    assert "high → high" in route["next"]
    assert "low → low" in route["next"]
    assert "default → low" in route["next"]
    assert "catch → pause" in route["next"]
    high = next(row for row in rows if row["id"] == "high")
    assert high["objective"] == "Review output"


def test_dashboard_input_fields_for_spec_uses_trigger_input_and_templates():
    assert _run_node_input_fields_for_spec(
        {"triggers": [{"input": {"topic": "", "count": 0}}]}
    ) == [{"name": "topic", "kind": "text"}, {"name": "count", "kind": "number"}]

    assert _run_node_input_fields_for_spec(
        {"nodes": {"start": {"prompt": "Summarize ${ input.topic }"}}}
    ) == [{"name": "topic", "kind": "text"}]


def test_dashboard_input_fields_detects_jsonpath_input_references():
    assert _run_node_input_fields_for_spec(
        {
            "nodes": {
                "route": {
                    "type": "switch",
                    "cases": [
                        {"path": "$.input.side", "equals": "left", "then": "left"},
                        {
                            "condition": {"path": "$.input.min_score", "gte": 10},
                            "then": "high",
                        },
                    ],
                }
            }
        }
    ) == [{"name": "min_score", "kind": "text"}, {"name": "side", "kind": "text"}]


def test_dashboard_input_fields_mark_nested_input_references_as_json():
    assert _run_node_input_fields_for_spec(
        {
            "nodes": {
                "start": {"prompt": "Use ${ input.user.name } and ${ input.topic }"}
            },
            "edges": [{"from": "start", "to": "done", "condition": "$.input.user.age"}],
        }
    ) == [{"name": "topic", "kind": "text"}, {"name": "user", "kind": "json"}]


def test_dashboard_input_fields_mark_array_index_paths_as_json():
    assert _run_node_input_fields_for_spec(
        {
            "nodes": {"start": {"prompt": "Use ${ input.items[0].name }"}},
            "edges": [{"from": "start", "to": "done", "condition": "$.input.users[0].age"}],
        }
    ) == [{"name": "items", "kind": "json"}, {"name": "users", "kind": "json"}]


def test_dashboard_input_fields_prefers_manual_trigger_input():
    assert _run_node_input_fields_for_spec(
        {
            "triggers": [
                {"type": "schedule", "input": {"cron": ""}},
                {"type": "manual", "input": {"topic": "", "count": 0}},
            ]
        }
    ) == [{"name": "topic", "kind": "text"}, {"name": "count", "kind": "number"}]


def test_dashboard_input_fields_ignores_malformed_triggers_before_manual_input():
    assert _run_node_input_fields_for_spec(
        {
            "triggers": [
                None,
                "bad",
                {"type": "manual", "input": {"amount": {"type": "float"}}},
            ]
        }
    ) == [{"name": "amount", "kind": "number"}]


def test_dashboard_input_fields_treats_properties_as_field_unless_schema_object():
    assert _run_node_input_fields_for_spec(
        {"triggers": [{"type": "manual", "input": {"properties": "literal", "topic": ""}}]}
    ) == [{"name": "properties", "kind": "text"}, {"name": "topic", "kind": "text"}]

    assert _run_node_input_fields_for_spec(
        {
            "triggers": [
                {
                    "type": "manual",
                    "input": {"type": "object", "properties": {"amount": {"type": "number"}}},
                }
            ]
        }
    ) == [{"name": "amount", "kind": "number"}]


def test_dashboard_input_fields_ignores_incomplete_object_schema_metadata():
    assert (
        _run_node_input_fields_for_spec(
            {
                "triggers": [
                    {
                        "type": "manual",
                        "input": {
                            "type": "object",
                            "required": ["topic"],
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        )
        == []
    )


def test_dashboard_input_fields_keeps_plain_input_field_named_type():
    assert _run_node_input_fields_for_spec(
        {"triggers": [{"type": "manual", "input": {"type": "object", "topic": ""}}]}
    ) == [{"name": "type", "kind": "text"}, {"name": "topic", "kind": "text"}]


def test_dashboard_input_fields_treats_literal_numbers_as_number_not_integer():
    assert _run_node_input_fields_for_spec(
        {"triggers": [{"type": "manual", "input": {"threshold": 1}}]}
    ) == [{"name": "threshold", "kind": "number"}]

    assert _run_dashboard_function(
        "inputObjectForFields",
        [[{"name": "threshold", "kind": "number"}], {"threshold": "1.5"}],
    ) == {"threshold": 1.5}


def test_dashboard_input_fields_preserves_integer_boolean_and_json_kinds():
    assert _run_node_input_fields_for_spec(
        {
            "triggers": [
                {
                    "type": "manual",
                    "input": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                            "payload": {"type": "object"},
                            "tags": {"type": "array"},
                        },
                    },
                }
            ]
        }
    ) == [
        {"name": "count", "kind": "integer"},
        {"name": "enabled", "kind": "boolean"},
        {"name": "payload", "kind": "json"},
        {"name": "tags", "kind": "json"},
    ]


def test_dashboard_input_object_for_fields_filters_stale_values_and_converts_numbers():
    result = _run_dashboard_function(
        "inputObjectForFields",
        [
            [
                {"name": "topic", "kind": "text"},
                {"name": "count", "kind": "number"},
                {"name": "blank_count", "kind": "number"},
                {"name": "space_count", "kind": "number"},
            ],
            {
                "topic": "release",
                "count": "3",
                "blank_count": "",
                "space_count": "   ",
                "stale": "do-not-send",
            },
        ],
    )

    assert result == {"topic": "release", "count": 3}

    assert _run_dashboard_function(
        "inputObjectForFields",
        [[{"name": "count", "kind": "number"}], {"count": "1.5"}],
    ) == {"count": 1.5}

    assert _run_dashboard_function(
        "inputObjectForFields",
        [[{"name": "count", "kind": "number"}], {"count": "0.0"}],
    ) == {"count": 0}


def test_dashboard_input_object_for_fields_preserves_typed_values():
    assert _run_dashboard_function(
        "inputObjectForFields",
        [
            [
                {"name": "count", "kind": "integer"},
                {"name": "enabled", "kind": "boolean"},
                {"name": "payload", "kind": "json"},
            ],
            {"count": "2", "enabled": "false", "payload": '{"ok":true}'},
        ],
    ) == {"count": 2, "enabled": False, "payload": {"ok": True}}


def test_dashboard_input_object_for_nested_json_fallback_field_submits_object():
    assert _run_dashboard_function(
        "inputObjectForFields",
        [[{"name": "user", "kind": "json"}], {"user": '{"name":"Alice"}'}],
    ) == {"user": {"name": "Alice"}}


def test_dashboard_input_object_for_fields_rejects_invalid_or_non_finite_numbers():
    for raw in ["abc", "0x10", "0b10", "1e309", "NaN", "Infinity", "9007199254740993"]:
        error = _run_dashboard_function_error(
            "inputObjectForFields",
            [[{"name": "count", "kind": "number"}], {"count": raw}],
        )
        assert "Invalid number for input field count" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "count", "kind": "number"}], {"count": "1e-1000"}],
    )
    assert "Invalid number for input field count" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "count", "kind": "integer"}], {"count": "1e-1000"}],
    )
    assert "Invalid integer for input field count" in error


def test_dashboard_input_object_for_fields_rejects_invalid_typed_values():
    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "count", "kind": "integer"}], {"count": "1.5"}],
    )
    assert "Invalid integer for input field count" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "count", "kind": "integer"}], {"count": "9007199254740991.1"}],
    )
    assert "Invalid integer for input field count" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "count", "kind": "number"}], {"count": "9007199254740991.1"}],
    )
    assert "Invalid number for input field count" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "enabled", "kind": "boolean"}], {"enabled": "maybe"}],
    )
    assert "Invalid boolean" in error

    error = _run_dashboard_function_error(
        "inputObjectForFields",
        [[{"name": "payload", "kind": "json"}], {"payload": "not-json"}],
    )
    assert "Invalid JSON" in error


def test_dashboard_bundle_registers_plugin_without_build_scaffolding():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    assert "window.__HERMES_PLUGIN_SDK__" in bundle
    assert "window.__HERMES_PLUGINS__" in bundle
    assert 'REG.register("workflows"' in bundle
    # app.js is a Vite entry module: `import` is required to wire its graph/api
    # adapters; `export` would change it into a library, which it isn't.
    assert not any(
        line.lstrip().startswith("export ") for line in bundle.splitlines()
    )
    assert "__webpack_require__" not in bundle


def test_dashboard_bundle_uses_generated_input_form_for_runs():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "renderRunStartPanel" in bundle
    assert "Start Workflow Run" in bundle
    assert "Start Run" in bundle
    assert "No start input fields are configured" in bundle
    assert "inputFieldValues" in bundle
    assert "Manual run form" not in bundle


def test_dashboard_topbar_run_opens_start_panel_instead_of_running_silently():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    topbar_pos = bundle.index("function renderTopBar")
    topbar_body = bundle[topbar_pos : bundle.index("function renderSidebar", topbar_pos)]

    assert "setRunPanelOpen(true)" in topbar_body
    assert "onClick: runWorkflow" not in topbar_body


def test_dashboard_run_start_panel_builds_typed_inputs_from_trigger_schema():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    panel_pos = bundle.index("function renderRunInputField")
    panel_body = bundle[panel_pos : bundle.index("function renderBottomPanel", panel_pos)]

    assert "inputFieldsForSpec(runInputSpec())" in panel_body
    assert "showAdvancedInputJson" in panel_body
    assert "runInputText" in panel_body
    assert "inputFieldValues" in panel_body
    assert "runWorkflow" in panel_body


def test_dashboard_run_workflow_uses_form_values_unless_advanced_json():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    run_pos = bundle.index("function runWorkflow")
    run_body = bundle[run_pos : bundle.index("function draftFromGoal", run_pos)]
    render_pos = bundle.index("function runWorkflow")
    render_body = bundle[render_pos : bundle.index("function draftFromGoal", render_pos)]

    assert "function runInputSpec" in bundle
    assert "showAdvancedInputJson" in run_body
    assert "inputFieldValues" in run_body
    assert 'JSON.parse(runInputText || "{}")' in run_body
    assert "inputFieldsForSpec(runInputSpec())" in run_body
    assert "body: JSON.stringify({ input: input })" in run_body
    assert "input_json" not in run_body
    assert "inputFieldsForSpec(spec)" in render_body or "runInputSpec()" in render_body


def test_dashboard_bundle_clears_run_input_values_when_active_spec_changes():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    load_pos = bundle.index("function loadDefinition")
    load_body = bundle[load_pos : bundle.index("function loadEvents", load_pos)]
    accept_pos = bundle.index("function acceptDraftCandidate")
    accept_body = bundle[accept_pos : bundle.index("function rejectDraftCandidate", accept_pos)]
    import_pos = bundle.index("function importDefinitionFile")
    import_body = bundle[import_pos : bundle.index("function exportYAML", import_pos)]

    def assert_resets_advanced_input_state(body):
        reset_pos = body.index("setInputFieldValues({})")
        assert "setShowAdvancedInputJson(false)" in body
        assert 'setRunInputText("{}")' in body
        assert reset_pos < body.index("setShowAdvancedInputJson(false)")
        assert reset_pos < body.index('setRunInputText("{}")')

    assert load_body.index("setDraftSpec(") < load_body.index("setInputFieldValues({})")
    assert accept_body.index("setDraftSpec(draft.spec)") < accept_body.index("setInputFieldValues({})")
    assert import_body.index("reader.onload = function") < import_body.index(
        "setInputFieldValues({})"
    )
    for body in (load_body, accept_body, import_body):
        assert_resets_advanced_input_state(body)


def test_dashboard_bundle_clears_run_input_values_when_advanced_yaml_changes():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    advanced = bundle[
        bundle.index("function renderAdvancedYaml") : bundle.index(
            "function renderTimeline"
        )
    ]
    change = advanced[
        advanced.index("onChange: function") : advanced.index(
            "}),", advanced.index("onChange: function")
        )
    ]

    for marker in [
        "setInputFieldValues({})",
        "setShowAdvancedInputJson(false)",
        'setRunInputText("{}")',
    ]:
        assert marker in change


def test_dashboard_bundle_is_prompt_first_not_yaml_first():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    render_start = bundle.index('return h("div", { className: "hermes-workflows" }')
    render_tree = bundle[render_start:]

    assert "New workflow" in bundle
    assert "Generate From Prompt" in bundle
    assert "Advanced YAML" in bundle or "YAML" in bundle
    assert "Validate / deploy definition" not in bundle
    # In the 3-zone layout, goal builder is in the sidebar which renders before the canvas
    assert render_tree.index("renderSidebar()") < render_tree.index(
        "renderBottomPanel()"
    )
    assert bundle.index("New workflow") < bundle.index("renderBuilderToolbar")


def test_dashboard_bundle_contains_draft_review_and_refine_ui():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "refineWorkflow" in bundle
    assert "Refine" in bundle


def test_dashboard_bundle_wires_draft_refine_before_advanced_yaml():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    render_start = bundle.index('return h("div", { className: "hermes-workflows" }')
    render_tree = bundle[render_start:]

    assert "/definitions/refine" in bundle
    assert 'setRefineText("")' in bundle
    assert "nodeSummaryRows" in bundle
    # In the 3-zone layout, draft review is in the bottom panel which renders before advanced YAML
    assert render_tree.index("renderBottomPanel()") < render_tree.index(
        "renderAdvancedYaml()"
    )


def test_dashboard_bundle_draft_review_labels_branch_and_failure_targets():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    rows_pos = bundle.index("function nodeSummaryRows")
    next_function_pos = bundle.index("function statusClass", rows_pos)
    rows_body = bundle[rows_pos:next_function_pos]

    for marker in ['"default → "', '"catch → "', "edge.label", "edge.condition", "parts[1]"]:
        assert marker in rows_body


def test_dashboard_bundle_refine_clears_stale_state_before_validation_and_requires_spec():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    refine_pos = bundle.index("function refineWorkflow")
    next_function_pos = bundle.index("function acceptDraftCandidate", refine_pos)
    refine_body = bundle[refine_pos:next_function_pos]
    early_return_pos = refine_body.index("if (!instruction || !spec)")

    assert refine_body.index('setStatus("")') < early_return_pos
    assert refine_body.index("setDraftResult(null)") < early_return_pos
    assert "Refine response did not include a workflow spec." in refine_body
    assert 'setCandidateSource("refine")' in refine_body
    assert "Review changes and Accept or Reject" in refine_body


def test_dashboard_bundle_syncs_editor_when_definition_is_selected():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    load_definition_pos = bundle.index("function loadDefinition")
    next_function_pos = bundle.index("function loadEvents", load_definition_pos)
    load_definition_body = bundle[load_definition_pos:next_function_pos]

    # ponytail: editor sync now checks dirty state before overwriting;
    # the guard is in the same function, the update call is conditional.
    assert "isDraftDirty" in load_definition_body
    assert "specToEditorText" in load_definition_body
    assert "setSavedDraft" in load_definition_body


def test_dashboard_bundle_clears_stale_draft_state_before_empty_goal_error():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    draft_pos = bundle.index("function draftFromGoal")
    next_function_pos = bundle.index("function importDefinitionFile", draft_pos)
    draft_body = bundle[draft_pos:next_function_pos]
    empty_goal_pos = draft_body.index("if (!goal)")

    assert draft_body.index('setStatus("")') < empty_goal_pos
    assert draft_body.index("setDraftResult(null)") < empty_goal_pos
    assert "Describe what you want the workflow to automate." in draft_body


def test_dashboard_bundle_resets_stale_selection_after_goal_draft():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    draft_pos = bundle.index("function draftFromGoal")
    next_function_pos = bundle.index("function refineWorkflow", draft_pos)
    draft_body = bundle[draft_pos:next_function_pos]

    # ponytail: draftFromGoal now stores the AI result as a candidate;
    # the actual working-draft reset (setDraftSpec, setSelectedDefinition, etc.)
    # happens in acceptDraftCandidate. Verify the candidate setup here.
    for marker in [
        'setCandidateSource("generate")',
        "setDraftResult(draft)",
        "Review and Accept or Reject",
    ]:
        assert marker in draft_body
    assert 'aria-label' in bundle and 'Describe workflow goal' in bundle


def test_dashboard_bundle_keeps_yaml_as_advanced_escape_hatch():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    advanced_pos = bundle.index("function renderAdvancedYaml")
    advanced_body = bundle[advanced_pos : bundle.index("function renderTimeline", advanced_pos)]

    assert "showAdvancedYaml" in bundle
    for marker in ["Validate", "Deploy", "Import YAML", "Export YAML", "Copy YAML"]:
        assert marker in advanced_body


def test_web_plugin_sdk_exposes_react_flow_to_static_plugins():
    package_json = json.loads((REPO_ROOT / "web" / "package.json").read_text())
    assert "@xyflow/react" in package_json["dependencies"]

    registry = (REPO_ROOT / "web" / "src" / "plugins" / "registry.ts").read_text(
        encoding="utf-8"
    )
    sdk_types = (REPO_ROOT / "web" / "src" / "plugins" / "sdk.d.ts").read_text(
        encoding="utf-8"
    )

    assert 'from "@xyflow/react"' in registry
    assert 'import "@xyflow/react/dist/style.css"' in registry
    assert "ReactFlow:" in registry
    assert "reactFlow:" in registry
    for marker in [
        "ReactFlowProvider",
        "Background",
        "Controls",
        "MiniMap",
        "Handle",
        "Position",
        "MarkerType",
        "addEdge",
        "applyNodeChanges",
        "applyEdgeChanges",
    ]:
        assert marker in registry
        assert marker in sdk_types


def test_dashboard_bundle_selects_execution_from_url_query():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    assert "URLSearchParams" in bundle
    assert "location.search" in bundle
    assert 'get("execution")' in bundle or "get('execution')" in bundle
    assert "refresh(initialExecutionId)" in bundle or "loadExecutions(initialExecutionId)" in bundle


def test_dashboard_bundle_contains_visual_editor_markers():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    assert "SDK.ReactFlow" in bundle or "SDK.reactFlow" in bundle
    assert "ReactFlowProvider" in bundle
    for marker in ["Background", "Controls", "MiniMap", "Handle", "Position"]:
        assert marker in bundle
    for marker in ["upsertSpecEdge", "Connection added to workflow draft"]:
        assert marker in bundle

    for node_type in [
        "trigger",
        "pass",
        "switch",
        "agent_task",
        "wait",
        "parallel",
        "join",
        "fail",
    ]:
        assert node_type in bundle

    assert ".split(\".\")" in bundle or ".split('.')" in bundle
    for marker in [
        "selectedNode",
        "applyNodeJson",
        "draftSpec",
        "setDraftSpec(null)",
        "Validate the YAML draft before converting",
        "Import YAML",
        "Export YAML",
        ".yaml",
        "statusByNode",
        "node_succeeded",
        "node_failed",
        "execution_waiting",
    ]:
        assert marker in bundle


def test_dashboard_bundle_contains_text_first_agent_cell_editor_markers():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    for marker in [
        "renderAgentTaskInspector",
        "Agent cell prompt",
        "Prompt assistant",
        "Advanced JSON",
        "applyAgentCellForm",
        "renderInspectorForType",
        "promptText",
        "resultContractText",
        "/prompt-assistant/draft",
        "draftPromptWithAssistant",
    ]:
        assert marker in bundle

    assert "Apply node JSON" in bundle  # still available only as advanced escape hatch


def test_dashboard_bundle_contains_workflow_mvp_api_and_ui_markers():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    for marker in [
        'const API = "/api/plugins/workflows"',
        "/api/plugins/workflows/definitions",
        "/definitions/validate",
        "/definitions/deploy",
        "/executions",
        "/events",
        "/run",
    ]:
        assert marker in bundle

    for marker in [
        "Advanced YAML",
                                "hermes-workflows-editor",
                "hermes-workflows-sidebar",
        "hermes-workflows-timeline",
        "hermes-workflows-graph",
    ]:
        assert marker in bundle


def test_dashboard_css_is_scoped_to_workflows_plugin():
    css_file = PLUGIN_DIR / "dist" / "style.css"
    css = css_file.read_text(encoding="utf-8")
    assert ".hermes-workflows" in css
    for marker in [
        ".hermes-workflows-prompt-editor",
        ".hermes-workflows-contract-editor",
        ".hermes-workflows-assistant",
    ]:
        assert marker in css


def test_validate_deploy_list_show_roundtrip(client):
    r = client.post("/api/plugins/workflows/definitions/validate", json=PASS_SPEC)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["definition"]["workflow_id"] == "dashboard_demo"
    _assert_pass_spec(body["definition"]["spec"])

    deployed = _deploy(client)
    assert deployed["workflow_id"] == "dashboard_demo"
    assert deployed["created_by"] == "dashboard"
    _assert_pass_spec(deployed["spec"])

    listed = client.get("/api/plugins/workflows/definitions").json()["definitions"]
    assert [item["workflow_id"] for item in listed] == ["dashboard_demo"]

    shown = client.get("/api/plugins/workflows/definitions/dashboard_demo").json()["definition"]
    assert shown["workflow_id"] == "dashboard_demo"
    assert shown["version"] == 1
    _assert_pass_spec(shown["spec"])


@pytest.mark.parametrize(
    "path",
    [
        "/api/plugins/workflows/definitions/validate",
        "/api/plugins/workflows/definitions/deploy",
    ],
)
def test_definition_validate_and_deploy_reject_unsupported_primitives(client, path):
    response = client.post(path, json={"spec": UNSUPPORTED_SPEC})

    assert response.status_code == 400
    assert "unsupported node type: send_message on node start" in str(response.json()["detail"])


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/plugins/workflows/definitions/validate", "x" * 1_100_000),
        ("/api/plugins/workflows/definitions/deploy", "x" * 1_100_000),
        ("/api/plugins/workflows/definitions/dashboard_demo/run", "x" * 1_100_000),
        ("/api/plugins/workflows/definitions/draft", {"goal": "x" * 1_100_000}),
        (
            "/api/plugins/workflows/definitions/refine",
            {"instruction": "x" * 1_100_000, "spec": PASS_SPEC},
        ),
    ],
)
def test_workflow_plugin_rejects_oversized_bodies(client, path, payload):
    if isinstance(payload, str):
        response = client.post(
            path,
            content=payload,
            headers={"content-type": "application/x-yaml"},
        )
    else:
        response = client.post(path, json=payload)

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "workflow_request_too_large"


def test_deploy_endpoint_returns_deployed_version_not_latest(client):
    workflow_id = "dashboard_deploy_version_demo"

    def spec(version):
        return {
            "id": workflow_id,
            "name": "Dashboard Deploy Version Demo",
            "version": version,
            "nodes": {"start": {"type": "pass"}},
        }

    first = _deploy(client, spec(2))
    assert first["version"] == 2

    second = _deploy(client, spec(1))

    assert second["workflow_id"] == workflow_id
    assert second["id"] == workflow_id
    assert second["version"] == 1
    assert second["created_by"] == "dashboard"

    with wfdb.connect() as conn:
        records = [
            record
            for record in wfdb.list_definitions(conn)
            if record.workflow_id == workflow_id
        ]
    assert {(record.version, record.created_by) for record in records} == {
        (1, "dashboard"),
        (2, "dashboard"),
    }


def test_run_endpoint_creates_execution_and_list_show_return_it(client):
    _deploy(client)

    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run",
        json={"input": {"value": 7}},
    )
    assert r.status_code == 200, r.text
    execution = r.json()["execution"]
    assert execution["workflow_id"] == "dashboard_demo"
    assert execution["input"] == {"value": 7}
    # Engine/DB status vocabulary — "running" is intentionally absent
    # (executions are queued/waiting/succeeded/failed/blocked/cancelled).
    assert execution["status"] in {"queued", "waiting", "succeeded"}

    with wfdb.connect() as conn:
        stored = wfdb.get_execution(conn, execution["execution_id"])
    assert stored.workflow_id == "dashboard_demo"
    assert stored.input == {"value": 7}

    listed = client.get(
        "/api/plugins/workflows/executions", params={"workflow_id": "dashboard_demo"}
    ).json()["executions"]
    assert [item["execution_id"] for item in listed] == [execution["execution_id"]]

    shown = client.get(
        f"/api/plugins/workflows/executions/{execution['execution_id']}"
    ).json()["execution"]
    assert shown["execution_id"] == execution["execution_id"]
    assert shown["input"] == {"value": 7}


def test_execution_detail_redacts_secret_like_values_for_dashboard_display(client):
    secret_spec = {
        "id": "dashboard_secret_demo",
        "name": "Dashboard Secret Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "start": {
                "type": "pass",
                "output": {
                    "api_key": "${ input.api_key }",
                    "topic": "${ input.topic }",
                },
            }
        },
    }
    _deploy(client, secret_spec)

    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_secret_demo/run",
        json={"input": {"api_key": "secret-value", "topic": "safe-topic"}},
    )
    assert r.status_code == 200, r.text
    execution = r.json()["execution"]
    execution_id = execution["execution_id"]

    assert execution["input"] == {"api_key": "[REDACTED]", "topic": "safe-topic"}
    assert "secret-value" not in r.text

    shown = client.get(f"/api/plugins/workflows/executions/{execution_id}").json()["execution"]
    assert shown["input"] == {"api_key": "[REDACTED]", "topic": "safe-topic"}
    assert shown["context"]["input"] == {"api_key": "[REDACTED]", "topic": "safe-topic"}

    node_runs = client.get(
        f"/api/plugins/workflows/executions/{execution_id}/node-runs"
    ).json()["node_runs"]
    start_run = next(run for run in node_runs if run["node_id"] == "start")
    assert start_run["output"] == {"api_key": "[REDACTED]", "topic": "safe-topic"}

    events = client.get(f"/api/plugins/workflows/executions/{execution_id}/events").json()["events"]
    assert "secret-value" not in json.dumps(events)
    assert "safe-topic" in json.dumps(events)

    with wfdb.connect() as conn:
        stored = wfdb.get_execution(conn, execution_id)
    assert stored.input == {"api_key": "secret-value", "topic": "safe-topic"}


def test_execution_node_runs_endpoint_returns_workflow_native_state(client):
    _deploy(client, WAIT_SPEC)
    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_wait/run",
        json={"input": {"value": "abc"}},
    )
    assert r.status_code == 200, r.text
    execution_id = r.json()["execution"]["execution_id"]

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["execution_id"] == execution_id
    node_ids = [run["node_id"] for run in body["node_runs"]]
    assert node_ids.index("start") < node_ids.index("pause")
    start_run = next(run for run in body["node_runs"] if run["node_id"] == "start")
    assert start_run["output"] == {"seen": "abc"}
    assert any(
        run["node_id"] == "start" and run["status"] == "succeeded"
        for run in body["node_runs"]
    )
    assert any(
        run["node_id"] == "pause" and run["status"] == "waiting"
        for run in body["node_runs"]
    )
    for run in body["node_runs"]:
        assert "kanban_task_id" in run
        assert "output" in run
        assert "error" in run


def test_execution_node_runs_endpoint_does_not_merge_success_output_into_failed_attempt(client):
    _deploy(client, PASS_SPEC)
    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run",
        json={"input": {}},
    )
    assert r.status_code == 200, r.text
    execution_id = r.json()["execution"]["execution_id"]

    with wfdb.connect() as conn:
        conn.execute(
            """
            INSERT INTO workflow_node_runs (
                execution_id, node_id, status, error, started_at, completed_at
            ) VALUES (?, 'start', 'failed', ?, 1, 2)
            """,
            (execution_id, json.dumps({"reason": "boom"})),
        )
        conn.commit()

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")

    assert r.status_code == 200, r.text
    failed_run = next(
        run
        for run in r.json()["node_runs"]
        if run["node_id"] == "start" and run["status"] == "failed"
    )
    assert failed_run["output"] == {}
    assert failed_run["error"] == {"reason": "boom"}


def test_execution_node_runs_endpoint_returns_empty_list_for_queued_execution(client):
    _deploy(client, PASS_SPEC)
    with wfdb.connect() as conn:
        execution_id = wfdb.start_execution(
            conn,
            PASS_SPEC["id"],
            input_data={},
            trigger_type="manual",
        )

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")

    assert r.status_code == 200, r.text
    assert r.json()["node_runs"] == []


def test_execution_node_runs_endpoint_returns_404_for_unknown_execution(client):
    r = client.get("/api/plugins/workflows/executions/missing/node-runs")
    assert r.status_code == 404


def test_execution_node_runs_endpoint_defensively_parses_null_and_malformed_json(client):
    _deploy(client, WAIT_SPEC)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_wait/run",
        json={"input": {"value": "abc"}},
    ).json()["execution"]["execution_id"]

    with wfdb.connect() as conn:
        conn.execute(
            """
            UPDATE workflow_node_runs
               SET output_json = 'null', error = 'not-json'
             WHERE execution_id = ? AND node_id = 'pause'
            """,
            (execution_id,),
        )
        conn.commit()

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")

    assert r.status_code == 200, r.text
    pause_run = next(run for run in r.json()["node_runs"] if run["node_id"] == "pause")
    assert pause_run["output"] == {}
    assert pause_run["error"] == {}


def test_execution_node_runs_endpoint_merges_completed_wait_event_output(client):
    _deploy(client, WAIT_SPEC)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_wait/run",
        json={"input": {"value": "abc"}},
    ).json()["execution"]["execution_id"]

    with wfdb.connect() as conn:
        pause = conn.execute(
            """
            SELECT wait_until FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'pause'
            """,
            (execution_id,),
        ).fetchone()
    assert pause is not None

    assert workflows_dispatcher.tick(limit=1, now=pause["wait_until"] + 1) == 1
    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/node-runs")

    assert r.status_code == 200, r.text
    pause_run = next(run for run in r.json()["node_runs"] if run["node_id"] == "pause")
    assert pause_run["status"] == "succeeded"
    assert pause_run["output"] == {"waited": True}


def test_events_endpoint_returns_append_only_events(client):
    _deploy(client)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run", json={}
    ).json()["execution"]["execution_id"]

    with wfdb.connect() as conn:
        wfdb.append_event(conn, execution_id, "custom_one", {"n": 1})
        wfdb.append_event(conn, execution_id, "custom_two", {"n": 2})

    r = client.get(f"/api/plugins/workflows/executions/{execution_id}/events")
    assert r.status_code == 200, r.text
    custom = [e for e in r.json()["events"] if e["kind"].startswith("custom_")]
    assert [e["kind"] for e in custom] == ["custom_one", "custom_two"]
    assert [e["payload"] for e in custom] == [{"n": 1}, {"n": 2}]
    assert custom[0]["id"] < custom[1]["id"]


def test_cancel_endpoint_is_idempotent(client):
    _deploy(client, WAIT_SPEC)
    execution_id = client.post(
        "/api/plugins/workflows/definitions/dashboard_wait/run",
        json={"input_json": '{"value": 3}'},
    ).json()["execution"]["execution_id"]

    first = client.post(f"/api/plugins/workflows/executions/{execution_id}/cancel")
    assert first.status_code == 200, first.text
    assert first.json()["cancelled"] is True
    assert first.json()["execution"]["status"] == "cancelled"

    second = client.post(f"/api/plugins/workflows/executions/{execution_id}/cancel")
    assert second.status_code == 200, second.text
    assert second.json()["cancelled"] is False
    assert second.json()["execution"]["status"] == "cancelled"

    with wfdb.connect() as conn:
        rows = conn.execute(
            """
            SELECT kind, payload_json FROM workflow_events
             WHERE execution_id = ? AND kind = 'execution_cancelled'
             ORDER BY id
            """,
            (execution_id,),
        ).fetchall()
    assert [row["kind"] for row in rows] == ["execution_cancelled"]
    assert [json.loads(row["payload_json"]) for row in rows] == [{"source": "dashboard"}]


def test_bad_spec_returns_400_with_validation_message(client):
    r = client.post(
        "/api/plugins/workflows/definitions/validate",
        json={"id": "bad_demo", "name": "Bad", "version": 1, "nodes": {}},
    )
    assert r.status_code == 400
    assert "workflow must define at least one node" in r.json()["detail"]


def test_dashboard_bundle_clears_draft_metadata_when_selecting_or_importing_definition():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")

    load_pos = bundle.index("function loadDefinition")
    load_end = bundle.index("function loadEvents", load_pos)
    load_body = bundle[load_pos:load_end]
    assert "setDraftResult(null)" in load_body

    import_pos = bundle.index("function importDefinitionFile")
    import_end = bundle.index("function exportYAML", import_pos)
    import_body = bundle[import_pos:import_end]
    assert "setDraftResult(null)" in import_body

    advanced_pos = bundle.index("function renderAdvancedYaml")
    advanced_end = bundle.index("function renderTimeline", advanced_pos)
    advanced_body = bundle[advanced_pos:advanced_end]
    textarea_pos = advanced_body.index("onChange: function")
    textarea_block = advanced_body[textarea_pos : advanced_body.index("}),", textarea_pos)]
    assert "setDraftResult(null)" in textarea_block
    update_pos = textarea_block.index("updateEditorText(event.target.value)")
    clear_pos = textarea_block.index("setDraftResult(null)")
    assert clear_pos < update_pos


def test_dashboard_bundle_summarizes_structured_prompts_for_draft_review():
    bundle = (PLUGIN_DIR / "src" / "app.js").read_text(encoding="utf-8")
    helper = bundle[
        bundle.index("function promptObjectiveText") : bundle.index("function nodeSummaryRows")
    ]
    rows = bundle[
        bundle.index("function nodeSummaryRows") : bundle.index(
            "function statusClass", bundle.index("function nodeSummaryRows")
        )
    ]

    for marker in [
        "Array.isArray(prompt)",
        "prompt.task",
        "prompt.objective",
        "prompt.description",
    ]:
        assert marker in helper
    assert "promptObjectiveText(node.prompt)" in rows
    assert "String(edge.from || edge.from_" in rows


def test_dashboard_api_sync_db_routes_are_threadpool_safe(client):
    import inspect

    plugin = sys.modules["hermes_dashboard_plugin_workflows_test"]

    for name in [
        "list_definitions",
        "get_definition",
        "list_executions",
        "get_execution",
        "cancel_execution",
        "list_events",
    ]:
        assert not inspect.iscoroutinefunction(getattr(plugin, name)), name

    deploy_source = inspect.getsource(plugin.deploy_definition)
    run_source = inspect.getsource(plugin.run_workflow)
    assert "await asyncio.to_thread(_deploy)" in deploy_source
    assert "await asyncio.to_thread(_run)" in run_source


def test_bad_run_input_returns_400(client):
    _deploy(client)
    r = client.post(
        "/api/plugins/workflows/definitions/dashboard_demo/run",
        json={"input_json": []},
    )
    assert r.status_code == 400
    assert "input_json" in r.json()["detail"]


# --- Task 3: dashboard draft / publish / archive / feed lifecycle API ---


def test_dashboard_draft_put_and_get_round_trip(client):
    spec = copy.deepcopy(PASS_SPEC)
    spec["name"] = "Draft Demo"
    put = client.put(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft",
        json={"spec": spec, "base_version": None},
    )
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["draft"]["workflow_id"] == PASS_SPEC["id"]
    assert body["draft"]["spec"]["name"] == "Draft Demo"
    assert body["draft"]["base_version"] is None

    fetched = client.get(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft"
    )
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["draft"]["spec"]["name"] == "Draft Demo"


def test_dashboard_draft_delete_removes_existing_draft(client):
    spec = copy.deepcopy(PASS_SPEC)
    spec["name"] = "Will Be Deleted"
    client.put(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft",
        json={"spec": spec, "base_version": None},
    )
    deleted = client.delete(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft"
    )
    assert deleted.status_code == 200, deleted.text
    missing = client.get(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft"
    )
    assert missing.status_code == 404


def test_dashboard_publish_creates_definition_and_clears_draft(client):
    spec = copy.deepcopy(PASS_SPEC)
    spec["name"] = "First Publish"
    client.put(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft",
        json={"spec": spec, "base_version": None},
    )
    publish = client.post(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/publish",
        json={"expected_latest_version": None},
    )
    assert publish.status_code == 200, publish.text
    body = publish.json()
    assert body["definition"]["version"] == 1
    assert body["definition"]["name"] == "First Publish"
    missing = client.get(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft"
    )
    assert missing.status_code == 404


def test_dashboard_publish_returns_409_on_version_conflict(client):
    _deploy(client, PASS_SPEC)
    spec = copy.deepcopy(PASS_SPEC)
    spec["name"] = "Stale Draft"
    spec["version"] = 2
    client.put(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft",
        json={"spec": spec, "base_version": 1},
    )
    conflict = client.post(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/publish",
        json={"expected_latest_version": 0},
    )
    assert conflict.status_code == 409, conflict.text
    assert conflict.json() == {
        "detail": {
            "code": "workflow_version_conflict",
            "message": "Workflow changed since this draft was created.",
            "field_errors": {},
            "hint": "Reload the latest version and review the draft again.",
        }
    }


def test_dashboard_archive_hides_workflow_and_include_archived_restores(client):
    _deploy(client, PASS_SPEC)

    listed = client.get("/api/plugins/workflows/definitions")
    assert listed.status_code == 200, listed.text
    assert len(listed.json()["definitions"]) == 1

    archived = client.post(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/archive",
        json={"archived": True},
    )
    assert archived.status_code == 200, archived.text
    assert archived.json()["workflow"]["archived"] is True

    default_list = client.get("/api/plugins/workflows")
    assert default_list.status_code == 200, default_list.text
    assert default_list.json()["workflows"] == []

    full_list = client.get("/api/plugins/workflows?include_archived=true")
    assert full_list.status_code == 200, full_list.text
    assert len(full_list.json()["workflows"]) == 1

    restored = client.post(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/archive",
        json={"archived": False},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["workflow"]["archived"] is False
    assert client.get("/api/plugins/workflows").json()["workflows"] != []


def test_dashboard_summary_includes_draft_enabled_and_archived(client):
    _deploy(client, PASS_SPEC)
    spec = copy.deepcopy(PASS_SPEC)
    spec["name"] = "Draft Open"
    client.put(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/draft",
        json={"spec": spec, "base_version": 1},
    )
    listed = client.get("/api/plugins/workflows").json()["workflows"]
    assert len(listed) == 1
    row = listed[0]
    assert row["workflow_id"] == PASS_SPEC["id"]
    assert row["has_draft"] is True
    assert row["latest_version"] == 1
    assert row["enabled"] is True
    assert row["archived"] is False
    assert row["open_feed_count"] == 0


def test_dashboard_enabled_toggle_appears_in_summary(client):
    _deploy(client, PASS_SPEC)
    toggle = client.post(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}/enabled",
        json={"enabled": False},
    )
    assert toggle.status_code == 200, toggle.text
    listed = client.get("/api/plugins/workflows").json()["workflows"][0]
    assert listed["enabled"] is False


def test_dashboard_paused_feed_writes_return_409(client):
    _deploy(client, CONTINUOUS_SPEC)
    feed = client.post(
        "/api/plugins/workflows/definitions/continuous_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    ).json()["feed"]
    paused = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "paused"},
    )
    assert paused.status_code == 200, paused.text

    enqueue = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/repo", "prompt": "Review README drift"}},
    )
    assert enqueue.status_code == 409, enqueue.text
    assert enqueue.json()["detail"]["code"] == "workflow_feed_not_open"


def test_dashboard_closed_feed_writes_and_reopen_return_409(client):
    _deploy(client, CONTINUOUS_SPEC)
    feed = client.post(
        "/api/plugins/workflows/definitions/continuous_demo/input-feeds",
        json={"trigger_id": "kickoff"},
    ).json()["feed"]
    client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "closed"},
    )

    enqueue = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/items",
        json={"input": {"repo_path": "/repo", "prompt": "Review README drift"}},
    )
    assert enqueue.status_code == 409, enqueue.text
    assert enqueue.json()["detail"]["code"] == "workflow_feed_closed"

    reopen = client.post(
        f"/api/plugins/workflows/input-feeds/{feed['feed_id']}/status",
        json={"status": "open"},
    )
    assert reopen.status_code == 409, reopen.text
    assert reopen.json()["detail"]["code"] == "workflow_feed_terminal"


def test_dashboard_delete_requires_explicit_purge_when_history_exists(client):
    definition = _deploy(client, PASS_SPEC)
    client.post(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}/run",
        json={"input": {}},
    )
    blocked = client.delete(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}"
    )
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["code"] == "workflow_history_exists"
    purged = client.delete(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}?purge=true"
    )
    assert purged.status_code == 200, purged.text
    assert purged.json()["deleted"] is True


def test_dashboard_delete_history_free_workflow_succeeds_without_purge(client):
    _deploy(client, PASS_SPEC)
    r = client.delete(
        f"/api/plugins/workflows/definitions/{PASS_SPEC['id']}",
    )
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True


# --- Task 5: AI-first draft review envelope (summary/assumptions/warnings) ---


ASSISTANT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "workflows" / "assistant_responses.json"


def _assistant_fixture(name: str) -> dict:
    return json.loads(ASSISTANT_FIXTURE_PATH.read_text())[name]


def test_definition_draft_endpoint_returns_summary_assumptions_and_warnings_envelope(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin
    from hermes_cli.workflows_assistant import parse_assistant_payload

    payload = _assistant_fixture("draft")

    def fake_draft(goal):
        assert goal == "guard the readme"
        return parse_assistant_payload(payload)

    monkeypatch.setattr(
        plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft
    )

    r = client.post(
        "/api/plugins/workflows/definitions/draft", json={"goal": "guard the readme"}
    )
    assert r.status_code == 200, r.text
    body = r.json()["draft"]
    assert body["summary"] == payload["summary"]
    assert body["assumptions"] == payload["assumptions"]
    assert body["warnings"] == payload["warnings"]
    assert body["spec"]["id"] == "readme_drift_guard"


def test_definition_refine_endpoint_returns_summary_assumptions_and_warnings_envelope(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin
    from hermes_cli.workflows_assistant import parse_assistant_payload

    payload = _assistant_fixture("refine")

    def fake_refine(spec, instruction):
        assert instruction == "add reviewer + retry"
        return parse_assistant_payload(payload)

    monkeypatch.setattr(
        plugin.workflows_assistant, "refine_workflow_with_default_runner", fake_refine
    )

    current_spec = _assistant_fixture("draft")["spec"]
    r = client.post(
        "/api/plugins/workflows/definitions/refine",
        json={"spec": current_spec, "instruction": "add reviewer + retry"},
    )
    assert r.status_code == 200, r.text
    body = r.json()["draft"]
    assert body["summary"] == payload["summary"]
    assert body["assumptions"] == payload["assumptions"]
    assert body["warnings"] == payload["warnings"]
    assert body["spec"]["version"] == 2


def test_definition_draft_endpoint_returns_typed_assistant_validation_error(client, monkeypatch):
    """Invalid candidate output must surface as the typed
    workflow_assistant_validation_error code so the UI can route to Repair-with-AI."""
    import hermes_dashboard_plugin_workflows_test as plugin
    from hermes_cli.workflows_assistant import AssistantValidationError

    def fake_draft(goal):
        raise AssistantValidationError(
            "agent_task node fetch_readme requires a non-empty result_contract"
        )

    monkeypatch.setattr(
        plugin.workflows_assistant, "draft_workflow_with_default_runner", fake_draft
    )

    r = client.post(
        "/api/plugins/workflows/definitions/draft", json={"goal": "broken"}
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "workflow_assistant_validation_error"
