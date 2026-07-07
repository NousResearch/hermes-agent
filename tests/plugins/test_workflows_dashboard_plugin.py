"""Tests for the Workflows dashboard plugin backend."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

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


def _load_plugin_router():
    plugin_file = PLUGIN_DIR / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_workflows_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


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


def _assert_pass_spec(spec: dict) -> None:
    assert spec["id"] == PASS_SPEC["id"]
    assert spec["name"] == PASS_SPEC["name"]
    assert spec["version"] == PASS_SPEC["version"]
    assert spec["nodes"]["start"]["type"] == "pass"


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
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_text"]
    assert "Review code changes before merge" in body["prompt_text"]
    assert "${ input.repo }" in body["prompt_text"]
    assert "verdict" in body["prompt_text"]
    assert body["result_contract"]["verdict"] == "approved|changes_requested"


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
    assert body["dispatcher"]["dispatch_in_gateway"] is False
    assert body["dispatcher"]["tick_interval_seconds"] == 30.0
    assert body["dispatcher"]["warning"]


def test_status_endpoint_handles_non_dict_workflow_config(client, monkeypatch):
    import hermes_dashboard_plugin_workflows_test as plugin

    monkeypatch.setattr(plugin, "load_config", lambda: {"workflow": "not-a-dict"})

    r = client.get("/api/plugins/workflows/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dispatcher"]["dispatch_in_gateway"] is False
    assert body["dispatcher"]["tick_interval_seconds"] == 30.0
    assert body["dispatcher"]["warning"]


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
    bundle = PLUGIN_DIR / "dist" / "index.js"
    result = subprocess.run(
        [node, "--check", str(bundle)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_dashboard_bundle_renders_node_runs_and_linked_worker_tasks():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    assert "node-runs" in bundle
    assert "Linked worker task" in bundle
    assert "waiting on agent" in bundle or "waiting_on_agent" in bundle
    assert "renderNodeRuns" in bundle


def test_dashboard_bundle_wires_node_runs_as_execution_drilldown():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    load_node_runs = bundle[
        bundle.index("function loadNodeRuns") : bundle.index("function loadExecution")
    ]
    load_execution = bundle[
        bundle.index("function loadExecution") : bundle.index("function loadDefinitions")
    ]
    render_timeline = bundle[
        bundle.index("function renderTimeline") : bundle.index("function renderSimpleGraph")
    ]
    goal_builder = bundle[
        bundle.index("function renderGoalBuilder") : bundle.index("function renderDraftReview")
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
    assert "renderNodeRuns()" not in goal_builder


def test_dashboard_bundle_contains_validation_checklist_and_dispatcher_banner():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    assert "Validation checklist" in bundle
    assert "No unsupported nodes (implemented today)" in bundle
    assert "No unsupported triggers (implemented today)" in bundle
    assert "implemented dashboard/dispatcher readiness" in bundle
    assert "Implemented triggers today" in bundle
    assert "Implemented node types today" in bundle
    assert "Dispatcher readiness" in bundle
    assert "workflow.dispatch_in_gateway" in bundle
    assert "renderValidationChecklist" in bundle
    assert "loadWorkflowStatus" in bundle


def test_dashboard_validation_checklist_waits_for_parsed_spec_before_showing_failures():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    render_body = bundle[
        bundle.index("function renderValidationChecklist") : bundle.index(
            "function renderDispatcherReadiness"
        )
    ]

    assert "Validate Advanced YAML to update the checklist" in render_body
    assert "if (!spec)" in render_body


def test_dashboard_initial_load_refreshes_workflow_status_without_waiting_for_it():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
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
        bundle.index("function renderDefinitionList") : bundle.index("function renderRunInputForm")
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
    assert "loadDefinition(definition.workflow_id, definition.version)" in definition_list


def test_dashboard_bundle_runs_selected_or_active_definition_version():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    selected_run_version = bundle[
        bundle.index("function selectedRunVersion") : bundle.index("function runWorkflow")
    ]
    run_workflow = bundle[
        bundle.index("function runWorkflow") : bundle.index("function draftFromGoal")
    ]
    run_form = bundle[
        bundle.index("function renderRunInputForm") : bundle.index("function renderExecutions")
    ]

    assert "function selectedRunVersion(workflowId)" in selected_run_version
    assert "return selectedDefinition.version" in selected_run_version
    assert "return versionForSpec(spec)" in selected_run_version
    assert "const runVersion = selectedRunVersion(workflowId)" in run_workflow
    assert '"/run" + versionQuery(runVersion)' in run_workflow
    assert "const runSelectValue = selectedDefinition ? definitionSelectionKey(selectedDefinition)" in run_form
    assert "value: runSelectValue" in run_form
    assert "loadDefinition(id, version)" in run_form
    assert "value: definitionSelectionKey(definition)" in run_form


def test_dashboard_dispatcher_readiness_handles_unknown_status_separately():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    render_start = bundle.index("function renderDispatcherReadiness(")
    render_end = bundle.index("function renderAdvancedYaml", render_start)
    render_body = bundle[render_start:render_end]

    assert "Dispatcher readiness unavailable" in bundle or "Dispatcher readiness unknown" in bundle
    assert 'typeof dispatcher.dispatch_in_gateway === "boolean"' in render_body


def _dashboard_helper_js() -> str:
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    kind_start = bundle.index("const NODE_KIND_LIST")
    kind_end = bundle.index("const EXAMPLE_DEFINITION", kind_start)
    start = bundle.index("function asArray")
    end = bundle.index("function statusClass", start)
    return bundle[kind_start:kind_end] + "\n" + bundle[start:end]


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


def _run_node_input_fields_for_spec(spec):
    return _run_dashboard_function("inputFieldsForSpec", [spec])


def _run_validation_checklist(spec, capabilities=None):
    args = [spec]
    if capabilities is not None:
        args.append(capabilities)
    return _run_dashboard_function("validationChecklist", args)


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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    assert "window.__HERMES_PLUGIN_SDK__" in bundle
    assert "window.__HERMES_PLUGINS__" in bundle
    assert 'REG.register("workflows"' in bundle
    assert not any(
        line.lstrip().startswith(("import ", "export ")) for line in bundle.splitlines()
    )
    assert "__webpack_require__" not in bundle


def test_dashboard_bundle_uses_generated_input_form_for_runs():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    assert "Run test" in bundle
    assert "Advanced input JSON" in bundle
    assert "renderRunInputForm" in bundle
    assert "inputFieldValues" in bundle
    assert "Manual run form" not in bundle


def test_dashboard_run_workflow_uses_form_values_unless_advanced_json():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    run_pos = bundle.index("function runWorkflow")
    run_body = bundle[run_pos : bundle.index("function draftFromGoal", run_pos)]
    render_pos = bundle.index("function renderRunInputForm")
    render_body = bundle[render_pos : bundle.index("function renderExecutions", render_pos)]

    assert "function runInputSpec" in bundle
    assert "showAdvancedInputJson" in run_body
    assert "inputFieldValues" in run_body
    assert 'JSON.parse(runInputText || "{}")' in run_body
    assert "inputFieldsForSpec(runInputSpec())" in run_body
    assert "body: JSON.stringify({ input: input })" in run_body
    assert "input_json" not in run_body
    assert "inputFieldsForSpec(spec)" in render_body or "runInputSpec()" in render_body
    assert 'field.kind === "integer"' in render_body
    assert 'field.kind === "boolean"' in render_body
    assert 'field.kind === "json"' in render_body
    assert 'step: field.kind === "number" ? "any" : field.kind === "integer" ? "1" : undefined' in render_body


def test_dashboard_bundle_clears_run_input_values_when_active_spec_changes():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    load_pos = bundle.index("function loadDefinition")
    load_body = bundle[load_pos : bundle.index("function loadEvents", load_pos)]
    draft_pos = bundle.index("function draftFromGoal")
    draft_body = bundle[draft_pos : bundle.index("function refineWorkflow", draft_pos)]
    refine_pos = bundle.index("function refineWorkflow")
    refine_body = bundle[refine_pos : bundle.index("function importDefinitionFile", refine_pos)]
    import_pos = bundle.index("function importDefinitionFile")
    import_body = bundle[import_pos : bundle.index("function exportYAML", import_pos)]

    def assert_resets_advanced_input_state(body):
        reset_pos = body.index("setInputFieldValues({})")
        assert "setShowAdvancedInputJson(false)" in body
        assert "setRunInputText(\"{}\")" in body
        assert reset_pos < body.index("setShowAdvancedInputJson(false)")
        assert reset_pos < body.index("setRunInputText(\"{}\")")

    assert load_body.index("setDraftSpec(") < load_body.index("setInputFieldValues({})")
    assert draft_body.index("if (draft.spec)") < draft_body.index("setInputFieldValues({})")
    assert refine_body.index("if (!draft.spec)") < refine_body.index("setInputFieldValues({})")
    assert import_body.index("reader.onload = function") < import_body.index(
        "setInputFieldValues({})"
    )
    for body in (load_body, draft_body, refine_body, import_body):
        assert_resets_advanced_input_state(body)


def test_dashboard_bundle_clears_run_input_values_when_advanced_yaml_changes():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    advanced = bundle[
        bundle.index("function renderAdvancedYaml") : bundle.index(
            "function renderDefinitionList"
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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    render_start = bundle.index('return h("div", { className: "hermes-workflows" }')
    render_tree = bundle[render_start:]

    assert "What do you want to automate?" in bundle
    assert "Describe workflow" in bundle
    assert "Use Kanban for one-off work queues" in bundle
    assert "Advanced YAML" in bundle
    assert "Validate / deploy definition" not in bundle
    assert render_tree.index("renderGoalBuilder()") < render_tree.index(
        'className: "hermes-workflows-grid"'
    )
    assert render_tree.index("renderGoalBuilder()") < render_tree.index("Workflow list")
    assert bundle.index("What do you want to automate?") < bundle.index("Workflow list")


def test_dashboard_bundle_contains_draft_review_and_refine_ui():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    assert "Draft review" in bundle
    assert "Questions" in bundle
    assert "Assumptions" in bundle
    assert "Unsupported requests" in bundle
    assert "Refine workflow" in bundle
    assert "renderDraftReview" in bundle
    assert "refineWorkflow" in bundle


def test_dashboard_bundle_draft_review_orders_questions_before_assumptions():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    review = bundle[
        bundle.index("function renderDraftReview") : bundle.index("function renderAdvancedYaml")
    ]

    assert review.index("Questions") < review.index("Assumptions")


def test_dashboard_bundle_draft_review_notes_when_assistant_metadata_is_missing():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    review = bundle[
        bundle.index("function renderDraftReview") : bundle.index("function renderAdvancedYaml")
    ]

    assert "No assistant draft metadata available." in review
    assert "hasDraftMetadata" in review


def test_dashboard_bundle_goal_builder_does_not_duplicate_draft_summary():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    goal = bundle[
        bundle.index("function renderGoalBuilder") : bundle.index("function renderDraftReview")
    ]

    assert "draftResult.summary" not in goal


def test_dashboard_bundle_wires_draft_refine_before_advanced_yaml():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    render_start = bundle.index('return h("div", { className: "hermes-workflows" }')
    render_tree = bundle[render_start:]

    assert "/definitions/refine" in bundle
    assert 'setRefineText("")' in bundle
    assert "nodeSummaryRows" in bundle
    assert render_tree.index("renderDraftReview()") < render_tree.index(
        "renderAdvancedYaml()"
    )


def test_dashboard_bundle_draft_review_labels_branch_and_failure_targets():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    rows_pos = bundle.index("function nodeSummaryRows")
    next_function_pos = bundle.index("function statusClass", rows_pos)
    rows_body = bundle[rows_pos:next_function_pos]

    for marker in ['"default → "', '"catch → "', "edge.label", "edge.condition", "parts[1]"]:
        assert marker in rows_body


def test_dashboard_bundle_refine_clears_stale_state_before_validation_and_requires_spec():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    refine_pos = bundle.index("function refineWorkflow")
    next_function_pos = bundle.index("function importDefinitionFile", refine_pos)
    refine_body = bundle[refine_pos:next_function_pos]
    early_return_pos = refine_body.index("if (!instruction || !spec)")

    assert refine_body.index('setStatus("")') < early_return_pos
    assert refine_body.index("setDraftResult(null)") < early_return_pos
    assert "Refine response did not include a workflow spec." in refine_body
    for marker in [
        "setSelectedDefinition(null)",
        "setSelectedNode(null)",
        'setNodeJson("")',
        'setNodeMessage("")',
    ]:
        assert marker in refine_body
    assert refine_body.index("if (!draft.spec)") < refine_body.index("setDraftResult(draft)")
    assert refine_body.index("setDraftResult(draft)") < refine_body.index(
        'setRefineText("")'
    )
    assert refine_body.index('setRefineText("")') < refine_body.index(
        'setStatus("Refined workflow draft.")'
    )


def test_dashboard_bundle_syncs_editor_when_definition_is_selected():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    load_definition_pos = bundle.index("function loadDefinition")
    next_function_pos = bundle.index("function loadEvents", load_definition_pos)
    load_definition_body = bundle[load_definition_pos:next_function_pos]

    assert "updateEditorText(specToEditorText(definition.spec))" in load_definition_body


def test_dashboard_bundle_clears_stale_draft_state_before_empty_goal_error():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    draft_pos = bundle.index("function draftFromGoal")
    next_function_pos = bundle.index("function importDefinitionFile", draft_pos)
    draft_body = bundle[draft_pos:next_function_pos]
    empty_goal_pos = draft_body.index("if (!goal)")

    assert draft_body.index('setStatus("")') < empty_goal_pos
    assert draft_body.index("setDraftResult(null)") < empty_goal_pos
    assert "Describe what you want the workflow to automate." in draft_body


def test_dashboard_bundle_resets_stale_selection_after_goal_draft():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    draft_pos = bundle.index("function draftFromGoal")
    next_function_pos = bundle.index("function refineWorkflow", draft_pos)
    draft_body = bundle[draft_pos:next_function_pos]

    for marker in [
        "setSelectedDefinition(null)",
        "setSelectedNode(null)",
        'setNodeJson("")',
        'setNodeMessage("")',
        "setDraftSpec(draft.spec)",
        "updateEditorText(specToEditorText(draft.spec))",
    ]:
        assert marker in draft_body
    assert 'aria-label' in bundle and 'Describe workflow goal' in bundle


def test_dashboard_bundle_keeps_yaml_as_advanced_escape_hatch():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
    advanced_pos = bundle.index("function renderAdvancedYaml")
    advanced_body = bundle[advanced_pos : bundle.index("function renderDefinitionList", advanced_pos)]

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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    assert "URLSearchParams" in bundle
    assert "location.search" in bundle
    assert 'get("execution")' in bundle or "get('execution')" in bundle
    assert "refresh(initialExecutionId)" in bundle or "loadExecutions(initialExecutionId)" in bundle


def test_dashboard_bundle_contains_visual_editor_markers():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    for marker in [
        "Cell editor",
        "Agent cell prompt",
        "Prompt assistant",
        "Advanced JSON",
        "applyAgentCellForm",
        "renderAgentCellEditor",
        "promptText",
        "resultContractText",
        "/prompt-assistant/draft",
        "draftPromptWithAssistant",
        "Available context placeholders",
        "Expected output contract JSON",
    ]:
        assert marker in bundle

    assert "Apply node JSON" in bundle  # still available only as advanced escape hatch


def test_dashboard_bundle_contains_workflow_mvp_api_and_ui_markers():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
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
        "Workflow list",
        "Advanced YAML",
        "Run test",
        "Execution list",
        "Execution detail timeline",
        "Visual workflow editor",
        "hermes-workflows-list",
        "hermes-workflows-editor",
        "hermes-workflows-run-form",
        "hermes-workflows-executions",
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
    assert execution["status"] in {"queued", "running", "waiting", "succeeded"}

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
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")

    load_pos = bundle.index("function loadDefinition")
    load_end = bundle.index("function loadEvents", load_pos)
    load_body = bundle[load_pos:load_end]
    assert "setDraftResult(null)" in load_body

    import_pos = bundle.index("function importDefinitionFile")
    import_end = bundle.index("function exportYAML", import_pos)
    import_body = bundle[import_pos:import_end]
    assert "setDraftResult(null)" in import_body

    advanced_pos = bundle.index("function renderAdvancedYaml")
    advanced_end = bundle.index("function renderDefinitionList", advanced_pos)
    advanced_body = bundle[advanced_pos:advanced_end]
    textarea_pos = advanced_body.index("onChange: function")
    textarea_block = advanced_body[textarea_pos : advanced_body.index("}),", textarea_pos)]
    assert "setDraftResult(null)" in textarea_block
    update_pos = textarea_block.index("updateEditorText(event.target.value)")
    clear_pos = textarea_block.index("setDraftResult(null)")
    assert clear_pos < update_pos


def test_dashboard_bundle_summarizes_structured_prompts_for_draft_review():
    bundle = (PLUGIN_DIR / "dist" / "index.js").read_text(encoding="utf-8")
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
