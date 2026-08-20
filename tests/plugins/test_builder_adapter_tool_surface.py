import json
from types import SimpleNamespace
from pathlib import Path

import model_tools
import yaml

from hermes_cli.kanban_db import _worker_policy_env
from plugins.builder_adapter.native import BUILDER_WORKER_POLICY
from plugins.builder_adapter.plugin_tools import TOOLS, handle_packet


EXPECTED_BUILDER_TOOLS = {
    "builder_patch",
    "builder_read_execution_packet",
    "builder_read_file",
    "builder_run_validation_profile",
    "builder_search_files",
    "builder_write_file",
}
EXPECTED_LIFECYCLE_TOOLS = {
    "kanban_block",
    "kanban_complete",
    "kanban_heartbeat",
}
FORBIDDEN_KANBAN_TOOLS = {
    "kanban_comment",
    "kanban_create",
    "kanban_link",
    "kanban_show",
}


def test_generic_worker_policy_exports_exact_runtime_owned_allowlist():
    task = SimpleNamespace(worker_policy=BUILDER_WORKER_POLICY)
    assert _worker_policy_env(task) == {
        "HERMES_INTERNAL_WORKER_POLICY": "hermes.builder_dispatch.v1",
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST": json.dumps(
            sorted(BUILDER_WORKER_POLICY["tool_allowlist"]),
            separators=(",", ":"),
        ),
    }
    assert _worker_policy_env(SimpleNamespace(worker_policy=None)) == {}


def test_governed_worker_gets_only_minimal_kanban_lifecycle(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
        json.dumps(sorted(EXPECTED_LIFECYCLE_TOOLS)),
    )
    definitions = model_tools._compute_tool_definitions(
        enabled_toolsets=[], quiet_mode=True
    )
    names = {definition["function"]["name"] for definition in definitions}
    assert names == EXPECTED_LIFECYCLE_TOOLS
    assert not names & FORBIDDEN_KANBAN_TOOLS


def test_runtime_owned_allowlist_bypasses_progressive_tool_disclosure(monkeypatch):
    from tools import tool_search

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
        json.dumps(sorted(EXPECTED_LIFECYCLE_TOOLS)),
    )
    monkeypatch.setattr(
        tool_search,
        "assemble_tool_defs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact worker tools must not be deferred")
        ),
    )

    definitions = model_tools._compute_tool_definitions(
        enabled_toolsets=[], quiet_mode=True
    )

    assert {
        definition["function"]["name"] for definition in definitions
    } == EXPECTED_LIFECYCLE_TOOLS


def test_ordinary_worker_behavior_remains_broad_and_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.delenv("HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST", raising=False)
    definitions = model_tools._compute_tool_definitions(
        enabled_toolsets=[], quiet_mode=True
    )
    names = {definition["function"]["name"] for definition in definitions}
    assert EXPECTED_LIFECYCLE_TOOLS <= names
    assert FORBIDDEN_KANBAN_TOOLS <= names


def test_plugin_registers_exact_builder_tools_with_closed_schemas():
    names = {name for name, _, _ in TOOLS}
    assert names == EXPECTED_BUILDER_TOOLS
    for name, schema, _ in TOOLS:
        assert schema["name"] == name
        assert schema["parameters"]["additionalProperties"] is False


def test_plugin_manifest_advertises_every_registered_builder_tool():
    manifest = yaml.safe_load(
        (
            Path(__file__).parents[2]
            / "plugins/builder_adapter/plugin.yaml"
        ).read_text(encoding="utf-8")
    )
    assert set(manifest["provides_tools"]) == EXPECTED_BUILDER_TOOLS


def test_plugin_handlers_require_the_exact_runtime_allowlist(monkeypatch):
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_POLICY", BUILDER_WORKER_POLICY["policy_id"]
    )
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
        json.dumps(["builder_read_execution_packet"]),
    )
    response = json.loads(handle_packet({}))
    assert response["ok"] is False
    assert response["errors"][0]["code"] == "AUTHORIZATION_FAILED"
