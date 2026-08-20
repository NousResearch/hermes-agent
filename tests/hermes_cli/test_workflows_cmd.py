"""Tests for hermes_cli.workflows_cmd — ``hermes workflows import-studio``.

Covers: Hermes Studio ``hermes-studio.workflow`` envelope parsing, node/edge
conversion into the Agent blueprint shape (agent + gate nodes, synthetic
start/end), approval-gate rerouting, condition/feedback edge handling, coding
agent nodes, any-join downgrades, credential rejection, name validation,
cycle rejection, and the file import path (write + --print).

Uses the profile_env fixture pattern from tests/hermes_cli/test_profiles.py:
Path.home() and HERMES_HOME are redirected to tmp_path so nothing touches
the real ~/.hermes.
"""

import json
from pathlib import Path

import pytest

from hermes_cli.workflows_cmd import (
    MAX_EDGES,
    MAX_NODES,
    WorkflowImportError,
    convert_studio_envelope,
    import_studio_file,
    workflows_dir,
)


# ---------------------------------------------------------------------------
# Shared fixture: redirect Path.home() and HERMES_HOME (profile_env pattern)
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated environment: Path.home() -> tmp_path, HERMES_HOME -> tmp/.hermes."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


@pytest.fixture()
def hermes_home(profile_env):
    return profile_env / ".hermes"


# ---------------------------------------------------------------------------
# Sample Studio envelopes
# ---------------------------------------------------------------------------

def studio_node(
    node_id: str,
    *,
    title: str,
    agent: str = "hermes",
    input_text: str = "",
    provider: str = "",
    model: str = "",
    skills: list = None,
    approval_required: bool = False,
    join: str = "all",
    x: float = 0.0,
    y: float = 0.0,
):
    data = {
        "title": title,
        "agent": agent,
        "input": input_text,
        "skills": skills or [],
        "approvalRequired": approval_required,
        "orchestration": {"join": join},
    }
    # Current Studio exports strip runtime bindings; pre-release exports may
    # carry provider/model (legacy import keys). Include them when requested.
    if provider:
        data["provider"] = provider
    if model:
        data["model"] = model
    return {
        "id": node_id,
        "type": "agent",
        "position": {"x": x, "y": y},
        "data": data,
    }


def studio_edge(edge_id: str, source: str, target: str, **orchestration):
    edge = {
        "id": edge_id,
        "source": source,
        "target": target,
        "sourceHandle": "output",
        "targetHandle": "input",
        "type": "smoothstep",
        "animated": True,
    }
    if orchestration:
        edge["data"] = {"orchestration": orchestration}
    return edge


def make_envelope(name: str = "Research Flow", nodes=None, edges=None):
    return {
        "format": "hermes-studio.workflow",
        "version": 1,
        "definition": {
            "name": name,
            "nodes": nodes or [],
            "edges": edges or [],
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        },
    }


SIMPLE_ENVELOPE = make_envelope(
    name="Research Flow",
    nodes=[
        studio_node("n1", title="Fetch PRs", input_text="Fetch open GitHub PRs", model="claude-3-5-sonnet"),
        studio_node("n2", title="Summarize", input_text="Summarize the PR list", model="gpt-4o"),
    ],
    edges=[studio_edge("e1", "n1", "n2")],
)


# ---------------------------------------------------------------------------
# Conversion basics
# ---------------------------------------------------------------------------

class TestEnvelopeValidation:
    def test_requires_envelope(self):
        with pytest.raises(WorkflowImportError, match="envelope is required"):
            convert_studio_envelope([])

    def test_rejects_wrong_format(self):
        env = {"format": "other.format", "version": 1, "definition": {"name": "x", "nodes": [], "edges": []}}
        with pytest.raises(WorkflowImportError, match="unsupported workflow export format"):
            convert_studio_envelope(env)

    def test_rejects_wrong_version(self):
        env = {"format": "hermes-studio.workflow", "version": 2, "definition": {"name": "x", "nodes": [], "edges": []}}
        with pytest.raises(WorkflowImportError, match="unsupported workflow export version"):
            convert_studio_envelope(env)

    def test_requires_name(self):
        env = {"format": "hermes-studio.workflow", "version": 1, "definition": {"nodes": [], "edges": []}}
        with pytest.raises(WorkflowImportError, match="name is required"):
            convert_studio_envelope(env)

    def test_rejects_credential_fields(self):
        env = make_envelope(
            name="x",
            nodes=[studio_node("n1", title="T", input_text="secret")],
        )
        env["definition"]["nodes"][0]["data"]["api_key"] = "sk-1234"
        with pytest.raises(WorkflowImportError, match="credential field"):
            convert_studio_envelope(env)

    def test_rejects_path_traversal_name(self):
        env = make_envelope(name="../evil")
        with pytest.raises(WorkflowImportError, match="invalid workflow name"):
            convert_studio_envelope(env)

    def test_rejects_non_agent_node(self):
        env = make_envelope(nodes=[{"id": "x", "type": "http", "position": {"x": 0, "y": 0}, "data": {}}])
        with pytest.raises(WorkflowImportError, match="agent-only"):
            convert_studio_envelope(env)

    def test_rejects_reserved_start_end_ids(self):
        env = make_envelope(nodes=[studio_node("start", title="Bad")])
        with pytest.raises(WorkflowImportError, match="reserved"):
            convert_studio_envelope(env)

    def test_enforces_node_and_edge_limits(self):
        env = make_envelope(
            nodes=[studio_node(f"n{i}", title=f"N{i}") for i in range(MAX_NODES + 1)]
        )
        with pytest.raises(WorkflowImportError, match="exceeds"):
            convert_studio_envelope(env)

        env2 = make_envelope(
            nodes=[studio_node("n1", title="N1")],
            edges=[studio_edge(f"e{i}", "n1", "n1") for i in range(MAX_EDGES + 1)],
        )
        with pytest.raises(WorkflowImportError, match="exceeds"):
            convert_studio_envelope(env2)

    def test_rejects_cycles(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="A", input_text="a"),
                studio_node("n2", title="B", input_text="b"),
            ],
            edges=[studio_edge("e1", "n1", "n2"), studio_edge("e2", "n2", "n1")],
        )
        with pytest.raises(WorkflowImportError, match="cycle"):
            convert_studio_envelope(env)


class TestConversion:
    def test_simple_graph_maps_nodes_and_edges(self):
        blueprint, warnings = convert_studio_envelope(SIMPLE_ENVELOPE)
        assert blueprint["name"] == "research-flow"  # slugified like the dashboard builder
        # start + 2 agent nodes + end
        assert [n["type"] for n in blueprint["nodes"]] == ["start", "agent", "agent", "end"]
        by_id = {n["id"]: n for n in blueprint["nodes"]}
        assert by_id["n1"]["data"]["label"] == "Fetch PRs"
        assert by_id["n1"]["data"]["prompt"] == "Fetch open GitHub PRs"
        assert by_id["n1"]["data"]["model"] == "claude-3-5-sonnet"
        # edges: original n1->n2 plus start->n1 and n2->end
        edge_pairs = {(e["source"], e["target"]) for e in blueprint["edges"]}
        assert ("n1", "n2") in edge_pairs
        assert ("start", "n1") in edge_pairs
        assert ("n2", "end") in edge_pairs
        assert warnings == []

    def test_provider_model_combined(self):
        env = make_envelope(
            nodes=[studio_node("n1", title="T", input_text="hi", provider="anthropic", model="claude-3-5-sonnet")]
        )
        blueprint, _ = convert_studio_envelope(env)
        by_id = {n["id"]: n for n in blueprint["nodes"]}
        assert by_id["n1"]["data"]["model"] == "anthropic/claude-3-5-sonnet"

    def test_skills_go_to_context(self):
        env = make_envelope(
            nodes=[studio_node("n1", title="T", input_text="hi", skills=["plan", "pdf"])]
        )
        blueprint, _ = convert_studio_envelope(env)
        by_id = {n["id"]: n for n in blueprint["nodes"]}
        assert "Selected skills: plan, pdf" in by_id["n1"]["data"]["context"]

    def test_approval_gate_inserted_and_edges_rerouted(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="Draft", input_text="draft", approval_required=True),
                studio_node("n2", title="Send", input_text="send"),
            ],
            edges=[studio_edge("e1", "n1", "n2")],
        )
        blueprint, warnings = convert_studio_envelope(env)
        types = [n["type"] for n in blueprint["nodes"]]
        assert types.count("gate") == 1
        gate = next(n for n in blueprint["nodes"] if n["type"] == "gate")
        assert gate["id"] == "n1-gate"
        assert gate["data"]["label"] == "Approve: Draft"
        edge_pairs = {(e["source"], e["target"]) for e in blueprint["edges"]}
        # n1's outgoing edge goes through the gate, not straight to n2
        assert ("n1", "n2") not in edge_pairs
        assert ("n1", "n1-gate") in edge_pairs
        assert ("n1-gate", "n2") in edge_pairs
        assert warnings == []

    def test_coding_agent_warns_and_becomes_agent_node(self):
        env = make_envelope(
            nodes=[studio_node("n1", title="Review", agent="codex", input_text="review code")]
        )
        blueprint, warnings = convert_studio_envelope(env)
        by_id = {n["id"]: n for n in blueprint["nodes"]}
        assert by_id["n1"]["type"] == "agent"
        assert any("codex" in w and "Hermes agent node" in w for w in warnings)
        assert "Original agent backend: codex" in by_id["n1"]["data"]["context"]

    def test_condition_edge_kept_with_warning(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="A", input_text="a"),
                studio_node("n2", title="B", input_text="b"),
            ],
            edges=[studio_edge("e1", "n1", "n2", condition={"path": "count", "operator": ">", "value": 5})],
        )
        blueprint, warnings = convert_studio_envelope(env)
        edge_pairs = {(e["source"], e["target"]) for e in blueprint["edges"]}
        assert ("n1", "n2") in edge_pairs
        assert any("condition" in w for w in warnings)

    def test_feedback_loop_edge_dropped(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="A", input_text="a"),
                studio_node("n2", title="B", input_text="b"),
            ],
            edges=[
                studio_edge("e1", "n1", "n2"),
                studio_edge("e2", "n2", "n1", feedback={"maxIterations": 3, "loopId": "loop1"}),
            ],
        )
        blueprint, warnings = convert_studio_envelope(env)
        edge_pairs = {(e["source"], e["target"]) for e in blueprint["edges"]}
        assert ("n1", "n2") in edge_pairs
        assert ("n2", "n1") not in edge_pairs
        assert any("feedback" in w for w in warnings)

    def test_any_join_downgrade_warns(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="A", input_text="a", join="any"),
            ]
        )
        blueprint, warnings = convert_studio_envelope(env)
        assert any("join='any'" in w for w in warnings)

    def test_name_override(self):
        blueprint, _ = convert_studio_envelope(SIMPLE_ENVELOPE, name_override="Custom Name")
        assert blueprint["name"] == "custom-name"

    def test_multi_start_and_end_synthesis(self):
        env = make_envelope(
            nodes=[
                studio_node("n1", title="A", input_text="a"),
                studio_node("n2", title="B", input_text="b"),
                studio_node("n3", title="C", input_text="c"),
            ],
            edges=[studio_edge("e1", "n1", "n3")],
        )
        blueprint, _ = convert_studio_envelope(env)
        edge_pairs = {(e["source"], e["target"]) for e in blueprint["edges"]}
        # n1 is fed by start; n2 (no incoming) is also fed by start
        assert ("start", "n1") in edge_pairs
        assert ("start", "n2") in edge_pairs
        # n2 and n3 (no outgoing) feed end
        assert ("n2", "end") in edge_pairs
        assert ("n3", "end") in edge_pairs


# ---------------------------------------------------------------------------
# File import path
# ---------------------------------------------------------------------------

class TestImportFile:
    def test_import_writes_blueprint_file(self, hermes_home, tmp_path):
        source = tmp_path / "export.json"
        source.write_text(json.dumps(SIMPLE_ENVELOPE), encoding="utf-8")
        out_path, n_nodes, n_edges, warnings = import_studio_file(source)
        assert out_path == workflows_dir() / "research-flow.json"
        assert out_path.exists()
        assert n_nodes == 4  # start + 2 agents + end
        assert n_edges == 3
        assert warnings == []
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(payload.keys()) == {"nodes", "edges"}
        assert len(payload["nodes"]) == 4

    def test_import_refuses_overwrite(self, hermes_home, tmp_path):
        source = tmp_path / "export.json"
        source.write_text(json.dumps(SIMPLE_ENVELOPE), encoding="utf-8")
        import_studio_file(source)
        with pytest.raises(WorkflowImportError, match="already exists"):
            import_studio_file(source)

    def test_import_print_only_no_write(self, hermes_home, tmp_path, capsys):
        source = tmp_path / "export.json"
        source.write_text(json.dumps(SIMPLE_ENVELOPE), encoding="utf-8")
        import_studio_file(source, print_only=True)
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert len(payload["nodes"]) == 4
        assert not (workflows_dir() / "research-flow.json").exists()

    def test_import_custom_out_dir(self, hermes_home, tmp_path):
        source = tmp_path / "export.json"
        source.write_text(json.dumps(SIMPLE_ENVELOPE), encoding="utf-8")
        out_dir = tmp_path / "blueprints"
        out_path, _, _, _ = import_studio_file(source, out_dir=out_dir)
        assert out_path == out_dir / "research-flow.json"
        assert out_path.exists()

    def test_import_missing_file(self, hermes_home, tmp_path):
        with pytest.raises(WorkflowImportError, match="cannot read"):
            import_studio_file(tmp_path / "nope.json")

    def test_import_invalid_json(self, hermes_home, tmp_path):
        source = tmp_path / "bad.json"
        source.write_text("{not json", encoding="utf-8")
        with pytest.raises(WorkflowImportError, match="not valid JSON"):
            import_studio_file(source)

    def test_name_override_controls_filename(self, hermes_home, tmp_path):
        source = tmp_path / "export.json"
        source.write_text(json.dumps(SIMPLE_ENVELOPE), encoding="utf-8")
        out_path, _, _, _ = import_studio_file(source, name="Renamed Flow")
        assert out_path.name == "renamed-flow.json"
