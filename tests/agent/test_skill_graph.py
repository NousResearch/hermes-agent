"""Tests for boring Skill Graphs v0."""

import json

import pytest

from agent.skill_graph import (
    build_skill_graph,
    get_skill_graph_canary_fixtures,
    lint_skill_graph,
    query_skill_graph,
    render_skill_graph_markdown,
    run_skill_graph_canaries,
)


def _make_skill(
    root,
    category,
    name,
    description,
    *,
    layer="atom",
    depends_on=None,
    reads=None,
    writes=None,
    verifies=None,
    approval_gate=None,
    body="Use the exact workflow and verify the result.",
):
    skill_dir = root / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    depends_on = depends_on or []
    reads = reads or []
    writes = writes or []
    verifies = verifies or []

    def _yaml_list(values, indent="        "):
        if not values:
            return "[]"
        lines = []
        for item in values:
            if isinstance(item, dict):
                lines.append(f"\n{indent}- skill: {item['skill']}")
                if item.get("layer"):
                    lines.append(f"{indent}  layer: {item['layer']}")
                if item.get("when"):
                    lines.append(f"{indent}  when: {item['when']}")
                if item.get("mode"):
                    lines.append(f"{indent}  mode: {item['mode']}")
            else:
                lines.append(f"\n{indent}- {item}")
        return "".join(lines)

    approval_line = f"      approval_gate: {approval_gate}\n" if approval_gate else ""
    content = f"""---
name: {name}
description: {description}
metadata:
  hermes:
    layer: {layer}
    depends_on: {_yaml_list(depends_on)}
    reads: {_yaml_list(reads)}
    writes: {_yaml_list(writes)}
    verifies: {_yaml_list(verifies)}
{approval_line}---

# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _opik_graph(root):
    _make_skill(
        root,
        "devops",
        "memory-system-monitor",
        "Monitor Hermes memory system health and telemetry.",
        layer="atom",
        verifies=["local_report"],
    )
    _make_skill(
        root,
        "devops",
        "opik-hourly-watch-canonical-runbook",
        "Run the Hermes Opik hourly watch and reconcile tool telemetry.",
        layer="compound",
        depends_on=[
            {
                "skill": "memory-system-monitor",
                "layer": "atom",
                "when": "memory telemetry must be reconciled",
                "mode": "explicit",
            }
        ],
        reads=["opik_traces", "state_db"],
        writes=["local_report", "wiki"],
        verifies=["trace_counts", "tool_counts"],
    )
    _make_skill(
        root,
        "devops",
        "hermes-opik-hourly-watch-runbook",
        "Reproducible 90-minute Opik hourly watch workflow with exact-window REST paging.",
        layer="atom",
    )
    _make_skill(
        root,
        "general",
        "hermes-opik-hourly-iteration-cap-recovery",
        "Continue and finalize Hermes Opik hourly watch reports when automated analysis hits tool-call limits or run interruption.",
        layer="atom",
    )
    _make_skill(
        root,
        "jetstream-ops",
        "qbo-create-customer",
        "Create or detect QuickBooks customers for JetStream billing.",
        layer="compound",
        reads=["qbo"],
        writes=["qbo"],
        approval_gate="tyler_approval",
    )
    _make_skill(
        root,
        "marketing",
        "social-content",
        "Create social media drafts for JetStream channels.",
        layer="molecule",
        writes=["draft_social"],
    )
    return build_skill_graph(root, generated_at="2026-04-23T12:00:00Z")


class TestSkillGraphManifest:
    def test_manifest_is_deterministic_and_contains_typed_edges(self, tmp_path):
        first = _opik_graph(tmp_path)
        second = _opik_graph(tmp_path)

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first["schema_version"] == 1
        assert first["counts"] == {"nodes": 6, "edges": 1}

        node_ids = [node["id"] for node in first["nodes"]]
        assert node_ids == sorted(node_ids)
        opik = next(
            node for node in first["nodes"]
            if node["name"] == "opik-hourly-watch-canonical-runbook"
        )
        assert opik["id"] == "local:devops/opik-hourly-watch-canonical-runbook"
        assert opik["layer"] == "compound"
        assert opik["source_kind"] == "local"
        assert opik["max_dependency_depth"] == 1
        assert opik["writes"] == ["local_report", "wiki"]

        assert first["edges"] == [
            {
                "from": "local:devops/opik-hourly-watch-canonical-runbook",
                "to": "local:devops/memory-system-monitor",
                "mode": "explicit",
                "when": "memory telemetry must be reconciled",
                "layer": "atom",
            }
        ]

    def test_markdown_manifest_is_stable_and_readable(self, tmp_path):
        graph = _opik_graph(tmp_path)
        markdown = render_skill_graph_markdown(graph)

        assert markdown.startswith("# Hermes Skill Graph")
        assert "## Counts" in markdown
        assert "local:devops/opik-hourly-watch-canonical-runbook" in markdown
        assert "local:devops/memory-system-monitor" in markdown
        assert markdown.index("local:devops/memory-system-monitor") < markdown.index(
            "local:devops/opik-hourly-watch-canonical-runbook"
        )

    def test_nested_graph_metadata_supports_ids_and_suggests(self, tmp_path):
        _make_skill(tmp_path, "devops", "opik-hourly-window-scripting", "Exact window pulls")
        _make_skill(tmp_path, "devops", "opik-hourly-fetch-reliability", "Fetch fallback checks")
        skill_dir = tmp_path / "devops" / "opik-hourly-watch-canonical-runbook"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: opik-hourly-watch-canonical-runbook
description: Reusable notes for generating and validating Hermes Opik hourly watch reports.
metadata:
  hermes:
    graph:
      layer: compound
      depends_on:
        - id: local:devops/opik-hourly-window-scripting
          when: Need exact window pulls
          mode: explicit
      suggests:
        - id: local:devops/opik-hourly-fetch-reliability
          when: Fetches are inconsistent
      reads:
        - opik traces
      writes:
        - local report markdown
      verifies:
        - trace count reconciliation
      side_effects: local_write
      approval_required: false
      max_dependency_depth: 1
---
# Canonical
""",
            encoding="utf-8",
        )

        graph = build_skill_graph(tmp_path, generated_at="2026-04-23T12:00:00Z")

        opik = next(node for node in graph["nodes"] if node["name"] == "opik-hourly-watch-canonical-runbook")
        assert opik["layer"] == "compound"
        assert opik["reads"] == ["opik traces"]
        assert opik["side_effects"] == "local_write"
        assert graph["counts"] == {"nodes": 3, "edges": 2}
        assert {edge["mode"] for edge in graph["edges"]} == {"explicit", "advisory"}


class TestSkillGraphLint:
    def test_lint_is_warning_first_for_broken_graphs(self):
        graph = {
            "schema_version": 1,
            "nodes": [
                {
                    "id": "local:devops/a",
                    "name": "a",
                    "layer": "compound",
                    "depends_on": ["missing-skill"],
                    "writes": ["public_social"],
                    "approval_gate": None,
                },
                {"id": "local:devops/b", "name": "b", "layer": "banana", "depends_on": []},
                {"id": "local:devops/b", "name": "b-copy", "layer": "atom", "depends_on": []},
            ],
            "edges": [
                {"from": "local:devops/a", "to": "local:devops/missing-skill", "mode": "explicit"},
                {"from": "local:devops/b", "to": "local:devops/a", "mode": "explicit"},
                {"from": "local:devops/a", "to": "local:devops/b", "mode": "explicit"},
            ],
        }

        report = lint_skill_graph(graph, strict=False)

        assert report["ok"] is False
        assert report["errors"] == []
        codes = {warning["code"] for warning in report["warnings"]}
        assert {
            "duplicate_node_id",
            "invalid_layer",
            "missing_dependency",
            "cycle",
            "approval_gate_missing",
        } <= codes

    def test_strict_lint_promotes_warnings_to_errors(self):
        graph = {
            "schema_version": 1,
            "nodes": [{"id": "local:x/bad", "name": "bad", "layer": "banana"}],
            "edges": [],
        }

        report = lint_skill_graph(graph, strict=True)

        assert report["ok"] is False
        assert report["warnings"] == []
        assert report["errors"][0]["code"] == "invalid_layer"


class TestSkillGraphQuery:
    def test_query_ranks_by_text_and_returns_one_hop_dependencies(self, tmp_path):
        graph = _opik_graph(tmp_path)

        results = query_skill_graph(graph, "run Hermes Opik hourly watch", limit=3)

        assert results[0]["name"] == "opik-hourly-watch-canonical-runbook"
        assert results[0]["dependency_suggestions"] == [
            {
                "skill": "memory-system-monitor",
                "id": "local:devops/memory-system-monitor",
                "mode": "explicit",
                "when": "memory telemetry must be reconciled",
            }
        ]
        assert all("dependency_suggestions" in result for result in results)

    def test_generic_query_prefers_canonical_runbook_over_recovery_notes(self, tmp_path):
        graph = _opik_graph(tmp_path)

        results = query_skill_graph(graph, "run Hermes Opik hourly watch", limit=3)

        assert results[0]["name"] == "opik-hourly-watch-canonical-runbook"
        assert results[0]["name"] != "hermes-opik-hourly-iteration-cap-recovery"

    def test_recovery_query_can_rank_recovery_runbook_first(self, tmp_path):
        graph = _opik_graph(tmp_path)

        results = query_skill_graph(
            graph,
            "recover interrupted Hermes Opik hourly watch after tool call cap",
            limit=3,
        )

        assert results[0]["name"] == "hermes-opik-hourly-iteration-cap-recovery"

    def test_query_does_not_walk_dependencies_recursively(self, tmp_path):
        _make_skill(tmp_path, "devops", "leaf", "Leaf verifier", layer="atom")
        _make_skill(
            tmp_path,
            "devops",
            "middle",
            "Middle verifier",
            layer="molecule",
            depends_on=[{"skill": "leaf", "mode": "explicit"}],
        )
        _make_skill(
            tmp_path,
            "devops",
            "root",
            "Root Opik canary workflow",
            layer="compound",
            depends_on=[{"skill": "middle", "mode": "explicit"}],
        )
        graph = build_skill_graph(tmp_path, generated_at="2026-04-23T12:00:00Z")

        results = query_skill_graph(graph, "root opik canary", limit=1)

        suggestions = results[0]["dependency_suggestions"]
        assert [item["skill"] for item in suggestions] == ["middle"]
        assert "leaf" not in json.dumps(suggestions)


class TestSkillGraphCanaries:
    def test_default_canary_fixture_covers_opik_hourly_watch(self, tmp_path):
        graph = _opik_graph(tmp_path)
        fixtures = get_skill_graph_canary_fixtures()

        assert fixtures[0]["id"] == "opik-hourly-watch"
        assert fixtures[0]["side_effects"] == "internal_only"

        report = run_skill_graph_canaries(graph, fixtures)

        assert report["ok"] is True
        assert report["results"][0]["matched"] is True
        assert report["results"][0]["top_skill"] == "opik-hourly-watch-canonical-runbook"
