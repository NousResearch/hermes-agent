"""Tests for the Roadmaps plan parser (T5c, Vision session).

``parse_plan`` turns the agent's strict JSON plan document (or the
documented, less-reliable Markdown fallback) into a validated, normalized
payload ready for ``RoadmapsWriter.create_plan``.  Validation errors raise
:class:`PlanParseError` with structured messages (field, index, and
line/column when available) so the Vision agent can rework the plan guided
by the error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import projects_db
from hermes_cli import roadmaps_plan_parser as plan_parser_mod
from hermes_cli.roadmaps_plan_parser import PlanParseError, parse_plan
from hermes_cli.roadmaps_writer import RoadmapsWriter

VALID_PLAN = """{
  "title": "Plan Vision",
  "purpose": "La meilleure voie pour la roadmap",
  "nodes": [
    {"node_id": "obj", "kind": "objective", "title": "Objectif"},
    {"node_id": "ph1", "kind": "phase", "title": "Phase 1", "parent_node_id": "obj"},
    {"node_id": "ms1", "kind": "milestone", "title": "Jalon 1", "parent_node_id": "ph1"},
    {"node_id": "st1", "kind": "step", "title": "Étape 1", "parent_node_id": "ms1",
     "state": "ready", "progress": 20}
  ],
  "relations": [
    {"relation_id": "r1", "from_node_id": "st1", "to_node_id": "ms1",
     "kind": "depends_on", "reason": "le jalon gate l'étape"}
  ],
  "todos": [
    {"todo_id": "t1", "node_id": "st1", "title": "Faire X", "position": 0},
    {"todo_id": "t2", "title": "Tâche globale"}
  ]
}"""


def _node_ids(payload: dict) -> list[str]:
    return [n["node_id"] for n in payload["nodes"]]


# ── strict JSON: happy paths ────────────────────────────────────────────────


def test_parse_plan_strict_json_ok():
    payload = parse_plan(VALID_PLAN)
    assert payload["title"] == "Plan Vision"
    assert payload["purpose"] == "La meilleure voie pour la roadmap"
    assert _node_ids(payload) == ["obj", "ph1", "ms1", "st1"]
    assert len(payload["relations"]) == 1
    assert len(payload["todos"]) == 2
    # Payload carries the source + actor for the create_plan call.
    assert payload["source"] == "vision"
    assert payload["actor"] == "user"


def test_parse_plan_json_in_code_fence():
    text = "Voici le plan :\n```json\n" + VALID_PLAN + "\n```\nFin."
    payload = parse_plan(text)
    assert payload["title"] == "Plan Vision"
    assert _node_ids(payload) == ["obj", "ph1", "ms1", "st1"]


def test_parse_plan_pure_json_text_without_fence():
    payload = parse_plan(VALID_PLAN)
    assert payload["nodes"][0]["node_id"] == "obj"


def test_parse_plan_normalizes_defaults():
    text = """{
      "title": "P",
      "nodes": [
        {"node_id": "a", "kind": "step", "title": "A"}
      ],
      "relations": [],
      "todos": []
    }"""
    payload = parse_plan(text)
    node = payload["nodes"][0]
    assert node["state"] == "planned"
    assert node["progress"] == 0
    assert node["parent_node_id"] is None
    assert node["description"] is None
    assert node["owner_agent"] is None
    assert node["block_reason"] is None


def test_parse_plan_keeps_explicit_state_and_progress():
    payload = parse_plan(VALID_PLAN)
    st1 = [n for n in payload["nodes"] if n["node_id"] == "st1"][0]
    assert st1["state"] == "ready"
    assert st1["progress"] == 20
    relation = payload["relations"][0]
    assert relation["state"] == "active"
    assert relation["reason"] == "le jalon gate l'étape"
    todo = payload["todos"][0]
    assert todo["state"] == "open"
    assert todo["position"] == 0
    assert payload["todos"][1]["node_id"] is None


# ── strict JSON: validation rejections ──────────────────────────────────────


def test_parse_plan_rejects_empty_title():
    with pytest.raises(PlanParseError) as exc:
        parse_plan('{"title": "   ", "nodes": [], "relations": [], "todos": []}')
    assert "title" in str(exc.value)
    assert exc.value.field == "title"


def test_parse_plan_rejects_missing_title():
    with pytest.raises(PlanParseError) as exc:
        parse_plan('{"nodes": [], "relations": [], "todos": []}')
    assert "title" in str(exc.value)


def test_parse_plan_rejects_invalid_kind():
    text = '{"title": "P", "nodes": [{"node_id": "a", "kind": "bogus", "title": "A"}]}'
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "kind" in str(exc.value)
    assert exc.value.field == "nodes[0].kind"


def test_parse_plan_rejects_unknown_parent():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "phase", "title": "A", '
            '"parent_node_id": "ghost"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "parent" in str(exc.value).lower()
    assert exc.value.field == "nodes[0].parent_node_id"


def test_parse_plan_rejects_self_parent():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "phase", "title": "A", '
            '"parent_node_id": "a"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "itself" in str(exc.value) or "self" in str(exc.value).lower()


def test_parse_plan_rejects_parent_cycle():
    text = ('{"title": "P", "nodes": ['
            '{"node_id": "a", "kind": "phase", "title": "A", "parent_node_id": "b"},'
            '{"node_id": "b", "kind": "phase", "title": "B", "parent_node_id": "a"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "cycle" in str(exc.value).lower()


def test_parse_plan_rejects_orphan_relation():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"}], '
            '"relations": [{"relation_id": "r1", "from_node_id": "a", '
            '"to_node_id": "ghost", "kind": "depends_on"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "relation" in str(exc.value).lower()
    assert exc.value.field == "relations[0].to_node_id"


def test_parse_plan_rejects_self_relation():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"}], '
            '"relations": [{"relation_id": "r1", "from_node_id": "a", '
            '"to_node_id": "a", "kind": "depends_on"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "same" in str(exc.value).lower() or "differ" in str(exc.value).lower()


def test_parse_plan_rejects_invalid_relation_kind():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"},'
            '{"node_id": "b", "kind": "step", "title": "B"}], '
            '"relations": [{"relation_id": "r1", "from_node_id": "a", '
            '"to_node_id": "b", "kind": "mystifies"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "relations[0].kind"


def test_parse_plan_rejects_unknown_todo_node():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"}], '
            '"todos": [{"todo_id": "t1", "node_id": "ghost", "title": "Do"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "todo" in str(exc.value).lower()
    assert exc.value.field == "todos[0].node_id"


def test_parse_plan_rejects_duplicate_node_id():
    text = ('{"title": "P", "nodes": ['
            '{"node_id": "a", "kind": "step", "title": "A"},'
            '{"node_id": "a", "kind": "step", "title": "B"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "duplicate" in str(exc.value).lower()
    assert exc.value.field == "nodes[1].node_id"


def test_parse_plan_rejects_duplicate_relation_id():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"},'
            '{"node_id": "b", "kind": "step", "title": "B"}], '
            '"relations": ['
            '{"relation_id": "r1", "from_node_id": "a", "to_node_id": "b", "kind": "depends_on"},'
            '{"relation_id": "r1", "from_node_id": "b", "to_node_id": "a", "kind": "depends_on"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "duplicate" in str(exc.value).lower()
    assert exc.value.field == "relations[1].relation_id"


def test_parse_plan_rejects_empty_node_title():
    text = '{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "  "}]}'
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "nodes[0].title"


def test_parse_plan_rejects_blank_node_id():
    text = '{"title": "P", "nodes": [{"node_id": "  ", "kind": "step", "title": "A"}]}'
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "nodes[0].node_id"


def test_parse_plan_rejects_invalid_state():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A", '
            '"state": "flying"}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "nodes[0].state"


def test_parse_plan_rejects_invalid_progress():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A", '
            '"progress": 150}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "nodes[0].progress"


def test_parse_plan_rejects_negative_position():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"}], '
            '"todos": [{"todo_id": "t1", "node_id": "a", "title": "Do", "position": -1}]}')
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "todos[0].position"


def test_parse_plan_rejects_non_json():
    with pytest.raises(PlanParseError) as exc:
        parse_plan("Ceci n'est pas du JSON du tout, juste du texte bavard.")
    assert exc.value.line is None  # semantic failure, not a JSON syntax error
    assert "json" in str(exc.value).lower() or "plan" in str(exc.value).lower()


def test_parse_plan_json_syntax_error_carries_line_and_column():
    text = '{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A",}]}'
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.line is not None
    assert exc.value.column is not None
    assert "line" in str(exc.value).lower()


# ── normalization: generated ids + content dedupe ───────────────────────────


def test_parse_plan_generates_missing_node_ids_with_n_prefix():
    text = ('{"title": "P", "nodes": ['
            '{"kind": "objective", "title": "Obj"},'
            '{"kind": "step", "title": "S", "parent_node_id": "n_0001"}]}')
    payload = parse_plan(text)
    ids = _node_ids(payload)
    assert len(set(ids)) == 2
    assert all(i.startswith("n_") for i in ids)
    assert payload["nodes"][1]["parent_node_id"] == payload["nodes"][0]["node_id"]


def test_parse_plan_generates_missing_relation_and_todo_ids():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"},'
            '{"node_id": "b", "kind": "step", "title": "B"}], '
            '"relations": [{"from_node_id": "a", "to_node_id": "b", "kind": "depends_on"}], '
            '"todos": [{"node_id": "a", "title": "Do"}]}')
    payload = parse_plan(text)
    assert payload["relations"][0]["relation_id"].startswith("r_")
    assert payload["todos"][0]["todo_id"].startswith("t_")
    assert payload["todos"][0]["state"] == "open"
    assert payload["todos"][0]["position"] == 0


def test_parse_plan_keeps_provided_ids_untouched():
    payload = parse_plan(VALID_PLAN)
    assert _node_ids(payload) == ["obj", "ph1", "ms1", "st1"]
    assert payload["relations"][0]["relation_id"] == "r1"
    assert [t["todo_id"] for t in payload["todos"]] == ["t1", "t2"]


def test_parse_plan_dedupes_identical_relations():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"},'
            '{"node_id": "b", "kind": "step", "title": "B"}], '
            '"relations": ['
            '{"relation_id": "r1", "from_node_id": "a", "to_node_id": "b", "kind": "depends_on"},'
            '{"relation_id": "r2", "from_node_id": "a", "to_node_id": "b", "kind": "depends_on"}]}')
    payload = parse_plan(text)
    assert len(payload["relations"]) == 1
    assert payload["relations"][0]["relation_id"] == "r1"


def test_parse_plan_dedupes_identical_todos():
    text = ('{"title": "P", "nodes": [{"node_id": "a", "kind": "step", "title": "A"}], '
            '"todos": ['
            '{"todo_id": "t1", "node_id": "a", "title": "Do"},'
            '{"todo_id": "t2", "node_id": "a", "title": "Do"}]}')
    payload = parse_plan(text)
    assert len(payload["todos"]) == 1
    assert payload["todos"][0]["todo_id"] == "t1"


# ── markdown fallback (documented, less reliable) ───────────────────────────


def test_parse_plan_markdown_fallback_documented_behavior():
    """The agent MUST emit strict JSON; Markdown is a documented fallback.

    Grammar (documented): ``# Title`` plan title, ``## <kind>: <title>``
    nodes (kind objective|phase|milestone|step|decision; heading level also
    maps 2→phase, 3→milestone, 4→step, 5→decision), ``- [ ]`` todos
    attached to the most recent node, ``- <kind>: A -> B`` relations
    resolved by unique node title.  IDs are generated deterministically.
    """
    text = """# Plan Markdown
## objective: Objectif
## phase: Phase 1
### milestone: Jalon 1
#### step: Étape 1
- [ ] Faire X
- [x] Déjà fait
- depends_on: Étape 1 -> Jalon 1
"""
    payload = parse_plan(text)
    assert payload["title"] == "Plan Markdown"
    kinds = [n["kind"] for n in payload["nodes"]]
    assert kinds == ["objective", "phase", "milestone", "step"]
    assert payload["nodes"][3]["parent_node_id"] == payload["nodes"][2]["node_id"]
    assert [t["title"] for t in payload["todos"]] == ["Faire X", "Déjà fait"]
    assert payload["todos"][0]["state"] == "open"
    assert payload["todos"][1]["state"] == "done"
    assert payload["todos"][0]["node_id"] == payload["nodes"][3]["node_id"]
    assert payload["relations"][0]["kind"] == "depends_on"
    assert payload["relations"][0]["from_node_id"] == payload["nodes"][3]["node_id"]
    assert payload["relations"][0]["to_node_id"] == payload["nodes"][2]["node_id"]


def test_parse_plan_markdown_fallback_requires_nodes():
    with pytest.raises(PlanParseError) as exc:
        parse_plan("# Just a title, no nodes at all")
    assert "node" in str(exc.value).lower()


# ── payload shape ready for plans.create ────────────────────────────────────


def test_parse_plan_payload_is_ready_for_create_plan():
    payload = parse_plan(VALID_PLAN, source="vision-test", default_actor="pierre")
    # Every node/relation/todo key the writer's validator normalizes is present.
    for node in payload["nodes"]:
        for key in ("node_id", "kind", "title", "description", "parent_node_id",
                    "state", "progress", "owner_agent", "block_reason"):
            assert key in node
    for relation in payload["relations"]:
        for key in ("relation_id", "from_node_id", "to_node_id", "kind",
                    "state", "reason"):
            assert key in relation
    for todo in payload["todos"]:
        for key in ("todo_id", "node_id", "title", "state", "position"):
            assert key in todo
    assert payload["source"] == "vision-test"
    assert payload["actor"] == "pierre"


# ── integration: parse → RoadmapsWriter.create_plan on a real temp DB ───────


def seed(path: Path) -> None:
    conn = projects_db.connect(path)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p', 'p', 'P', 1)")
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("prof", "p", "r", "Roadmap", None, "draft", None, "creator", "creator", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("prof", "p", "r", 1, "draft", "seed", None, "creator", 1, None),
    )
    conn.commit()
    conn.close()


def test_parse_plan_to_create_plan_integration(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path)
    payload = parse_plan(VALID_PLAN)
    result = RoadmapsWriter(path).create_plan(
        "prof", "p", "r", payload["actor"],
        nodes=payload["nodes"], relations=payload["relations"],
        todos=payload["todos"], source=payload["source"],
    )
    assert result["success"] is True
    assert result["version"] == 2  # version 1 reserved by roadmaps.create
    assert result["state"] == "proposed"
    assert result["counts"] == {"nodes": 4, "relations": 1, "todos": 2}
    # The version's content is persisted: read it back from the store.
    conn = projects_db.connect(path)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM roadmap_nodes "
        "WHERE profile_id='prof' AND project_id='p' AND roadmap_id='r' AND version=2"
    ).fetchone()["n"]
    conn.close()
    assert n == 4


# ── hardening: input limits, recursion safety, O(n) markdown (gate 2026-08-15) ──


def test_parse_plan_rejects_oversized_input():
    purpose = "x" * (plan_parser_mod.MAX_PLAN_TEXT_BYTES + 1)
    text = json.dumps(
        {"title": "P", "purpose": purpose, "nodes": [], "relations": [], "todos": []}
    )
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "input"
    assert "limit" in str(exc.value) or "bytes" in str(exc.value)


def test_parse_plan_accepts_input_just_under_size_limit():
    # The size gate must be off-by-one safe: a document just under the limit
    # reaches semantic validation (fails on empty nodes, not on size).
    purpose = "x" * (plan_parser_mod.MAX_PLAN_TEXT_BYTES - 512)  # wrapper headroom
    text = json.dumps(
        {"title": "P", "purpose": purpose, "nodes": [], "relations": [], "todos": []}
    )
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field != "input"


def test_parse_plan_rejects_deeply_nested_json():
    for brackets in ("[]", "{}"):
        text = brackets[0] * 5000 + brackets[1] * 5000
        with pytest.raises(PlanParseError) as exc:
            parse_plan(text)
        assert exc.value.field == "input"
        assert "nested" in str(exc.value).lower()


def _reversed_chain_nodes(count: int) -> list[dict]:
    # Declared deepest-first so DFS starts at the bottom of the parent chain:
    # the recursive cycle detection used to overflow the stack on this shape.
    return [
        {
            "node_id": f"n{i}",
            "kind": "step",
            "title": f"S{i}",
            "parent_node_id": None if i == 0 else f"n{i-1}",
        }
        for i in range(count - 1, -1, -1)
    ]


def test_parse_plan_long_parent_chain_does_not_overflow_stack(monkeypatch):
    monkeypatch.setattr(plan_parser_mod, "MAX_PLAN_NODES", 10_000)
    text = json.dumps(
        {"title": "P", "nodes": _reversed_chain_nodes(3000), "relations": [], "todos": []}
    )
    payload = parse_plan(text)  # a RecursionError would fail the test
    assert len(payload["nodes"]) == 3000


def test_parse_plan_long_parent_chain_cycle_is_structured(monkeypatch):
    monkeypatch.setattr(plan_parser_mod, "MAX_PLAN_NODES", 10_000)
    nodes = _reversed_chain_nodes(3000)
    nodes[-1]["parent_node_id"] = "n2999"  # close the 3000-long cycle
    text = json.dumps({"title": "P", "nodes": nodes, "relations": [], "todos": []})
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert "cycle" in str(exc.value).lower()
    assert exc.value.field == "parent_node_id"
    assert exc.value.index is not None


def test_parse_plan_rejects_too_many_nodes():
    nodes = [
        {"node_id": f"n{i}", "kind": "step", "title": f"S{i}"}
        for i in range(plan_parser_mod.MAX_PLAN_NODES + 1)
    ]
    text = json.dumps({"title": "P", "nodes": nodes, "relations": [], "todos": []})
    with pytest.raises(PlanParseError) as exc:
        parse_plan(text)
    assert exc.value.field == "input"
    assert "node" in str(exc.value).lower()


def test_parse_plan_rejects_too_many_markdown_nodes():
    lines = ["# P", "## objective: Obj"]
    lines += [f"### step: S{i}" for i in range(plan_parser_mod.MAX_PLAN_NODES + 1)]
    with pytest.raises(PlanParseError) as exc:
        parse_plan("\n".join(lines))
    assert exc.value.field == "input"
    assert "node" in str(exc.value).lower()


def test_parse_plan_markdown_scales_linearly(monkeypatch):
    monkeypatch.setattr(plan_parser_mod, "MAX_PLAN_NODES", 20_000)
    count = 8_000
    lines = ["# Plan Massif"] + [f"## step: Étape {i}" for i in range(count)]
    start = time.monotonic()
    payload = parse_plan("\n".join(lines))
    elapsed = time.monotonic() - start
    assert len(payload["nodes"]) == count
    assert len({n["node_id"] for n in payload["nodes"]}) == count
    # Loose bound on purpose: the O(n^2) id-generation loop took ~7.6s at
    # 8k nodes; the O(n) version finishes in well under a second.  The
    # structural asserts (count + unique ids) are the primary contract.
    assert elapsed < 5.0, f"markdown parse took {elapsed:.2f}s — expected O(n)"
