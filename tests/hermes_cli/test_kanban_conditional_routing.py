import argparse
from concurrent.futures import ThreadPoolExecutor

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from tools import kanban_tools


def _board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return path


def test_conditional_outcomes_promote_only_matching_routes(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    cases = {
        "direction_accepted": "review",
        "bounded_rework": "rework",
        "owner_judgment_required": "owner",
        "stop_wrong_direction": None,
    }

    with kb.connect(path) as conn:
        for outcome, expected_route in cases.items():
            parent = kb.create_task(conn, title=f"decision-{outcome}")
            routes = {
                route: kb.create_task(
                    conn,
                    title=f"{outcome}-{route}",
                    parents=[parent],
                    parent_outcomes={parent: [required]},
                )
                for route, required in {
                    "review": "direction_accepted",
                    "rework": "bounded_rework",
                    "owner": "owner_judgment_required",
                }.items()
            }

            assert kb.complete_task(conn, parent, outcome=outcome)
            assert kb.recompute_ready(conn) == 0
            for route, task_id in routes.items():
                expected = "ready" if route == expected_route else "todo"
                assert kb.get_task(conn, task_id).status == expected


def test_review_required_completes_implementation_and_unlocks_reviewer(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        implementation = kb.create_task(conn, title="implement")
        reviewer = kb.create_task(
            conn,
            title="review",
            parents=[implementation],
            parent_outcomes={implementation: ["review_required"]},
        )

        assert kb.complete_task(
            conn,
            implementation,
            outcome="review_required",
            summary="ready for independent review",
        )
        assert kb.get_task(conn, implementation).status == "done"
        assert kb.get_task(conn, implementation).completion_outcome == "review_required"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, reviewer).status == "ready"


def test_unconditional_dependency_remains_backward_compatible(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"

        assert kb.complete_task(conn, parent)
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "ready"


def test_relink_without_outcomes_preserves_existing_condition(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        parent = kb.create_task(conn, title="decision")
        child = kb.create_task(conn, title="rework")
        kb.link_tasks(conn, parent, child, when_outcomes=["bounded_rework"])

        kb.link_tasks(conn, parent, child)
        assert kb.unsatisfied_parent_dependencies(conn, child) == [{
            "id": parent,
            "status": "ready",
            "completion_outcome": None,
            "when_outcomes": ["bounded_rework"],
        }]


def test_relink_to_satisfied_outcome_immediately_promotes_child(
    tmp_path, monkeypatch
):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        parent = kb.create_task(conn, title="decision")
        child = kb.create_task(conn, title="review")
        assert kb.complete_task(conn, parent, outcome="direction_accepted")
        kb.link_tasks(conn, parent, child, when_outcomes=["bounded_rework"])
        assert kb.get_task(conn, child).status == "todo"

        kb.link_tasks(conn, parent, child, when_outcomes=["direction_accepted"])
        assert kb.get_task(conn, child).status == "ready"


def test_cli_legacy_link_call_omits_when_outcomes(monkeypatch):
    captured = []
    monkeypatch.setattr(kb, "connect_closing", lambda: _NullContext())
    monkeypatch.setattr(
        kb, "link_tasks", lambda *args, **kwargs: captured.append(kwargs)
    )

    args = argparse.Namespace(parent_id="parent", child_id="child")
    assert kanban_cli._cmd_link(args) == 0
    assert captured == [{}]


def test_archiving_does_not_synthesize_a_conditional_outcome(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        parent = kb.create_task(conn, title="decision")
        conditional = kb.create_task(
            conn,
            title="review",
            parents=[parent],
            parent_outcomes={parent: ["direction_accepted"]},
        )
        unconditional = kb.create_task(conn, title="cleanup", parents=[parent])

        assert kb.archive_task(conn, parent)
        assert kb.get_task(conn, conditional).status == "todo"
        assert kb.get_task(conn, unconditional).status == "ready"


def test_conditional_recompute_is_idempotent_across_connections(tmp_path, monkeypatch):
    path = _board(tmp_path, monkeypatch)
    with kb.connect(path) as conn:
        parent = kb.create_task(conn, title="decision")
        child = kb.create_task(
            conn,
            title="selected route",
            parents=[parent],
            parent_outcomes={parent: ["direction_accepted"]},
        )
        assert kb.complete_task(conn, parent, outcome="direction_accepted")
        # Put the satisfied child back into the state that concurrent dispatcher
        # ticks race to promote. This isolates recompute idempotency from the
        # automatic recompute performed by complete_task.
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
        conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'promoted'", (child,))
        conn.commit()

    def recompute():
        with kb.connect(path) as conn:
            return kb.recompute_ready(conn)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: recompute(), range(8)))

    assert sum(results) == 1
    with kb.connect(path) as conn:
        assert kb.get_task(conn, child).status == "ready"
        promoted_events = [
            event for event in kb.list_events(conn, child)
            if event.kind == "promoted"
        ]
        assert len(promoted_events) == 1
        assert kb.recompute_ready(conn) == 0


def test_public_tool_schemas_expose_conditional_routing_contract():
    complete = kanban_tools.KANBAN_COMPLETE_SCHEMA["parameters"]["properties"]
    create = kanban_tools.KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    link = kanban_tools.KANBAN_LINK_SCHEMA["parameters"]["properties"]

    assert complete["outcome"]["type"] == "string"
    assert create["parent_outcomes"]["additionalProperties"]["type"] == "array"
    assert link["when_outcomes"]["items"]["type"] == "string"


class _NullContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False
