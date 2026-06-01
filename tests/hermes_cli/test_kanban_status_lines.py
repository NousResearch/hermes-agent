from hermes_cli.kanban_status_lines import (
    TaskStatusInput,
    descendants_for,
    kanban_status_to_live,
    render_status_lines,
)


REGISTRY = {
    "project": "hermes-maintenance",
    "profiles": {
        "pa_yunuen": {"children": ["coordinator_hermes_maintenance"]},
        "coordinator_hermes_maintenance": {
            "children": ["responsible_kanban_semantics", "responsible_matrix_hardening"]
        },
        "responsible_kanban_semantics": {"children": []},
        "responsible_matrix_hardening": {"children": []},
    },
    "dependencies": {"tasks": {}},
}


def test_kanban_status_to_live_uses_four_notification_states():
    assert kanban_status_to_live("ready") == "working"
    assert kanban_status_to_live("queue") == "working"
    assert kanban_status_to_live("running") == "working"
    assert kanban_status_to_live("in_progress") == "working"
    assert kanban_status_to_live("review") == "working"
    assert kanban_status_to_live("todo") == "waiting"
    assert kanban_status_to_live("scheduled") == "waiting"
    assert kanban_status_to_live("blocked") == "blocked"
    assert kanban_status_to_live("triage") == "dormant"
    assert kanban_status_to_live("done") is None
    assert kanban_status_to_live("archived") is None


def test_descendants_are_structural_not_dependencies():
    assert descendants_for("coordinator_hermes_maintenance", REGISTRY) == [
        "responsible_kanban_semantics",
        "responsible_matrix_hardening",
    ]


def test_render_status_lines_uses_generic_self_and_lineage_labels_only():
    tasks = [
        TaskStatusInput(
            id="t_self",
            assignee="coordinator_hermes_maintenance",
            status="running",
            title="coordinate",
        ),
        TaskStatusInput(
            id="t_child_wait",
            assignee="responsible_kanban_semantics",
            status="scheduled",
            title="non-human wait",
        ),
        TaskStatusInput(
            id="t_child_block",
            assignee="responsible_matrix_hardening",
            status="blocked",
            title="human action required",
        ),
    ]

    self_line, lineage_line, payload = render_status_lines(
        "coordinator_hermes_maintenance", REGISTRY, tasks
    )

    assert self_line.startswith("Self status: WORKING — ")
    assert lineage_line.startswith("Lineage status: BLOCKED — ")
    rendered = "\n".join([self_line, lineage_line])
    forbidden = [
        "Coordinator" + " status",
        "PA" + " status",
        "Responsible" + " status",
        "Blocker" + " status",
        "NOT" + " BLOCKED",
    ]
    for label in forbidden:
        assert label not in rendered
    assert payload["self_status"] == "working"
    assert payload["lineage_status"] == "blocked"
    assert payload["lineage_counts"] == {
        "working": 0,
        "waiting": 1,
        "blocked": 1,
        "dormant": 0,
    }
    assert payload["lineage_evidence"] == ["t_child_block"]


def test_lineage_working_precedence_keeps_blocked_count_visible():
    tasks = [
        TaskStatusInput("t_work", "responsible_kanban_semantics", "ready"),
        TaskStatusInput("t_block", "responsible_matrix_hardening", "blocked"),
    ]

    _self_line, lineage_line, payload = render_status_lines(
        "coordinator_hermes_maintenance", REGISTRY, tasks
    )

    assert lineage_line.startswith("Lineage status: WORKING — ")
    assert "1 descendant blocked" in lineage_line
    assert payload["lineage_counts"]["blocked"] == 1


def test_taskless_profile_renders_dormant_not_not_blocked():
    self_line, lineage_line, payload = render_status_lines(
        "responsible_kanban_semantics", REGISTRY, []
    )

    assert self_line.startswith("Self status: DORMANT — ")
    assert lineage_line.startswith("Lineage status: DORMANT — ")
    assert "NOT" + " BLOCKED" not in self_line + lineage_line
    assert payload["self_status"] == "dormant"
    assert payload["lineage_status"] == "dormant"
