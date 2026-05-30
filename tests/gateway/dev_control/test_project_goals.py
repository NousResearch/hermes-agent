import pytest

from gateway.dev_control.project_goals import (
    DevProjectGoalStore,
    abandon_project_goal,
    create_project_goal,
    get_project_goal_tree,
    list_project_goals,
    recompute_rollup,
)


@pytest.fixture
def store(tmp_path):
    goal_store = DevProjectGoalStore(tmp_path / "state.db")
    yield goal_store
    goal_store.close()


def _hierarchy(store, project_id="OrynWorkspace"):
    vision = create_project_goal(
        store=store,
        kind="vision",
        title="Product vision",
        project_id=project_id,
        status="active",
    )
    goal = create_project_goal(
        store=store,
        kind="goal",
        title="Q2 feature",
        project_id=project_id,
        parent_goal_id=vision["goal_id"],
        status="active",
    )
    milestone = create_project_goal(
        store=store,
        kind="milestone",
        title="Ship v1",
        project_id=project_id,
        parent_goal_id=goal["goal_id"],
        status="active",
    )
    sub_a = create_project_goal(
        store=store,
        kind="subgoal",
        title="API routes",
        project_id=project_id,
        parent_goal_id=milestone["goal_id"],
        status="active",
    )
    sub_b = create_project_goal(
        store=store,
        kind="subgoal",
        title="CLI surface",
        project_id=project_id,
        parent_goal_id=milestone["goal_id"],
        status="active",
    )
    return vision, goal, milestone, sub_a, sub_b


def test_create_vision_without_parent(store):
    created = create_project_goal(
        store=store,
        kind="vision",
        title="North star",
        project_id="OrynWorkspace",
    )
    assert created["object"] == "hermes.dev_project_goal"
    assert created["status"] == "proposed"
    assert created["progress"] == 0.0
    assert created["parent_goal_id"] is None


def test_reject_non_vision_without_parent(store):
    with pytest.raises(ValueError, match="require a parent_goal_id"):
        create_project_goal(store=store, kind="goal", title="Missing parent")


def test_reject_vision_with_parent(store):
    with pytest.raises(ValueError, match="must not have a parent_goal_id"):
        create_project_goal(
            store=store,
            kind="vision",
            title="Bad vision",
            parent_goal_id="parent-1",
        )


def test_reject_invalid_parent_kind(store):
    vision = create_project_goal(store=store, kind="vision", title="Vision")
    with pytest.raises(ValueError, match="parent must be kind=goal"):
        create_project_goal(
            store=store,
            kind="milestone",
            title="Bad milestone",
            parent_goal_id=vision["goal_id"],
        )


def test_list_filters_by_kind_and_status(store):
    _hierarchy(store)
    goals = list_project_goals(store=store, kind="subgoal", status="active")
    assert goals["total"] == 2


def test_tree_returns_nested_hierarchy(store):
    _hierarchy(store)
    tree = get_project_goal_tree(store=store, project_id="OrynWorkspace")
    assert tree["total"] == 5
    assert len(tree["roots"]) == 1
    root = tree["roots"][0]
    assert root["kind"] == "vision"
    assert root["children"][0]["children"][0]["children"][0]["kind"] == "subgoal"


def test_rollup_partial_progress(store):
    vision, goal, milestone, sub_a, sub_b = _hierarchy(store)
    store.update(sub_a["goal_id"], {"status": "achieved"})
    recompute_rollup(store, milestone["goal_id"])

    milestone_row = store.get(milestone["goal_id"])
    assert milestone_row["progress"] == pytest.approx(0.5)
    assert milestone_row["status"] == "active"


def test_rollup_all_achieved_bubbles_up(store):
    vision, goal, milestone, sub_a, sub_b = _hierarchy(store)
    store.update(sub_a["goal_id"], {"status": "achieved"})
    store.update(sub_b["goal_id"], {"status": "achieved"})
    recompute_rollup(store, milestone["goal_id"])

    milestone_row = store.get(milestone["goal_id"])
    goal_row = store.get(goal["goal_id"])
    vision_row = store.get(vision["goal_id"])
    assert milestone_row["status"] == "achieved"
    assert milestone_row["progress"] == pytest.approx(1.0)
    assert goal_row["status"] == "achieved"
    assert vision_row["status"] == "achieved"


def test_blocked_child_blocks_parent(store):
    vision, goal, milestone, sub_a, sub_b = _hierarchy(store)
    store.update(sub_a["goal_id"], {"status": "blocked"})
    recompute_rollup(store, milestone["goal_id"])

    milestone_row = store.get(milestone["goal_id"])
    assert milestone_row["status"] == "blocked"


def test_abandon_excludes_from_rollup_and_triggers_parent_recompute(store):
    vision, goal, milestone, sub_a, sub_b = _hierarchy(store)
    store.update(sub_a["goal_id"], {"status": "achieved"})
    abandon_project_goal(store=store, goal_id=sub_b["goal_id"])
    recompute_rollup(store, milestone["goal_id"])

    milestone_row = store.get(milestone["goal_id"])
    assert milestone_row["status"] == "achieved"
    assert milestone_row["progress"] == pytest.approx(1.0)


def test_persistence_survives_store_restart(tmp_path):
    db_path = tmp_path / "state.db"
    store_a = DevProjectGoalStore(db_path)
    created = create_project_goal(store=store_a, kind="vision", title="Persisted")
    store_a.close()

    store_b = DevProjectGoalStore(db_path)
    loaded = store_b.get(created["goal_id"])
    store_b.close()
    assert loaded["title"] == "Persisted"


def test_progress_cannot_be_authored_via_update(store):
    vision = create_project_goal(store=store, kind="vision", title="Vision")
    updated = store.update(vision["goal_id"], {"progress": 0.9, "title": "Renamed"})
    assert updated["title"] == "Renamed"
    assert updated["progress"] == 0.0
