"""Deterministic semantic fixtures for #112734 §F / Phase 1A.

The fixtures are the reproducible microbenchmark below OSWorld the issue asks for:
known state transitions with ground-truth bindings, built from synthetic AX trees
(no Bot Screen, no driver) so they run in CI. These tests pin the fixture
contracts — every transition carries explicit ground truth covering every new
element — without asserting anything about matcher behavior.
"""

from tools.computer_use.semantic_fixtures import (
    W, H, all_fixtures, canvas_fixture, delayed_render_fixture,
    element_list, fixture_ids, form_fixture, modal_fixture, reordering_list_fixture,
)


def test_fixture_registry_has_five_fixtures_with_transitions():
    fixtures = all_fixtures()
    assert len(fixtures) == 5
    assert len({f.name for f in fixtures}) == 5  # unique names
    assert all(f.transitions for f in fixtures)


def test_every_transition_carries_complete_ground_truth():
    for fx in all_fixtures():
        for t in fx.transitions:
            assert t.old and t.new
            old_fids = fixture_ids(t.old)
            new_fids = fixture_ids(t.new)
            assert len(set(old_fids)) == len(old_fids)  # ids unique per state
            assert len(set(new_fids)) == len(new_fids)
            # Ground truth covers every new element: bound to an old id, or None.
            assert set(t.expected_binding) == set(new_fids)
            assert set(t.expect_ambiguity) <= set(new_fids)


def test_element_list_projects_to_driver_elements():
    for fx in all_fixtures():
        for t in fx.transitions:
            els = element_list(t.old)
            assert len(els) == len(t.old)
            assert all(e.index >= 1 for e in els)


def test_delayed_render_stall_is_deterministic():
    # The 5s stall is a latency constant on the fixture, not a wall-clock sleep:
    # the fixture runs at full speed in CI.
    fx = delayed_render_fixture()
    assert fx.stall_ms == 5000
    assert [t.name for t in fx.transitions] == ["stall-begins", "render-completes"]
    assert fx.transitions[1].expected_binding["render:export"] == "render:button"


def test_modal_and_list_fixtures_have_the_expected_shapes():
    assert [t.name for t in modal_fixture().transitions] == ["dialog-opens", "dialog-dismisses"]
    assert [t.name for t in reordering_list_fixture().transitions] == ["reorder"]
    assert len(form_fixture().transitions[0].new) == 4
    assert len(canvas_fixture().transitions[0].new) == 3


def test_viewport_constants_are_sane():
    assert W > 0 and H > 0
