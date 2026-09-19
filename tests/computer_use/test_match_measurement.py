"""Incorrect semantic-match measurement against fixture ground truth (#112734 §J).

``diff_states`` decides bindings; fixtures know the right answers. These tests
prove the audit binds exactly what the ground truth says — no invented
identities, no silent mis-binds — across every deterministic fixture.
"""

from tools.computer_use.match_measurement import (
    audit_all_fixtures, measure_transition,
)
from tools.computer_use.semantic_fixtures import (
    W, H, all_fixtures, canvas_fixture, delayed_render_fixture,
    element_list, fixture_ids, form_fixture, modal_fixture, reordering_list_fixture,
)
from tools.computer_use.semantic_state import build_state
from tools.computer_use.state_diff import diff_states


def _states(fixture, transition):
    target = f"FixtureApp/{fixture.name}"
    old = build_state(element_list(transition.old), revision=1, target=target, width=W, height=H)
    new = build_state(element_list(transition.new), revision=2, target=target, width=W, height=H)
    return old, new


def test_form_fixture_matches_ground_truth_exactly():
    fx = form_fixture()
    q = measure_transition(fx, fx.transitions[0])
    assert q.incorrect == 0 and q.correct == q.total == 4


def test_modal_fixture_adds_and_removes_without_rebinding():
    fx = modal_fixture()
    opened = measure_transition(fx, fx.transitions[0])
    assert opened.incorrect == 0
    closed = measure_transition(fx, fx.transitions[1])
    assert closed.incorrect == 0


def test_reordering_list_keeps_identity_despite_geometry_change():
    fx = reordering_list_fixture()
    q = measure_transition(fx, fx.transitions[0])
    assert q.incorrect == 0


def test_delayed_render_relabel_binds_with_name_change_recorded():
    # Render → Export is the same control relabeled (role + parent + geometry
    # agree): bound with the name change recorded in `changed`, not swapped for a
    # fabricated new element. Late content stays unbound.
    fx = delayed_render_fixture()
    begins = measure_transition(fx, fx.transitions[0])
    assert begins.incorrect == 0  # Working… is the same status label, new text
    done = measure_transition(fx, fx.transitions[1])
    assert done.incorrect == 0
    old, new = _states(fx, fx.transitions[1])
    delta = diff_states(old, new)
    new_fids = dict(zip(new.elements, fixture_ids(fx.transitions[1].new)))
    added = {new_fids[e] for e in delta.added}
    assert {"render:row1", "render:row2"} <= added
    assert "render:export" not in added
    name_changes = [c for c in delta.changed
                    if new_fids[c.element] == "render:export"
                    and any(f == "name" for f, _ in c.changes)]
    assert len(name_changes) == 1 and name_changes[0].confidence == 0.5


def test_canvas_gains_no_fabricated_identity():
    fx = canvas_fixture()
    q = measure_transition(fx, fx.transitions[0])
    assert q.incorrect == 0


def test_audit_all_fixtures_zero_incorrect_matches():
    report = audit_all_fixtures(all_fixtures())
    assert report.transitions == 7  # modal and delayed-render have two steps each
    assert report.incorrect == 0
    assert report.incorrect_rate == 0.0
    assert len(report.per_transition) == 7


def test_aggregate_report_renders_section_j_view():
    report = audit_all_fixtures(all_fixtures())
    text = report.render()
    assert "incorrect match rate: 0.0%" in text
    assert "ambiguity:" in text
