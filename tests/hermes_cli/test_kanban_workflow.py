"""Tests for ``hermes_cli.kanban_workflow`` (the workflow-template
machinery behind ``hermes kanban workflow …``).

The reference spec lives at
``hermes_cli/kanban_templates/sec-vuln-remediation.md`` — these tests
verify the implementer honours it end-to-end (load → validate → plan →
write → ship-card sticky block → idempotency), without hitting the
filesystem locations of the production kanban DB.

Run with ``pytest tests/hermes_cli/test_kanban_workflow.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workflow as kw


# ------------------------------------------------------------------ fixtures


SCOUT_BODY = """\
[gh] veroscale-services: [SEC][HIGH] Example vuln in the widget service (VULN-TST-001) — #999 unaddressed

FULL CONTEXT: veroscale-services#999 (VULN-TST-001) — example vulnerability
in the widget service, severity HIGH. Test fixture — do not act on.

Evidence: https://github.com/veroscale/veroscale-services/issues/999

CORPUS-FIRST: search signals for 'VULN-TST-001' before scoping.

Done when: (1) widget service validates input, (2) regression test asserts
the failure mode, (3) PR merged to main.

Idempotency: scout:gh:veroscale-services:999
"""


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def template_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``HERMES_KANBAN_TEMPLATES_DIR`` at a temp copy of the spec.

    We copy rather than symlink so the env override is hermetic; tests
    never touch the real ``hermes_cli/kanban_templates/`` directory even
    if the impl reads it eagerly.
    """
    real = Path(__file__).resolve().parents[2] / "hermes_cli" / "kanban_templates"
    target = tmp_path / "templates"
    target.mkdir()
    for src in real.glob("*.md"):
        shutil.copy2(src, target / src.name)
    monkeypatch.setenv("HERMES_KANBAN_TEMPLATES_DIR", str(target))
    return target


# -------------------------------------------------------------- load_template


def test_load_template_returns_canonical_sec_vuln_remediation(template_dir):
    """The bundled spec must parse without error and expose the canonical
    step keys + ship step."""
    tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
    assert tpl.template_id == "sec-vuln-remediation"
    assert tpl.version == "1.0.0"
    keys = [s.key for s in tpl.steps]
    assert keys == ["corpus-recon", "repro-patch", "regression-test", "ship-pr"]
    assert tpl.ship_step_key == "ship-pr"
    # Severities & required fields from the frontmatter.
    assert set(tpl.severities) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    for required in ("vuln_id", "severity", "component", "body",
                     "evidence_links", "source_card_id"):
        assert required in tpl.required_fields


def test_load_template_rejects_filename_frontmatter_mismatch(tmp_path):
    """Filename must equal frontmatter template_id (spec §7.1)."""
    f = tmp_path / "foo.md"
    f.write_text(
        "---\ntemplate_id: bar\nversion: 1.0.0\n"
        "input:\n  role: scout-issue\n  severity_tag: SEC\n"
        "  severities: [HIGH]\n  required_fields: [vuln_id]\n"
        "steps:\n  - key: ship-pr\n    title: 'X'\n    assignee: default\n    assignee_fallback: default\n    gate: approval\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(kw.WorkflowTemplateError) as ei:
        kw.load_template("foo", templates_dir=tmp_path)
    assert "template_id mismatch" in str(ei.value)


def test_load_template_rejects_missing_approval_step(tmp_path):
    """Spec requires exactly one approval-gated ship step."""
    f = tmp_path / "no-ship.md"
    f.write_text(
        "---\ntemplate_id: no-ship\nversion: 1.0.0\n"
        "input:\n  role: scout-issue\n  severity_tag: SEC\n"
        "  severities: [HIGH]\n  required_fields: [vuln_id]\n"
        "steps:\n  - key: recon\n    title: 'R'\n    assignee: default\n    assignee_fallback: default\n    gate: auto\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(kw.WorkflowTemplateError) as ei:
        kw.load_template("no-ship", templates_dir=tmp_path)
    assert "approval" in str(ei.value)


def test_list_templates_skips_malformed(template_dir, tmp_path):
    """``workflow list`` must not crash when a sibling template is broken."""
    # Drop a malformed file alongside the real spec.
    (template_dir / "broken.md").write_text(
        "---\n# no template_id field\n---\n", encoding="utf-8",
    )
    templates = kw.list_templates(templates_dir=template_dir)
    ids = [t.template_id for t in templates]
    assert "sec-vuln-remediation" in ids
    assert "broken" not in ids


# ------------------------------------------------------------- resolve_source


def test_resolve_source_card_by_id(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="[gh] x: VULN-TST-001 sample",
            body=SCOUT_BODY, created_by="scout",
        )
    with kb.connect() as conn:
        card = kw.resolve_source_card(tid, conn)
    assert card.id == tid


def test_resolve_source_card_by_vuln_id_prefers_scout(kanban_home):
    """Among multiple title hits, prefer scout / [gh] cards (spec §7.1)."""
    with kb.connect() as conn:
        # A non-scout copy of the VULN-id title.
        non_scout = kb.create_task(
            conn, title="Note referencing VULN-TST-001",
            body="just a comment",
        )
        # The scout-created card — preferred.
        scout = kb.create_task(
            conn, title="[gh] svc: VULN-TST-001 vuln",
            body=SCOUT_BODY, created_by="scout",
        )
    with kb.connect() as conn:
        card = kw.resolve_source_card("VULN-TST-001", conn)
    assert card.id == scout
    assert card.id != non_scout


def test_resolve_source_card_ambiguous_error(kanban_home):
    """Two scout cards with the same VULN-id → ambiguous error listing ids."""
    with kb.connect() as conn:
        for i in range(2):
            kb.create_task(
                conn,
                title=f"[gh] svc-{i}: VULN-TST-001 issue",
                body=SCOUT_BODY, created_by="scout",
            )
    with kb.connect() as conn:
        with pytest.raises(kw.WorkflowValidationError) as ei:
            kw.resolve_source_card("VULN-TST-001", conn)
    msg = str(ei.value)
    assert "ambiguous" in msg
    assert "t_" in msg


# --------------------------------------------------------------- validation


def _make_scout_card(kanban_home, *, title=None, body=None):
    with kb.connect() as conn:
        return kb.create_task(
            conn,
            title=title or "[gh] svc: VULN-TST-001 vuln (VULN-TST-001)",
            body=body or SCOUT_BODY,
            created_by="scout",
            idempotency_key="scout:gh:svc:999",
        )


def test_validate_source_happy_path(kanban_home, template_dir):
    tid = _make_scout_card(kanban_home)
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        resolved: dict[str, str] = {}
        v = kw.validate_source(card, tpl, profiles=["default", "calcifer"],
                               assignees_resolved=resolved)
    assert v.ok, v.errors
    assert v.fields["VULN_ID"] == "VULN-TST-001"
    assert v.fields["SEVERITY"] == "HIGH"
    assert "widget service" in v.fields["COMPONENT"].lower()
    assert v.fields["SOURCE_CARD_ID"] == tid
    # done_when mapping: items (1)+ (2) → repro-patch / regression-test,
    # item (3) ("PR merged to main") → ship-pr.
    assert any("regression" in s.lower() for s in v.done_when_mapped["regression-test"])
    assert any("pr" in s.lower() for s in v.done_when_mapped["ship-pr"])


def test_validate_source_rejects_non_scout(kanban_home, template_dir):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="[SEC][HIGH] VULN-TST-001 vuln",
            body=SCOUT_BODY,
            created_by="not-scout",
        )
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        v = kw.validate_source(card, tpl, profiles=["default"],
                               assignees_resolved={})
    assert not v.ok
    assert any("scout" in e for e in v.errors)


def test_validate_source_rejects_non_sec(kanban_home, template_dir):
    body = SCOUT_BODY.replace("[SEC][HIGH]", "[P1]")
    tid = _make_scout_card(kanban_home, body=body)
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        v = kw.validate_source(card, tpl, profiles=["default"],
                               assignees_resolved={})
    assert not v.ok
    assert any("SEC" in e for e in v.errors)


def test_validate_source_rejects_missing_severity(kanban_home, template_dir):
    body = re.sub(r"\[SEC\]\[HIGH\]", "[SEC]", SCOUT_BODY)
    body = re.sub(r"severity HIGH\.", "severity (none).", body)
    tid = _make_scout_card(kanban_home, body=body)
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        v = kw.validate_source(card, tpl, profiles=["default"],
                               assignees_resolved={})
    assert not v.ok
    assert any("severity" in e.lower() for e in v.errors)


def test_validate_source_rejects_unknown_assignee(kanban_home, template_dir):
    """A profile not present on disk fails assignee resolution."""
    tid = _make_scout_card(kanban_home)
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        v = kw.validate_source(card, tpl, profiles=["only-one-profile"],
                               assignees_resolved={})
    assert not v.ok
    assert any("assignee" in e.lower() for e in v.errors)


# -------------------------------------------------------------- instantiate


def _validate_or_skip(kanban_home, template_dir):
    tid = _make_scout_card(kanban_home)
    with kb.connect() as conn:
        card = kb.get_task(conn, tid)
        tpl = kw.load_template("sec-vuln-remediation", templates_dir=template_dir)
        resolved: dict[str, str] = {}
        v = kw.validate_source(card, tpl,
                               profiles=["default", "calcifer"],
                               assignees_resolved=resolved)
    assert v.ok, v.errors
    return tid, card, tpl, v, resolved


def test_instantiate_chain_creates_four_children(kanban_home, template_dir):
    tid, card, tpl, v, resolved = _validate_or_skip(kanban_home, template_dir)
    plan = kw.build_chain_plan(
        card, tpl, v,
        assignee_overrides={}, assignees_resolved=resolved,
    )
    assert len(plan.children) == 4
    # Ship card has the sticky-block payload set.
    ship = plan.ship_child
    assert ship is not None
    assert ship.ship_block_kind == "needs_input"
    assert "APPROVAL GATE" in ship.ship_block_reason
    # Steps 1-3 are auto.
    for c in plan.children[:3]:
        assert c.gate == "auto"
        assert c.ship_block_kind is None


def test_instantiate_chain_dry_run_creates_no_tasks(kanban_home, template_dir):
    """``dry_run=True`` on the planner must not touch the DB."""
    tid, card, tpl, v, resolved = _validate_or_skip(kanban_home, template_dir)
    plan = kw.build_chain_plan(
        card, tpl, v,
        assignee_overrides={}, assignees_resolved=resolved,
    )
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    # Pure planner — should not write.
    # (We don't even call instantiate; just confirm the plan's body text.)
    assert "{{" not in plan.children[0].body
    assert "{{" not in plan.ship_child.body
    with kb.connect() as conn:
        after = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    assert before == after == 1  # only the source card


def test_instantiate_chain_writes_and_ship_card_is_sticky_blocked(
    kanban_home, template_dir,
):
    tid, card, tpl, v, resolved = _validate_or_skip(kanban_home, template_dir)
    plan = kw.build_chain_plan(
        card, tpl, v,
        assignee_overrides={}, assignees_resolved=resolved,
    )
    # Substitute real prev ids into bodies (mirrors cmd_workflow_run).
    prev_ids = [card.id]
    for planned in plan.children:
        idx = next(
            i for i, s in enumerate(tpl.steps, start=1)
            if s.key == planned.step_key
        )
        planned.body = kw._render_step_body(
            tpl.step(planned.step_key),
            fields={
                **v.fields,
                "TEMPLATE_ID": tpl.template_id,
                "STEP_KEY": planned.step_key,
                "STEP_N": str(idx),
                "PREV_STEP_ID": prev_ids[-1],
                "CHAIN_IDS": (
                    "\n".join(
                        f"- step `{tpl.steps[i].key}`: {prev_ids[i]}"
                        for i in range(len(prev_ids))
                    ) if planned.gate == "approval" else ""
                ),
                "DONE_WHEN_MAPPED": kw._format_done_when_mapped(
                    v.done_when_mapped.get(planned.step_key, []),
                ),
            },
        )
    with kb.connect() as conn:
        prev_ids = [card.id]
        created_ids: list[str] = []
        for planned in plan.children:
            step_idx = next(
                i for i, s in enumerate(tpl.steps, start=1)
                if s.key == planned.step_key
            )
            parent_ids = [card.id]
            if step_idx > 1:
                parent_ids.append(prev_ids[step_idx - 1])
            tid_c = kw.create_child_with_sticky_block(
                conn, planned=planned, real_parent_ids=parent_ids,
            )
            created_ids.append(tid_c)
            while len(prev_ids) < step_idx + 1:
                prev_ids.append(tid_c)
            prev_ids[step_idx] = tid_c
        kb.recompute_ready(conn)
    assert len(created_ids) == 4
    # Ship card (4th) is blocked, kind=needs_input, sticky.
    ship_id = created_ids[-1]
    with kb.connect() as conn:
        ship = kb.get_task(conn, ship_id)
        assert ship.status == "blocked"
        assert ship.block_kind == "needs_input"
        # Sticky: events row for "blocked" exists.
        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (ship_id,),
        ).fetchone()
        assert ev is not None, "ship card must have a blocked event for stickiness"
        # workflow_template_id + current_step_key recorded.
        assert ship.workflow_template_id == "sec-vuln-remediation"
        assert ship.current_step_key == "ship-pr"
    # Recompute-ready must not auto-promote the sticky ship card.
    with kb.connect() as conn:
        # Force-advance the chain so the ship card's parents are done,
        # then recompute — it must remain blocked.
        for cid in created_ids[:-1]:
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=? WHERE id=?",
                (int(__import__('time').time()), cid),
            )
            conn.commit()
        kb.recompute_ready(conn)
        ship = kb.get_task(conn, ship_id)
        assert ship.status == "blocked", "sticky-blocked ship card must not auto-promote"
        # Auto steps (1-3) were force-set to done so the ship card's parents
        # are done — they remain done after recompute. The sticky-block on
        # the ship card is what prevents it from auto-promoting.


def test_instantiate_chain_idempotency_key_dedup(kanban_home, template_dir):
    """Re-running create_task with the same idempotency_key returns the
    existing card; no duplicates."""
    tid, card, tpl, v, resolved = _validate_or_skip(kanban_home, template_dir)
    plan = kw.build_chain_plan(
        card, tpl, v,
        assignee_overrides={}, assignees_resolved=resolved,
    )
    idem = plan.children[0].idempotency_key
    assert idem == "sec-vuln-remediation:VULN-TST-001:corpus-recon"
    # First create — succeeds.
    with kb.connect() as conn:
        first = kb.create_task(
            conn, title=plan.children[0].title, body=plan.children[0].body,
            assignee=plan.children[0].assignee,
            idempotency_key=idem,
        )
        # Second create with the same key — returns the same id.
        second = kb.create_task(
            conn, title="something else", body="x",
            assignee=plan.children[0].assignee,
            idempotency_key=idem,
        )
    assert first == second


# -------------------------------------------------------------- CLI dispatch


def test_cmd_workflow_list_renders_canonical_template(kanban_home, template_dir):
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["list"], dry_run=False, force=False,
        assignee=[], json=False,
    )
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        rc = kw.cmd_workflow(args)
    assert rc == 0
    out = buf_out.getvalue()
    assert "sec-vuln-remediation" in out
    assert "corpus-recon(auto)" in out
    assert "ship-pr(approval)" in out


def test_cmd_workflow_list_json(kanban_home, template_dir):
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["list"], dry_run=False, force=False,
        assignee=[], json=True,
    )
    buf_out, _ = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out):
        rc = kw.cmd_workflow(args)
    assert rc == 0
    payload = json.loads(buf_out.getvalue())
    assert payload[0]["template_id"] == "sec-vuln-remediation"
    steps = {s["key"]: s["gate"] for s in payload[0]["steps"]}
    assert steps["ship-pr"] == "approval"
    assert steps["corpus-recon"] == "auto"


def test_cmd_workflow_dry_run_returns_four_children_and_exit_zero(
    kanban_home, template_dir,
):
    tid = _make_scout_card(kanban_home)
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["sec-vuln-remediation", tid], dry_run=True,
        force=False, assignee=[], json=True,
    )
    buf_out, _ = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out):
        rc = kw.cmd_workflow(args)
    assert rc == 0
    payload = json.loads(buf_out.getvalue())
    assert payload["template_id"] == "sec-vuln-remediation"
    assert payload["source_card"] == tid
    assert payload["dry_run"] is True
    assert len(payload["children"]) == 4
    assert all(c["id"] is None for c in payload["children"])
    # Ship card present in the gate block but no id yet (dry-run).
    assert payload["ship_gate"]["step"] == "ship-pr"
    assert payload["ship_gate"]["card_id"] is None
    # Parents field: step 1 = [source]; steps 2-4 = [source, prev-sentinel].
    # Sentinels MUST be self-descriptive, NOT the internal "__PENDING__"
    # placeholder from build_chain_plan (regression guard for the bug where
    # the dry-run JSON leaked an internal symbol to operators).
    step_keys = [c["step"] for c in payload["children"]]
    assert step_keys == ["corpus-recon", "repro-patch", "regression-test", "ship-pr"]
    for i, child in enumerate(payload["children"], start=1):
        parents = child["parents"]
        assert tid in parents, f"step {i} parents must include source card"
        if i == 1:
            assert parents == [tid], f"step 1 parents must be exactly [{tid}], got {parents}"
        else:
            # Steps 2-4 should NOT contain the internal __PENDING__ marker
            # — that's the bug regression guard.
            assert not any(
                "__PENDING__" in str(p) for p in parents
            ), f"step {i} parents leaked __PENDING__ marker: {parents}"
            # Must contain a self-descriptive sentinel referencing the
            # would-be previous step's id (so operators can read it).
            assert len(parents) == 2
            assert str(parents[1]).startswith("<id-of-step-"), (
                f"step {i} prev-id sentinel should be self-descriptive, "
                f"got {parents[1]!r}"
            )


def test_cmd_workflow_rejects_non_scout_with_exit_2(kanban_home, template_dir):
    """Validation failure path → exit code 2, no DB writes."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="[SEC][HIGH] VULN-TST-001 vuln",
            body=SCOUT_BODY,
            created_by="not-scout",
        )
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["sec-vuln-remediation", tid], dry_run=False,
        force=False, assignee=[], json=False,
    )
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        rc = kw.cmd_workflow(args)
    assert rc == 2
    assert "scout" in buf_err.getvalue().lower()
    # No children created.
    with kb.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE workflow_template_id = ?",
            ("sec-vuln-remediation",),
        ).fetchone()["n"]
    assert n == 0


def test_cmd_workflow_show_prints_contract(kanban_home, template_dir):
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["show", "sec-vuln-remediation"], dry_run=False,
        force=False, assignee=[], json=False,
    )
    buf_out, _ = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out):
        rc = kw.cmd_workflow(args)
    assert rc == 0
    out = buf_out.getvalue()
    assert "sec-vuln-remediation" in out
    assert "ship-pr" in out
    assert "approval" in out


def test_cmd_workflow_missing_template_errors(kanban_home, template_dir):
    """Unknown template id → exit 2 (spec §7.2)."""
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["does-not-exist", "t_x"], dry_run=False,
        force=False, assignee=[], json=False,
    )
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        rc = kw.cmd_workflow(args)
    assert rc == 2
    assert "does-not-exist" in buf_err.getvalue()


def test_cmd_workflow_real_run_creates_four_cards_with_provenance(
    kanban_home, template_dir,
):
    """End-to-end real (non-dry-run) against the synthetic SEC scout card
    creates 4 children, ship is sticky-blocked, every child has the
    auto-spawn provenance block."""
    tid = _make_scout_card(kanban_home)
    args = argparse.Namespace(
        kanban_action="workflow",
        positional=["sec-vuln-remediation", tid], dry_run=False,
        force=False, assignee=[], json=True,
    )
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        rc = kw.cmd_workflow(args)
    assert rc == 0, buf_err.getvalue()
    payload = json.loads(buf_out.getvalue())
    assert payload["template_id"] == "sec-vuln-remediation"
    assert len(payload["children"]) == 4
    children_ids = [c["id"] for c in payload["children"]]
    assert all(children_ids), "every child must have a real id"
    ship_id = payload["ship_gate"]["card_id"]
    assert ship_id is not None
    assert ship_id == children_ids[-1]
    # Every child body carries the provenance marker + VULN-id.
    with kb.connect() as conn:
        for cid in children_ids:
            body = kb.get_task(conn, cid).body
            assert "AUTO-SPAWNED by hermes kanban workflow sec-vuln-remediation" in body
            assert "VULN-TST-001" in body
        # Ship card status / kind.
        ship = kb.get_task(conn, ship_id)
        assert ship.status == "blocked"
        assert ship.block_kind == "needs_input"
    # Parents field on the real run: every step's parents must contain
    # real ids (the source card + the previous step's real id for steps
    # 2-4). This is the production-side regression guard for the
    # __PENDING__ placeholder bug.
    for i, child in enumerate(payload["children"], start=1):
        parents = child["parents"]
        assert tid in parents, f"step {i} parents must include source card"
        # No placeholders of any kind in the production JSON.
        for p in parents:
            assert "__PENDING__" not in str(p)
            assert not str(p).startswith("<id-of-step-")
        if i == 1:
            assert parents == [tid]
        else:
            assert parents == [tid, children_ids[i - 2]], (
                f"step {i} parents must be [source, prev_real_id]; "
                f"got {parents}, expected [{tid}, {children_ids[i - 2]}]"
            )
    # Bodies must carry REAL previous-step ids — the provenance
    # "previous step:" line and the "## Chain" section must not leak the
    # dry-run sentinels (<id-of-step-N-would-be-created>) into the
    # production cards (regression guard for the body path, which the
    # JSON-output placeholder fix missed).
    with kb.connect() as conn:
        for i, cid in enumerate(children_ids, start=1):
            body = kb.get_task(conn, cid).body
            assert "<id-of-step-" not in body, f"step {i} body leaks a sentinel"
            assert "__PENDING__" not in body, f"step {i} body leaks __PENDING__"
            if i > 1:
                assert f"previous step: {children_ids[i - 2]}" in body, (
                    f"step {i} provenance must name the real previous child "
                    f"{children_ids[i - 2]}"
                )
        # Ship card's "## Chain" section lists source + all three real
        # sibling ids, in step order.
        ship_body = kb.get_task(conn, ship_id).body
        assert f"- step `corpus-recon`: {tid}" in ship_body
        assert f"- step `repro-patch`: {children_ids[0]}" in ship_body
        assert f"- step `regression-test`: {children_ids[1]}" in ship_body
        assert f"- step `ship-pr`: {children_ids[2]}" in ship_body
    # Re-running is a no-op.
    args2 = argparse.Namespace(
        kanban_action="workflow",
        positional=["sec-vuln-remediation", tid], dry_run=False,
        force=False, assignee=[], json=True,
    )
    buf_out2 = io.StringIO()
    with redirect_stdout(buf_out2):
        rc2 = kw.cmd_workflow(args2)
    assert rc2 == 0
    payload2 = json.loads(buf_out2.getvalue())
    assert payload2.get("no_op") is True
    assert set(payload2["existing_chain"]) == set(children_ids)


def test_cmd_workflow_force_bypasses_preflight_but_still_blocks_ship(
    kanban_home, template_dir,
):
    """``--force`` re-creates missing steps but never unblocks the ship."""
    tid = _make_scout_card(kanban_home)
    base_args = dict(
        kanban_action="workflow",
        positional=["sec-vuln-remediation", tid],
        dry_run=False, force=True, assignee=[], json=True,
    )
    buf_out = io.StringIO()
    with redirect_stdout(buf_out):
        rc = kw.cmd_workflow(argparse.Namespace(**base_args))
    assert rc == 0
    first = json.loads(buf_out.getvalue())
    ship_id = first["ship_gate"]["card_id"]
    # Force-run again — chain already exists, --force says "still go".
    buf_out2 = io.StringIO()
    with redirect_stdout(buf_out2):
        rc2 = kw.cmd_workflow(argparse.Namespace(**base_args))
    assert rc2 == 0
    # Ship card remains sticky-blocked.
    with kb.connect() as conn:
        s = kb.get_task(conn, ship_id)
        assert s.status == "blocked"
        assert s.block_kind == "needs_input"
