"""decompose_gate: free deterministic pre-gate for the kanban decomposer."""

from __future__ import annotations

from hermes_cli import decompose_gate as dg


def test_eligible_triage_card():
    ok, why = dg.should_decompose(title="ship a feature", body="")
    assert ok is True
    assert why == "eligible"


def test_not_triage():
    ok, why = dg.should_decompose(status="ready")
    assert ok is False
    assert "not in triage" in why


def test_already_specified_or_decomposed_or_has_children():
    for kwargs in ({"specified": True}, {"decomposed": True},
                   {"has_children": True}):
        ok, why = dg.should_decompose(**kwargs)
        assert ok is False
        assert why == "already specified/decomposed"


def test_no_decompose_marker():
    ok, why = dg.should_decompose(title="x", body="please no-decompose this")
    assert ok is False
    assert why == "no-decompose marker"


def test_marker_is_case_and_separator_insensitive():
    assert dg.has_no_decompose_marker("No_Decompose this")
    assert dg.has_no_decompose_marker("keep-as-one please")
    assert not dg.has_no_decompose_marker("decompose this fully")


def test_opt_in_only_marker_or_allowlist():
    ok, why = dg.should_decompose(title="do a thing", opt_in_only=True)
    assert ok is False and "not opted in" in why
    assert dg.should_decompose(title="#decompose do a thing", opt_in_only=True)[0]
    assert dg.should_decompose(
        title="do a thing", created_by="manager", opt_in_only=True,
        opt_in_created_by=("manager",))[0]
    # a non-allowlisted creator without the marker is skipped
    assert dg.should_decompose(
        title="do a thing", created_by="worker-base", opt_in_only=True,
        opt_in_created_by=("manager",))[0] is False


def test_require_multi_deliverable():
    ok, why = dg.should_decompose(title="bump the pin",
                                  require_multi_deliverable=True)
    assert ok is False and "single deliverable" in why
    assert dg.should_decompose(title="do work", body="- a\n- b\n- c\n",
                               require_multi_deliverable=True)[0]


def test_recon_detection_and_filter_children():
    kids = [
        {"title": "Implement the fix"},
        {"title": "Read-only recon of the lane"},
        {"title": "Audit the upstream headers"},
        {"title": "Write the release note"},
    ]
    kept, dropped = dg.filter_children(kids, max_recon_children=0)
    assert dropped == 2
    assert [k["title"] for k in kept] == ["Implement the fix",
                                          "Write the release note"]
    _, dropped1 = dg.filter_children(kids, max_recon_children=1)
    assert dropped1 == 1
    assert dg.is_recon("Read-only audit of lane opencode_go")
    assert not dg.is_recon("Implement per-lane backoff")
