"""Tests for the read-only Radar Hermes dashboard snapshot helper."""

from hermes_cli.radar_hermes import build_radar_hermes_snapshot


def test_radar_hermes_snapshot_ranks_top_controversial_and_parkable_read_only():
    data = {
        "updated_at": "2026-06-23T20:46:00Z",
        "proposals": [
            {
                "id": "live-work",
                "title": "Mission Control Live Work read-only",
                "kind": "evolution",
                "origin": "Systems Reliability Steward",
                "priority": "P0",
                "benefit": "high",
                "effort": "medium",
                "risk": "low",
                "confidence": "high",
                "status": "raccomandata",
                "whyNow": "Daniele non vede facilmente lavori API/Telegram in Mission Control.",
                "acceptance": "Vista read-only senza transcript completo.",
                "recommendation": "do_now",
                "chief_review_score": 91,
                "source_key": "autonomous-team:mission-control-live-work",
                "last_signal_at": "2026-06-23T20:40:00Z",
            },
            {
                "id": "spec-builder",
                "title": "Spec builder approval-gated",
                "kind": "evolution",
                "origin": "Strategy Chief",
                "priority": "P1",
                "benefit": "high",
                "effort": "medium",
                "risk": "medium",
                "confidence": "medium",
                "status": "proposta",
                "whyNow": "Le proposte evolutive devono diventare specifiche verificabili.",
                "acceptance": "Preview locale prima di Kanban.",
                "recommendation": "prepare",
                "chief_review_score": 72,
                "source_key": "autonomous-team:spec-builder",
            },
            {
                "id": "risky-auto-controls",
                "title": "Controlli autonomi dashboard",
                "kind": "evolution",
                "origin": "Systems Reliability Steward",
                "priority": "P1",
                "benefit": "high",
                "effort": "high",
                "risk": "high",
                "confidence": "medium",
                "status": "proposta",
                "whyNow": "Potrebbero ridurre attrito ma aumentano rischio operativo.",
                "acceptance": "Richiede gate umano prima di qualsiasi azione.",
                "recommendation": "prepare",
                "challenge": {
                    "supporter": "reliability",
                    "critic": "ops",
                    "challenge": "Rischio di auto-dispatch non voluto.",
                    "chief_synthesis": "Tenere read-only finché i gate non sono testati.",
                    "veto_risk": "high",
                },
                "chief_review_score": 68,
            },
            {
                "id": "docx-claims-kill-switch",
                "title": "Claim-kill-switch per DOCX prima dell’upload",
                "kind": "evolution",
                "origin": "Legal & Claims Guardian",
                "priority": "P2",
                "benefit": "high",
                "effort": "medium",
                "risk": "medium",
                "confidence": "medium",
                "status": "parcheggiata",
                "whyNow": "Idea utile ma non deve competere con la stabilizzazione dashboard.",
                "acceptance": "Scanner locale futuro.",
                "recommendation": "park",
            },
            {
                "id": "operative-not-radar",
                "title": "SOC sampling plan per Kania",
                "kind": "operative",
                "origin": "MRV Architect",
                "priority": "P1",
                "benefit": "high",
                "effort": "medium",
                "risk": "medium",
                "confidence": "medium",
                "status": "proposta",
                "whyNow": "Operativa, non evoluzione Hermes.",
                "acceptance": "N/A",
                "recommendation": "prepare",
            },
        ],
    }

    snapshot = build_radar_hermes_snapshot(data, generated_at="2026-06-23T21:00:00Z")

    assert snapshot["version"] == "radar_hermes.v1"
    assert snapshot["generated_at"] == "2026-06-23T21:00:00Z"
    assert snapshot["source_summary"]["proposals_seen"] == 4
    assert snapshot["source_summary"]["proposals_returned"] == 4
    assert snapshot["source_summary"]["side_effects"] == {
        "kanban_mutated": False,
        "cron_created": False,
        "external_send": False,
        "subagent_spawned": False,
    }
    assert snapshot["approval_policy"]["read_only_first"] is True
    assert snapshot["approval_policy"]["requires_explicit_approval_for_kanban"] is True
    assert [item["title"] for item in snapshot["blocks"]["top"]] == [
        "Mission Control Live Work read-only",
        "Spec builder approval-gated",
        "Controlli autonomi dashboard",
    ]
    assert snapshot["blocks"]["controversial"]["id"] == "radar:team_proposals:risky-auto-controls"
    assert snapshot["blocks"]["parkable"]["id"] == "radar:team_proposals:docx-claims-kill-switch"
    for item in snapshot["blocks"]["top"]:
        assert item["approval"]["kanban_creation_available"] is False
        assert item["governance"]["read_only_surface"] is True
        assert item["governance"]["no_cron_created"] is True
        assert item["governance"]["no_external_send"] is True
        assert item["governance"]["no_subagent_spawned"] is True
        assert "body" not in item
        assert "prompt" not in item
        assert "transcript" not in item


def test_radar_hermes_snapshot_empty_state_when_no_evolution_proposals():
    snapshot = build_radar_hermes_snapshot({"proposals": []}, generated_at="2026-06-23T21:00:00Z")

    assert snapshot["blocks"] == {"top": [], "controversial": None, "parkable": None}
    assert snapshot["empty_state"]["title"] == "Nessuna proposta Radar source-grounded pronta"
    assert snapshot["controversy_state"] == {
        "status": "insufficient_controversy",
        "title": "Nessuna proposta controversa qualificata",
        "message": "Le fonti lette non contengono una proposta con challenge, veto risk materiale, rischio alto o raccomandazione di reject. Nessuna top proposta viene rilabelizzata come controversa.",
    }
    assert snapshot["source_summary"]["proposals_returned"] == 0


def test_radar_hermes_does_not_relabel_low_risk_top_proposal_as_controversial():
    data = {
        "updated_at": "2026-06-23T20:46:00Z",
        "proposals": [
            {
                "id": "team-proposals-page",
                "title": "Team Proposals page polish",
                "kind": "evolution",
                "origin": "Systems Reliability Steward",
                "priority": "P1",
                "benefit": "high",
                "effort": "medium",
                "risk": "low",
                "confidence": "high",
                "status": "raccomandata",
                "whyNow": "Rende più chiaro il backlog evolutivo senza introdurre automazioni.",
                "acceptance": "Solo UI read-only.",
                "recommendation": "prepare",
                "chief_review_score": 88,
                "source_key": "team-proposals-page",
            }
        ],
    }

    snapshot = build_radar_hermes_snapshot(data, generated_at="2026-06-23T21:00:00Z")

    assert [item["id"] for item in snapshot["blocks"]["top"]] == ["radar:team_proposals:team-proposals-page"]
    assert snapshot["blocks"]["top"][0]["flags"]["controversial"] is False
    assert snapshot["blocks"]["controversial"] is None
    assert snapshot["controversy_state"]["status"] == "insufficient_controversy"
    assert snapshot["source_summary"]["proposals_returned"] == 1


def test_radar_hermes_controversial_block_does_not_duplicate_top_without_challenge_rationale():
    data = {
        "updated_at": "2026-06-23T20:46:00Z",
        "proposals": [
            {
                "id": "top-medium-risk-no-challenge",
                "title": "Medium risk but unchallenged candidate",
                "kind": "evolution",
                "origin": "Systems Reliability Steward",
                "priority": "P1",
                "benefit": "high",
                "effort": "medium",
                "risk": "medium",
                "confidence": "high",
                "status": "raccomandata",
                "whyNow": "Ha un rischio medio ma nessun supporter-vs-critic esplicito.",
                "acceptance": "Richiede solo review ordinaria.",
                "recommendation": "prepare",
                "chief_review_score": 82,
            }
        ],
    }

    snapshot = build_radar_hermes_snapshot(data, generated_at="2026-06-23T21:00:00Z")

    assert snapshot["blocks"]["top"][0]["id"] == "radar:team_proposals:top-medium-risk-no-challenge"
    assert snapshot["blocks"]["controversial"] is None
    assert snapshot["controversy_state"]["message"].endswith("Nessuna top proposta viene rilabelizzata come controversa.")
