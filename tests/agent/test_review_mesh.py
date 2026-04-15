from agent.review_memory import record_review_adjudication
from agent.review_mesh import (
    ReviewMeshRequest,
    aggregate_review_findings,
    build_specialist_task,
    normalize_specialist_payload,
    plan_review_mesh,
)


def test_build_specialist_task_assigns_profile_and_context_hint():
    request = ReviewMeshRequest(
        goal="Review the auth changes",
        context="Look for token handling mistakes.",
        touched_paths=["src/auth/login.py"],
    )
    plan = plan_review_mesh(request)
    spec = build_specialist_task("security", request, plan)

    assert spec["subagent_profile"] == "security_reviewer"
    assert spec["toolsets"] == ["terminal", "file"]
    assert "Use the security_reviewer subagent profile" in spec["context"]


def test_plan_review_mesh_selects_specialists_from_scope_and_red_team_flag():
    request = ReviewMeshRequest(
        goal="Review the auth and caching changes",
        context="Focus on login token validation, query hot paths, and test coverage.",
        touched_paths=[
            "src/auth/login.py",
            "src/cache/redis_cache.py",
            "tests/test_login.py",
        ],
        enable_red_team=True,
    )

    plan = plan_review_mesh(request)

    assert plan.specialists == [
        "testing",
        "security",
        "performance",
        "maintainability",
        "red_team",
    ]
    assert plan.activation_reasons["security"]
    assert plan.activation_reasons["performance"]
    assert plan.activation_reasons["red_team"] == ["explicitly_requested"]


def test_normalize_specialist_payload_maps_legacy_fields_and_clamps_values():
    payload = {
        "findings": [
            {
                "level": "urgent",
                "headline": "Missing authz check",
                "details": "Admin route trusts a client flag.",
                "path": "src/auth/routes.py",
                "line": 41,
                "confidence": 1.7,
                "recommendation": "Enforce server-side permission validation.",
            }
        ]
    }

    normalized = normalize_specialist_payload(
        specialist="security",
        payload=payload,
        task_index=2,
    )

    finding = normalized["findings"][0]
    assert finding["severity"] == "high"
    assert finding["title"] == "Missing authz check"
    assert finding["summary"] == "Admin route trusts a client flag."
    assert finding["file_path"] == "src/auth/routes.py"
    assert finding["line_start"] == 41
    assert finding["confidence"] == 1.0
    assert finding["specialist"] == "security"
    assert finding["source_task_index"] == 2


def test_aggregate_review_findings_ranks_by_severity_and_confidence():
    specialist_runs = [
        {
            "specialist": "testing",
            "summary": "Testing review complete.",
            "findings": [
                {
                    "severity": "medium",
                    "title": "Missing regression test",
                    "summary": "No coverage for the cache invalidation path.",
                    "confidence": 0.7,
                    "specialist": "testing",
                }
            ],
        },
        {
            "specialist": "security",
            "summary": "Security review complete.",
            "findings": [
                {
                    "severity": "critical",
                    "title": "Privilege escalation",
                    "summary": "Endpoint accepts a forged role header.",
                    "confidence": 0.6,
                    "specialist": "security",
                },
                {
                    "severity": "high",
                    "title": "JWT audience not checked",
                    "summary": "Tokens issued for another service are accepted.",
                    "confidence": 0.95,
                    "specialist": "security",
                },
            ],
        },
    ]

    aggregate = aggregate_review_findings(specialist_runs)

    assert [finding["title"] for finding in aggregate["findings"]] == [
        "Privilege escalation",
        "JWT audience not checked",
        "Missing regression test",
    ]
    assert aggregate["severity_counts"] == {
        "critical": 1,
        "high": 1,
        "medium": 1,
    }
    assert aggregate["highest_severity"] == "critical"
    assert aggregate["specialists_run"] == ["testing", "security"]
    assert aggregate["suppressed_count"] == 0


def test_aggregate_review_findings_reports_suppression_metadata(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    finding = {
        "severity": "medium",
        "title": "Missing regression test",
        "summary": "No coverage for the cache invalidation path.",
        "confidence": 0.7,
        "specialist": "testing",
        "file_path": "src/cache/service.py",
    }
    record_review_adjudication(
        finding,
        adjudication="already_fixed",
        project_root=str(project_root),
        recorded_at="2026-04-15T21:00:00Z",
    )

    aggregate = aggregate_review_findings(
        [{"specialist": "testing", "findings": [finding]}],
        project_root=str(project_root),
    )

    assert aggregate["findings"] == []
    assert aggregate["suppressed_count"] == 1
    assert aggregate["suppressed_findings"][0]["review_memory"]["latest_adjudication"] == "already_fixed"
    assert aggregate["review_memory"]["project_root"] == str(project_root.resolve())
