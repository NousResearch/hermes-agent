"""Tests for advisory preflight risk classification."""

from agent.preflight import (
    PreflightRecommendation,
    PreflightRiskLevel,
    classify_request,
    format_preflight_summary,
)


def signal_names(report):
    return {signal.name for signal in report.signals}


def test_read_only_request_is_low_risk():
    report = classify_request("show me run_agent.py and explain how it works")

    assert report.level == PreflightRiskLevel.LOW
    assert report.score == 0
    assert report.recommendations == (PreflightRecommendation.OBSERVE,)


def test_code_edit_request_is_medium_risk_and_recommends_tdd_review():
    report = classify_request("refactor run_agent.py to add a new validation helper")

    assert report.level == PreflightRiskLevel.MEDIUM
    assert "hot_path_edit" in signal_names(report)
    assert report.recommendations == (
        PreflightRecommendation.TDD,
        PreflightRecommendation.CLAUDE_REVIEW,
    )


def test_destructive_request_is_high_risk_and_requires_approval():
    report = classify_request("rm -rf node_modules and git reset --hard origin/main")

    assert report.level == PreflightRiskLevel.HIGH
    assert "destructive_command" in signal_names(report)
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations
    assert PreflightRecommendation.CLAUDE_REVIEW in report.recommendations


def test_secret_or_config_request_is_high_risk():
    report = classify_request("overwrite .env.local with a new api_key value")

    assert report.level == PreflightRiskLevel.HIGH
    assert "sensitive_config_or_secret" in signal_names(report)
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations


def test_runtime_control_request_is_high_risk():
    report = classify_request("restart the Slack gateway and kill hermes if it hangs")

    assert report.level == PreflightRiskLevel.HIGH
    assert "runtime_process_control" in signal_names(report)
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations


def test_schema_change_recommends_tdd_even_when_high_risk():
    report = classify_request("drop table users then migrate the schema")

    assert report.level == PreflightRiskLevel.HIGH
    assert "schema_change" in signal_names(report)
    assert PreflightRecommendation.TDD in report.recommendations
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations


def test_high_signal_dominates_read_only_words():
    report = classify_request("please explain and then delete from users")

    assert report.level == PreflightRiskLevel.HIGH
    assert "destructive_command" in signal_names(report)


def test_format_preflight_summary_redacts_request_and_matches():
    report = classify_request("write SECRET_TOKEN=abc123 into .env.local")

    summary = format_preflight_summary(report)

    assert "preflight risk summary:" in summary
    assert "level=high" in summary
    assert "sensitive_config_or_secret" in summary
    assert "SECRET_TOKEN" not in summary
    assert "abc123" not in summary
    assert ".env" not in summary


def test_classifier_is_deterministic_and_handles_empty_input():
    first = classify_request("   ")
    second = classify_request("   ")

    assert first == second
    assert first.level == PreflightRiskLevel.LOW
    assert first.score == 0
    assert first.signals == ()
    assert first.recommendations == (PreflightRecommendation.OBSERVE,)


def test_classifier_is_case_insensitive():
    report = classify_request("RESTART HERMES GATEWAY")

    assert report.level == PreflightRiskLevel.HIGH
    assert "runtime_process_control" in signal_names(report)


def test_production_deploy_is_medium_risk_and_recommends_review():
    report = classify_request("Deploy the new build to production after tests pass")

    assert report.level == PreflightRiskLevel.MEDIUM
    assert "deployment_or_production" in signal_names(report)
    assert PreflightRecommendation.TDD in report.recommendations
    assert PreflightRecommendation.CLAUDE_REVIEW in report.recommendations


def test_dependency_update_is_medium_risk():
    report = classify_request("Upgrade the requests package and update dependencies")

    assert report.level == PreflightRiskLevel.MEDIUM
    assert "dependency_change" in signal_names(report)


def test_credential_rotation_requires_explicit_approval():
    report = classify_request("Rotate the Slack signing secret and API token")

    assert report.level == PreflightRiskLevel.HIGH
    assert "credential_rotation" in signal_names(report)
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations


def test_mass_refactor_scope_combines_with_code_change_intent():
    report = classify_request("Refactor every Python file in the repository")

    assert report.level == PreflightRiskLevel.HIGH
    assert "mass_scope" in signal_names(report)
    assert PreflightRecommendation.EXPLICIT_APPROVAL in report.recommendations
