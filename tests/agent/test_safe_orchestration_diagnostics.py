from agent.safe_orchestration_diagnostics import summarize_safe_orchestration_log


def test_summarizes_safe_orchestration_lines_without_echoing_sensitive_content():
    log_text = "\n".join(
        [
            "INFO preflight risk summary: level=low score=0 signals=none recommendations=observe",
            "WARNING preflight risk summary: level=high score=3 signals=sensitive_config_or_secret recommendations=explicit_approval,claude_review token=abc123 .env",
            "WARNING safe orchestration verifier summary: tools=2 findings=1 statuses=ok:2 tools_seen=terminal,write_file codes=sensitive_config_write:1 command='pkill -f hermes' path=.env.local SECRET",
            "WARNING safe orchestration verifier finding: code=sensitive_config_write severity=warning tool=write_file message=File mutation targets a sensitive config/env path; report-only finding.",
            "INFO unrelated line with .env and SECRET should be ignored",
        ]
    )

    summary = summarize_safe_orchestration_log(log_text)

    assert summary.total_lines == 4
    assert summary.preflight_levels == {"high": 1, "low": 1}
    assert summary.preflight_signals == {"none": 1, "sensitive_config_or_secret": 1}
    assert summary.verifier_codes == {"sensitive_config_write": 2}
    assert summary.verifier_tools == {"terminal": 1, "write_file": 2}
    rendered = summary.render()
    assert "safe orchestration diagnostics:" in rendered
    assert "preflight_levels=high:1,low:1" in rendered
    assert "verifier_codes=sensitive_config_write:2" in rendered
    assert ".env" not in rendered
    assert "SECRET" not in rendered
    assert "abc123" not in rendered
    assert "pkill" not in rendered


def test_empty_log_summary_is_deterministic():
    summary = summarize_safe_orchestration_log("")

    assert summary.total_lines == 0
    assert summary.preflight_levels == {}
    assert summary.verifier_codes == {}
    assert summary.render() == (
        "safe orchestration diagnostics: total_lines=0 "
        "preflight_levels=none preflight_signals=none "
        "verifier_codes=none verifier_tools=none"
    )
