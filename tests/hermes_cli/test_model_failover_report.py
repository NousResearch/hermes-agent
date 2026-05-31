from hermes_cli.model_failover_report import (
    REQUIRED_TRACEABILITY_SECTIONS,
    build_model_failover_dashboard_markdown,
    missing_model_failover_traceability_sections,
    render_model_failover_traceability_sections,
)


def test_model_failover_report_renders_all_required_sections_and_escapes_values():
    html = render_model_failover_traceability_sections({
        "model_routing_and_failures": ["MiniMax -> GPT-5.5", "bad <script>"],
        "attribution_reconciliation": {"runtime": 3, "manual": 1},
    })

    assert missing_model_failover_traceability_sections(html) == []
    for section in REQUIRED_TRACEABILITY_SECTIONS:
        assert section in html
    assert "bad &lt;script&gt;" in html
    assert "bad <script>" not in html
    assert "None recorded." in html


def test_model_failover_dashboard_markdown_contains_required_headings_and_defaults():
    markdown = build_model_failover_dashboard_markdown({
        "Opus Usage Audit": "No Opus 4.8 fallback routes reachable."
    })

    for section in REQUIRED_TRACEABILITY_SECTIONS:
        assert f"## {section}" in markdown
    assert "No Opus 4.8 fallback routes reachable." in markdown
    assert "None recorded." in markdown


def test_missing_model_failover_traceability_sections_reports_gaps():
    assert missing_model_failover_traceability_sections("Model Routing and Failures") == [
        "Attribution Reconciliation",
        "Helper/Code Reliability",
        "Opus Usage Audit",
        "Offline Receipts and Residual Risks",
    ]
