from unittest.mock import patch

from cron.html_artifacts import (
    CSP,
    REPORT_END,
    REPORT_START,
    HtmlReportMetadata,
    extract_html_report_body,
    render_html_report,
    render_report_from_output,
    strip_html_report_section,
)


def test_extracts_exact_report_section():
    text = f"short summary\n{REPORT_START}\n# Full report\nBody\n{REPORT_END}\nfooter"
    assert extract_html_report_body(text) == "# Full report\nBody"


def test_extract_returns_none_for_missing_or_ambiguous_markers():
    assert extract_html_report_body("no report") is None
    assert extract_html_report_body(f"{REPORT_END}\noops\n{REPORT_START}") is None
    assert extract_html_report_body(f"{REPORT_START}\n{REPORT_END}") is None
    assert extract_html_report_body(f"{REPORT_START}a{REPORT_END}{REPORT_START}b{REPORT_END}") is None


def test_renderer_escapes_raw_html_and_blocks_scripts():
    html = render_html_report(
        "# Safe\n<script>alert(1)</script>\n<img src=https://evil.test/x.png onerror=alert(1)>",
        HtmlReportMetadata(job_id="job1", job_name="Pilot"),
    )

    assert "<script" not in html.lower()
    assert "onerror=" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert CSP in html


def test_renderer_supports_headings_bullets_code_and_safe_links():
    html = render_html_report(
        "# Title\n- item with https://example.com/a?b=1\n\n```\n<raw>\n```",
        {"job_id": "job1", "job_name": "Pilot", "run_time": "2026-05-19T00:00:00Z"},
    )

    assert '<h1 class="report-title">Title</h1>' in html
    assert "<ul>" in html and "<li>item with" in html
    assert '<a href="https://example.com/a?b=1"' in html
    assert "&lt;raw&gt;" in html


def test_renderer_supports_safe_markdown_emphasis_links_and_numbered_lists():
    html = render_html_report(
        "# Title\n1. **bold** and *italic* and `code` plus [Example](https://example.com/x)",
        {"job_id": "job1", "job_name": "Pilot"},
    )

    assert "<ol>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert '<a href="https://example.com/x"' in html


def test_render_from_output_has_no_full_log_fallback():
    assert render_report_from_output("# whole cron archive only", {"job_id": "job1"}) is None


def test_strip_html_report_section_removes_only_valid_report_block():
    text = f"Short Telegram ping\n\n{REPORT_START}\n# Full report\nBody\n{REPORT_END}\n\nTail"
    assert strip_html_report_section(text) == "Short Telegram ping\n\nTail"
    ambiguous = f"Keep\n{REPORT_START}a{REPORT_END}{REPORT_START}b{REPORT_END}"
    assert strip_html_report_section(ambiguous) == ambiguous


def test_rendered_metadata_does_not_require_absolute_source_path():
    html = render_html_report(
        "# Body",
        HtmlReportMetadata(
            job_id="job1",
            job_name="Pilot",
            run_time="2026-05-19T00:00:00Z",
            source_filename="/Users/mozzie/.hermes/cron/output/job1/2026-05-19_12-00-00.md",
        ),
    )

    assert "2026-05-19_12-00-00.md" in html
    assert "/Users/" not in html


def test_rendered_body_redacts_absolute_local_paths_without_breaking_urls():
    html = render_html_report(
        "# Body\nSee /Users/mozzie/.hermes/secrets.txt and /tmp/work/file.md and /etc/passwd and /opt/app/config "
        "and C:\\Users\\Bob\\secret.txt and D:\\tmp\\file.md and C:/Users/Bob/secret.txt "
        "and D:/tmp/file.md and https://example.com/a?b=1\n\n"
        "```\n/home/alice/private.txt\nC:\\Temp\\private.txt\nC:/Temp/private.txt\n```",
        HtmlReportMetadata(job_id="job1", job_name="Pilot"),
    )

    assert "/Users/" not in html
    assert "/tmp/work" not in html
    assert "/home/alice" not in html
    assert "C:\\Users" not in html
    assert "D:\\tmp" not in html
    assert "C:\\Temp" not in html
    assert "C:/Users" not in html
    assert "D:/tmp" not in html
    assert "C:/Temp" not in html
    assert '<a href="https://example.com/a?b=1"' in html
    assert html.count("[LOCAL_PATH]") >= 9


def test_renderer_redacts_secrets_in_body_and_metadata():
    html = render_html_report(
        "# Body\nOPENAI_API_KEY=sk-1234567890abcdef\nAuthorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz123456",
        HtmlReportMetadata(
            job_id="job1",
            job_name="Pilot sk-1234567890abcdef",
            run_time="2026-05-19T00:00:00Z",
        ),
    )

    assert "sk-1234567890abcdef" not in html
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in html


def test_renderer_secret_redaction_fails_closed():
    with patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("redactor unavailable")):
        html = render_html_report(
            "# Body\nOPENAI_API_KEY=sk-1234567890abcdef",
            HtmlReportMetadata(job_id="job1", job_name="Pilot"),
        )

    assert "sk-1234567890abcdef" not in html
    assert "[REDACTED]" in html


def test_renderer_strips_event_handlers_at_start_and_in_code_blocks():
    html = render_html_report(
        "onclick=alert(1) visible\n\n```\nonerror=steal() <tag>\n```",
        HtmlReportMetadata(job_id="job1", job_name="Pilot"),
    )

    assert "onclick=" not in html.lower()
    assert "onerror=" not in html.lower()
    assert "visible" in html
    assert "&lt;tag&gt;" in html
