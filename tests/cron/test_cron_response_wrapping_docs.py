from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CRON_DOCS = (
    "website/docs/user-guide/features/cron.md",
    "website/docs/developer-guide/cron-internals.md",
    "website/docs/guides/cron-troubleshooting.md",
    "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/cron.md",
    "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/developer-guide/cron-internals.md",
    "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/cron-troubleshooting.md",
)
INTERNALS_DOCS = (
    "website/docs/developer-guide/cron-internals.md",
    "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/developer-guide/cron-internals.md",
)


@pytest.mark.parametrize("relative_path", CRON_DOCS)
def test_response_wrapping_docs_cover_footer_opt_in_and_raw_output(relative_path):
    text = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "cron.wrap_response: true" in text
    assert "cron.include_management_footer: false" in text
    assert "cron.include_management_footer: true" in text
    assert "cron.wrap_response: false" in text
    assert "(job_id:" in text


@pytest.mark.parametrize("relative_path", INTERNALS_DOCS)
def test_response_wrapping_internals_name_the_actual_management_footer(relative_path):
    text = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "To stop or manage this job" in text
