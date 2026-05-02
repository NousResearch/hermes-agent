import pytest

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter


@pytest.mark.asyncio
async def test_thread_context_includes_small_text_attachment(monkeypatch, tmp_path):
    adapter = SlackAdapter(PlatformConfig(token="xoxb-test", extra={}))

    async def fake_download(url, team_id=""):
        assert url == "https://files.slack.test/report.csv"
        return b"keyword,avg monthly searches\nwhisky investment,500\n"

    cached = tmp_path / "doc_abc_report.csv"
    monkeypatch.setattr(adapter, "_download_slack_file_bytes", fake_download)
    monkeypatch.setattr(
        "gateway.platforms.slack.cache_document_from_bytes",
        lambda data, filename: str(cached),
    )

    parts = await adapter._format_thread_file_context(
        {
            "files": [
                {
                    "name": "report.csv",
                    "mimetype": "text/csv",
                    "url_private_download": "https://files.slack.test/report.csv",
                    "size": 49,
                }
            ]
        },
        channel_id="C123",
        team_id="T123",
    )

    rendered = "\n".join(parts)
    assert "Slack thread attachment cached: report.csv" in rendered
    assert str(cached) in rendered
    assert "[Content of Slack thread attachment report.csv]" in rendered
    assert "whisky investment,500" in rendered


@pytest.mark.asyncio
async def test_thread_context_mentions_unsupported_attachment_type():
    adapter = SlackAdapter(PlatformConfig(token="xoxb-test", extra={}))

    parts = await adapter._format_thread_file_context(
        {
            "files": [
                {
                    "name": "creative.mov",
                    "mimetype": "video/quicktime",
                    "url_private_download": "https://files.slack.test/creative.mov",
                    "size": 1024,
                }
            ]
        },
        channel_id="C123",
        team_id="T123",
    )

    assert parts == [
        "[Slack thread attachment: creative.mov (video/quicktime), unsupported file type for automatic download]"
    ]
