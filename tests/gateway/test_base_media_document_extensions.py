from gateway.platforms.base import BasePlatformAdapter


def test_extract_media_accepts_markdown_files():
    content = "Report attached\nMEDIA:/tmp/example_report.md"

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == [("/tmp/example_report.md", False)]
    assert "MEDIA:" not in cleaned
    assert cleaned == "Report attached"
