class _Adapter:
    @staticmethod
    def extract_media(content):
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_media(content)

    @staticmethod
    def extract_images(content):
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_images(content)


def test_split_response_attachments_strips_media_from_text(tmp_path, monkeypatch):
    from gateway.run import _split_response_attachments_for_direct_send

    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
    report = tmp_path / "diameter.csv.gz"
    report.write_text("rows\n", encoding="utf-8")

    response = f"Attached file below\nMEDIA:{report}\nDone"

    text, images, media_files = _split_response_attachments_for_direct_send(
        _Adapter(), response
    )

    assert "MEDIA:" not in text
    assert "Attached file below" in text
    assert "Done" in text
    assert images == []
    assert media_files == [(str(report), False)]
