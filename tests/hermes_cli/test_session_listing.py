from hermes_cli.session_listing import query_session_listing


def test_query_session_listing_skips_empty_untitled_placeholders():
    captured = {}

    class _DB:
        def list_sessions_rich(self, **kwargs):
            captured.update(kwargs)
            rows = [
                {
                    "id": "ghost",
                    "source": "cli",
                    "title": "",
                    "preview": "",
                    "message_count": 0,
                },
                {
                    "id": "real",
                    "source": "cli",
                    "title": "",
                    "preview": "hello",
                    "message_count": 1,
                },
            ]
            if kwargs.get("exclude_empty_untitled"):
                rows = [
                    row
                    for row in rows
                    if (row.get("message_count") or 0) > 0
                    or str(row.get("title") or "").strip()
                ]
            return rows

    rows = query_session_listing(
        _DB(),
        source="cli",
        include_unnamed=True,
        limit=10,
    )

    assert captured["exclude_empty_untitled"] is True
    assert [row["id"] for row in rows] == ["real"]
