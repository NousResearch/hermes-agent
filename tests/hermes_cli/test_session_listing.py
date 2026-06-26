from hermes_cli.session_listing import query_session_listing


class _StubSessionDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


def test_query_session_listing_requests_last_active_order():
    db = _StubSessionDB([
        {
            "id": "old_active",
            "source": "cli",
            "title": "Old but active",
        }
    ])

    rows = query_session_listing(
        db,
        source="cli",
        current_session_id="current",
        include_all_sources=False,
        include_unnamed=False,
        limit=10,
        exclude_sources=["tool"],
    )

    assert [row["id"] for row in rows] == ["old_active"]
    assert db.calls == [
        {
            "source": "cli",
            "exclude_sources": ["tool"],
            "limit": 40,
            "order_by_last_active": True,
        }
    ]


def test_query_session_listing_keeps_all_source_mode_ordered_by_last_active():
    db = _StubSessionDB([
        {
            "id": "telegram_active",
            "source": "telegram",
            "title": "Telegram",
        }
    ])

    rows = query_session_listing(
        db,
        source="cli",
        include_all_sources=True,
        include_unnamed=False,
        limit=5,
        exclude_sources=["tool"],
    )

    assert [row["id"] for row in rows] == ["telegram_active"]
    assert db.calls[0]["source"] is None
    assert db.calls[0]["limit"] == 20
    assert db.calls[0]["order_by_last_active"] is True
