from hermes_cli.session_listing import (
    parse_session_listing_args,
    parse_session_listing_request,
    query_session_listing,
)


class FakeSessionDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        offset = kwargs.get("offset", 0)
        return self.rows[offset: offset + kwargs["limit"]]


def test_parse_session_search_request():
    parsed = parse_session_listing_request("all full search Billing Audit")

    assert parsed.include_all_sources is True
    assert parsed.include_unnamed is True
    assert parsed.search_query == "Billing Audit"
    assert parsed.target == ""


def test_parse_session_search_request_requires_query():
    parsed = parse_session_listing_request("search")

    assert parsed.search_requested is True
    assert parsed.search_query == ""
    assert parsed.target == ""


def test_legacy_session_arg_parser_preserves_target_behavior():
    assert parse_session_listing_args("Existing Title") == (
        False,
        False,
        "Existing Title",
    )


def test_query_session_listing_filters_title_and_id_case_insensitively():
    db = FakeSessionDB(
        [
            {"id": "abc-001", "title": "Billing Audit"},
            {"id": "release-ABC", "title": "Release Notes"},
            {"id": "other", "title": "Roadmap"},
        ]
    )

    rows = query_session_listing(
        db,
        source="telegram",
        search_query="abc",
        limit=10,
    )

    assert [row["id"] for row in rows] == ["abc-001", "release-ABC"]
    assert db.calls[0]["search_query"] == "abc"


def test_query_session_listing_includes_unnamed_id_matches_for_search():
    db = FakeSessionDB(
        [
            {"id": "untitled-match-123", "title": None},
            {"id": "named-miss", "title": "Named Session"},
        ]
    )

    rows = query_session_listing(
        db,
        source="telegram",
        search_query="match-123",
        include_unnamed=False,
        limit=10,
    )

    assert [row["id"] for row in rows] == ["untitled-match-123"]
