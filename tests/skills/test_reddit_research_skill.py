"""Tests for optional-skills/research/reddit-research/scripts/reddit_search.py

These tests exercise the helper script offline — every external HTTP call
is intercepted via ``unittest.mock`` so a network failure or Reddit outage
cannot flake the suite. They cover the contract failures teknium1 flagged
on PR #45690 (#27786):

- ``fetch_post_comments`` must exist and be importable.
- The ``comments`` command's positional argument is ``post_id`` so the
  documented ``comments abc123`` form reaches the handler.
- ``fetch_post_comments`` returns the comment list and propagates API
  errors as ``{"error": ...}`` payloads, not exceptions.
"""

import importlib
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "reddit-research"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

# Import once per process; reset module state between tests.
reddit_search = importlib.import_module("reddit_search")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(argv: list[str], expect_exit: bool = False) -> tuple[int, dict, str]:
    """Run the CLI with mocked argv and capture stdout/stderr.

    Returns (exit_code, parsed_stdout_json, stderr_string). When the
    command is expected to raise SystemExit (e.g. argparse rejects bad
    args), pass ``expect_exit=True`` so SystemExit is caught and
    reported instead of propagating.
    """
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with mock.patch("sys.argv", ["reddit_search"] + argv):
        with mock.patch("sys.stdout", out_buf), mock.patch("sys.stderr", err_buf):
            try:
                reddit_search.main()
                exit_code = 0
            except SystemExit as exc:
                if not expect_exit:
                    raise
                exit_code = exc.code if isinstance(exc.code, int) else 1
    raw = out_buf.getvalue()
    try:
        return exit_code, json.loads(raw) if raw.strip() else {}, err_buf.getvalue()
    except json.JSONDecodeError:
        return exit_code, {}, err_buf.getvalue()


def _make_reddit_listing(comment_rows: list[dict]) -> list:
    """Build a fake Reddit /comments/<id>.json response.

    The real endpoint returns a JSON array of two listings: the first is
    the post itself, the second is the comment listing. ``comment_rows``
    are passed through as-is so each test can control its own ``depth``.
    """
    post_listing = [{"kind": "t3", "data": {"id": "abc123", "title": "stub"}}]
    comment_listing = [
        {"kind": "t1", "data": row} for row in comment_rows
    ]
    listing = {"data": {"children": comment_listing}}
    return [post_listing, listing]


# ---------------------------------------------------------------------------
# 1. fetch_post_comments must exist and work
# ---------------------------------------------------------------------------


class TestFetchPostComments:
    def test_function_is_defined(self):
        """The function must exist on the module — this was the original bug."""
        assert hasattr(reddit_search, "fetch_post_comments"), (
            "fetch_post_comments is missing; the comments command is broken"
        )
        assert callable(reddit_search.fetch_post_comments)

    def test_returns_top_level_comments(self):
        rows = [
            {
                "id": "c1",
                "body": "Top-level reply",
                "author": "u1",
                "score": 12,
                "created_utc": 1.0,
                "permalink": "/r/test/comments/abc123/c1",
                "parent_id": "t3_abc123",
                "link_id": "t3_abc123",
            },
        ]
        payload = _make_reddit_listing(rows)
        with mock.patch.object(
            reddit_search, "api_request", return_value=payload
        ) as api_mock:
            results = reddit_search.fetch_post_comments("abc123", limit=10)
        assert api_mock.call_count == 1
        assert isinstance(results, list)
        assert len(results) == 1
        c = results[0]
        assert c["id"] == "c1"
        assert c["body"] == "Top-level reply"
        assert c["permalink"] == "https://reddit.com/r/test/comments/abc123/c1"

    def test_empty_post_id(self):
        """Empty post_id is defensible: return an error payload, don't crash."""
        results = reddit_search.fetch_post_comments("   ", limit=10)
        assert results == [{"error": "post_id is required"}]

    def test_api_error_propagates_as_payload(self):
        """API errors (HTTPError / URLError) become {"error": ...}; no raise."""
        with mock.patch.object(
            reddit_search,
            "api_request",
            return_value={"error": "Reddit API returned HTTP 503: unavailable"},
        ):
            results = reddit_search.fetch_post_comments("abc123", limit=10)
        assert results == [{"error": "Reddit API returned HTTP 503: unavailable"}]

    def test_malformed_payload_yields_empty_list(self):
        """A non-list or short payload returns [] rather than raising."""
        with mock.patch.object(reddit_search, "api_request", return_value={"oops": "wrong shape"}):
            results = reddit_search.fetch_post_comments("abc123", limit=10)
        assert results == []

    def test_skips_nested_replies(self):
        """Only depth==0 comments are surfaced; nested ones are ignored."""
        rows = [
            {"id": "top", "body": "Top", "permalink": "/x", "depth": 0},
            {"id": "reply", "body": "Reply", "permalink": "/y", "depth": 1},
        ]
        payload = _make_reddit_listing(rows)
        # The helper takes depth from the listing; the second child has depth: 1 here.
        with mock.patch.object(reddit_search, "api_request", return_value=payload):
            results = reddit_search.fetch_post_comments("abc123", limit=10)
        ids = [c["id"] for c in results]
        assert ids == ["top"]


# ---------------------------------------------------------------------------
# 2. cmd_comments must work via the documented `comments <post_id>` form
# ---------------------------------------------------------------------------


class TestCommentsCommand:
    def test_post_id_reaches_handler(self, capsys):
        """`comments abc123` should call fetch_post_comments with 'abc123'."""
        captured_kwargs = {}

        def fake_fetch(post_id, limit):
            captured_kwargs["post_id"] = post_id
            captured_kwargs["limit"] = limit
            return [{"id": "c1", "body": "hi"}]

        with mock.patch.object(reddit_search, "fetch_post_comments", side_effect=fake_fetch):
            exit_code, payload, _stderr = _run(["comments", "abc123", "--limit", "5"])
        assert exit_code == 0
        assert payload["post_id"] == "abc123"
        assert payload["results"] == [{"id": "c1", "body": "hi"}]
        assert captured_kwargs == {"post_id": "abc123", "limit": 5}

    def test_default_limit_when_omitted(self):
        """Without --limit, the command default of 10 reaches the helper."""
        captured = {}

        def fake_fetch(post_id, limit):
            captured["limit"] = limit
            return []

        with mock.patch.object(reddit_search, "fetch_post_comments", side_effect=fake_fetch):
            exit_code, payload, _stderr = _run(["comments", "zzz999"])
        assert exit_code == 0
        assert payload["post_id"] == "zzz999"
        assert captured["limit"] == 10

    def test_post_id_parser_rejects_query_flag(self):
        """The positional is `post_id`, not `query`; passing --subreddit must fail."""
        # The previous broken parser accepted `--subreddit NAME`. After the
        # fix, `comments` only exposes the positional `post_id` plus `--limit`.
        exit_code, _payload, _stderr = _run(["comments", "abc123", "--subreddit", "wallstreetbets"], expect_exit=True)
        assert exit_code != 0, "comments must reject --subreddit (it doesn't accept it)"

    def test_missing_post_id_errors_cleanly(self):
        """Forgetting the post_id should exit non-zero rather than crash with AttributeError."""
        exit_code, payload, stderr = _run(["comments"], expect_exit=True)
        assert exit_code != 0
        assert "post_id" in stderr.lower() or "argument" in stderr.lower()
        # No JSON output should be emitted on an argparse error.
        assert payload == {}


# ---------------------------------------------------------------------------
# 3. Smoke checks for the other commands (regression: ensure parser refactors
#    didn't break the rest of the CLI).
# ---------------------------------------------------------------------------


class TestOtherCommandsSmoke:
    def test_search_runs(self):
        sample = _make_reddit_listing([])
        # search_submissions consumes a params dict and calls api_request once.
        fake_response = {"data": {"children": [
            {
                "kind": "t3",
                "data": {
                    "id": "p1", "title": "hi", "selftext": "body",
                    "subreddit": "x", "author": "u", "score": 1,
                    "num_comments": 0, "created_utc": 1.0, "url": "u",
                    "permalink": "/r/x/comments/abc/p1", "domain": "self.x",
                },
            }
        ]}}
        with mock.patch.object(reddit_search, "api_request", return_value=fake_response):
            exit_code, payload, _stderr = _run(["search", "Nvidia", "--limit", "1"])
        assert exit_code == 0
        assert payload["query"] == "Nvidia"
        assert payload["total_found"] == 1
        assert payload["results"][0]["title"] == "hi"

    def test_unknown_command_exits_nonzero(self):
        exit_code, _payload, _stderr = _run(["bogus-command"], expect_exit=True)
        assert exit_code != 0
