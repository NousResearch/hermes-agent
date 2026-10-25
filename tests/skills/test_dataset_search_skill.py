"""Tests for optional-skills/research/dataset-search/scripts/dataset_search.py.

All network access is mocked at the single ``api_request`` boundary, so the
suite exercises real URL construction, namespaced-ID encoding, and client-side
filtering without any live HTTP calls.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "dataset-search"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_search  # noqa: E402

HF_API = dataset_search.HF_API


def _fake_dataset(ds_id, tags=None, likes=0, downloads=0):
    """Build a raw HuggingFace API dataset dict for mocked responses."""
    return {
        "id": ds_id,
        "likes": likes,
        "downloads": downloads,
        "tags": tags or [],
        "siblings": [{"rfilename": "README.md"}],
        "createdAt": "2024-01-01T00:00:00.000Z",
        "lastModified": "2024-02-01T00:00:00.000Z",
    }


def _run(argv):
    """Run main() with given argv, return (parsed_json, captured_url)."""
    captured = {}

    def _fake_api_request(url):
        captured["url"] = url
        # search/popular: URL is HF_API?<query>; detail: HF_API/<id>.
        if url.split("?")[0].rstrip("/") == HF_API:
            return [_fake_dataset("a/b"), _fake_dataset("c/d")]
        return _fake_dataset(url.rsplit("/", 1)[-1])

    with mock.patch.object(dataset_search, "api_request", side_effect=_fake_api_request):
        with mock.patch("sys.argv", ["dataset_search"] + argv):
            with mock.patch("sys.stdout") as out:
                # Capture printed JSON by redirecting print through StringIO.
                import io

                buf = io.StringIO()
                out.write = buf.write
                dataset_search.main()
                text = buf.getvalue()
    return (json.loads(text) if text.strip() else None), captured.get("url")


class TestSearchUrlConstruction:
    def test_basic_search_params(self):
        result, url = _run(["search", "chest xray"])
        assert result["query"] == "chest xray"
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(url).query)
        assert qs["search"] == ["chest xray"]
        assert qs["sort"] == ["likes"]
        assert qs["direction"] == ["-1"]
        assert qs["limit"] == ["10"]

    def test_search_limit_flag(self):
        result, url = _run(["search", "foo", "--limit", "25"])
        assert result["total"] == 2
        from urllib.parse import parse_qs, urlparse

        assert parse_qs(urlparse(url).query)["limit"] == ["25"]


class TestFilterParams:
    def test_task_modality_lang_params(self):
        _result, url = _run(
            ["search", "image", "--task", "image-classification",
             "--modality", "image", "--lang", "en"]
        )
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(url).query)
        assert qs["task_categories"] == ["image-classification"]
        assert qs["modality"] == ["image"]
        assert qs["language"] == ["en"]


class TestNamespacedIdEncoding:
    def test_slash_preserved_in_detail_url(self):
        _result, url = _run(["detail", "keremberke/chest-xray-classification"])
        assert "keremberke/chest-xray-classification" in url
        assert "%2F" not in url  # slash must NOT be percent-encoded

    def test_other_chars_still_encoded(self):
        _result, url = _run(["detail", "foo/bar baz"])
        assert "foo/bar%20baz" in url  # space encoded, slash preserved


class TestSizeFilter:
    def test_small_bucket(self):
        data = [
            _fake_dataset("s1", ["size_categories:n<1K"]),
            _fake_dataset("s2", ["size_categories:1K<n<10K"]),
            _fake_dataset("m1", ["size_categories:10K<n<100K"]),
            _fake_dataset("l1", ["size_categories:1M<n<10M"]),
        ]
        kept = dataset_search._filter_by_size(data, "small")
        ids = {d["id"] for d in kept}
        assert ids == {"s1", "s2"}

    def test_medium_bucket(self):
        data = [
            _fake_dataset("s1", ["size_categories:n<1K"]),
            _fake_dataset("m1", ["size_categories:10K<n<100K"]),
            _fake_dataset("l1", ["size_categories:1M<n<10M"]),
        ]
        kept = dataset_search._filter_by_size(data, "medium")
        assert {d["id"] for d in kept} == {"m1"}

    def test_large_bucket(self):
        data = [
            _fake_dataset("s1", ["size_categories:n<1K"]),
            _fake_dataset("l1", ["size_categories:1M<n<10M"]),
            _fake_dataset("l2", ["size_categories:n>10B"]),
        ]
        kept = dataset_search._filter_by_size(data, "large")
        assert {d["id"] for d in kept} == {"l1", "l2"}

    def test_untagged_excluded(self):
        """Datasets without a size tag are excluded when a size filter is set."""
        data = [
            _fake_dataset("s1", ["size_categories:n<1K"]),
            _fake_dataset("untagged", []),
        ]
        kept = dataset_search._filter_by_size(data, "small")
        assert {d["id"] for d in kept} == {"s1"}

    def test_size_filter_applied_via_search_command(self):
        # _run's fake api_request returns two datasets regardless of query;
        # tag them so the size filter has something to prune.
        captured = {}

        def _fake_api_request(url):
            captured["url"] = url
            return [
                _fake_dataset("small-ds", ["size_categories:n<1K"]),
                _fake_dataset("big-ds", ["size_categories:1M<n<10M"]),
            ]

        with mock.patch.object(dataset_search, "api_request", side_effect=_fake_api_request):
            with mock.patch("sys.argv", ["dataset_search", "search", "x", "--size", "small"]):
                import io

                buf = io.StringIO()
                with mock.patch("sys.stdout") as out:
                    out.write = buf.write
                    dataset_search.main()
                result = json.loads(buf.getvalue())
        ids = {r["id"] for r in result["results"]}
        assert ids == {"small-ds"}


class TestPopularCommand:
    def test_popular_url_and_flag(self):
        result, url = _run(["popular", "--limit", "5"])
        assert result["popular"] is True
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(url).query)
        assert qs["sort"] == ["likes"]
        assert qs["direction"] == ["-1"]
        assert qs["limit"] == ["5"]


class TestDetailExtraction:
    def test_card_data_fields_extracted(self):
        captured = {}
        raw = {
            "id": "owner/ds",
            "description": "A dataset.",
            "likes": 3,
            "downloads": 100,
            "tags": ["foo"],
            "citation": "cite",
            "cardData": {
                "license": "mit",
                "size_categories": ["size_categories:1K<n<10K"],
                "task_categories": ["text-classification"],
                "modality": ["text"],
                "language": ["en"],
            },
            "configs": [{"config_name": "default"}],
            "siblings": [{"rfilename": "README.md"}],
            "paperUrl": None,
            "createdAt": "2024-01-01T00:00:00.000Z",
            "lastModified": "2024-02-01T00:00:00.000Z",
        }

        def _fake_api_request(url):
            captured["url"] = url
            return raw

        with mock.patch.object(dataset_search, "api_request", side_effect=_fake_api_request):
            with mock.patch("sys.argv", ["dataset_search", "detail", "owner/ds"]):
                import io

                buf = io.StringIO()
                with mock.patch("sys.stdout") as out:
                    out.write = buf.write
                    dataset_search.main()
                result = json.loads(buf.getvalue())
        assert result["license"] == "mit"
        assert result["size"] == ["size_categories:1K<n<10K"]
        assert result["task"] == ["text-classification"]
        assert result["modality"] == ["text"]
        assert result["language"] == ["en"]
        assert result["configs"] == ["default"]
