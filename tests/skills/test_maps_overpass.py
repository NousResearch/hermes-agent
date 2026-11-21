"""Overpass fallback, cache and error output of the maps skill (no network)."""

import argparse
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "maps"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import maps_client as mc

A = "https://a.example/api/interpreter"
B = "https://b.example/api/interpreter"
OK = {"elements": [{"type": "node", "id": 1, "lat": 52.5, "lon": 13.4,
                    "tags": {"amenity": "cafe", "name": "Kaffee"}}]}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def overpass(monkeypatch, tmp_path):
    """Fake mirrors A and B; each test sets per-URL behaviour in ``plan``."""
    monkeypatch.setattr(mc, "OVERPASS_URLS", [A, B])
    monkeypatch.setattr(mc, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(mc, "_OVERPASS_BLOCKED", {})
    calls = []
    plan = {}

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        action = plan[url]
        if isinstance(action, BaseException):
            raise action
        return _Resp(json.dumps(action).encode())

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    return plan, calls


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "x", {}, None)


def test_falls_through_to_next_mirror_once_each(overpass):
    plan, calls = overpass
    plan[A] = _http_error(A, 504)
    plan[B] = OK
    assert mc.overpass_query("q1") == OK
    assert calls == [A, B]


def test_all_mirrors_fail_raises_with_tried_list(overpass):
    plan, calls = overpass
    plan[A] = _http_error(A, 504)
    plan[B] = urllib.error.URLError(TimeoutError("timed out"))
    with pytest.raises(mc.OverpassError) as exc:
        mc.overpass_query("q2")
    assert calls == [A, B]
    assert [t["url"] for t in exc.value.tried] == [A, B]
    assert exc.value.tried[0]["result"].startswith("504")
    assert exc.value.tried[1]["result"].startswith("timeout")


def test_mirror_that_answered_429_is_not_asked_again(overpass):
    plan, calls = overpass
    plan[A] = _http_error(A, 429)
    plan[B] = OK
    mc.overpass_query("q3")
    mc.overpass_query("q4")
    assert calls == [A, B, B]


def test_runtime_error_remark_counts_as_failure(overpass):
    plan, calls = overpass
    plan[A] = {"elements": [], "remark": "runtime error: Query timed out"}
    plan[B] = OK
    assert mc.overpass_query("q5") == OK
    assert calls == [A, B]


def test_answer_is_cached(overpass):
    plan, calls = overpass
    plan[A] = OK
    mc.overpass_query("q6")
    assert mc.overpass_query("q6") == OK
    assert calls == [A]


def test_unwritable_cache_is_ignored(overpass, monkeypatch, tmp_path):
    plan, _calls = overpass
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(mc, "CACHE_DIR", str(blocker / "sub"))
    plan[A] = OK
    assert mc.overpass_query("q7") == OK


def _nearby_args(categories):
    return argparse.Namespace(lat=52.5, lon=13.4, near=None, category=None,
                              category_list=categories, radius=500, limit=10)


def test_nearby_reports_partial_when_one_category_fails(monkeypatch, capsys):
    def fake_query(query):
        if '"pharmacy"' in query:
            raise mc.OverpassError([{"url": A, "result": "504 after 1.0s"}])
        return OK

    monkeypatch.setattr(mc, "overpass_query", fake_query)
    mc.cmd_nearby(_nearby_args(["cafe", "pharmacy"]))
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "partial"
    assert out["count"] == 1
    assert out["failed_categories"][0]["category"] == "pharmacy"


def test_nearby_prints_single_json_error_when_everything_fails(monkeypatch, capsys):
    def fake_query(query):
        raise mc.OverpassError([{"url": A, "result": "504 after 1.0s"}])

    monkeypatch.setattr(mc, "overpass_query", fake_query)
    with pytest.raises(SystemExit):
        mc.cmd_nearby(_nearby_args(["cafe"]))
    out = json.loads(capsys.readouterr().out)  # exactly one JSON document
    assert out["status"] == "error"
    assert out["error"] == "overpass unavailable"
