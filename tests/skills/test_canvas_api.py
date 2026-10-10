"""canvas_api pagination must not leak the bearer token to a foreign origin.

`_paginated_get` follows the Canvas `Link: <...>; rel="next"` header; it must
only follow a `next` URL that stays on the configured Canvas origin, or a
compromised/misconfigured base could hand the personal access token to an
attacker host.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "canvas" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import canvas_api  # noqa: E402


def _resp(items, link=""):
    headers = {"Link": link} if link else {}
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: items, headers=headers)


def test_paginated_get_does_not_follow_a_cross_origin_next(monkeypatch):
    monkeypatch.setattr(canvas_api, "CANVAS_BASE_URL", "https://canvas.example.edu")
    monkeypatch.setattr(canvas_api, "CANVAS_API_TOKEN", "tok")
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return _resp([{"id": 1}], '<https://evil.example.com/api/v1/courses?page=2>; rel="next"')
        return _resp([{"id": 2}])

    monkeypatch.setattr(canvas_api.requests, "get", fake_get)

    result = canvas_api._paginated_get("https://canvas.example.edu/api/v1/courses")

    assert calls == ["https://canvas.example.edu/api/v1/courses"]
    assert result == [{"id": 1}]


def test_paginated_get_follows_a_same_origin_next(monkeypatch):
    monkeypatch.setattr(canvas_api, "CANVAS_BASE_URL", "https://canvas.example.edu")
    monkeypatch.setattr(canvas_api, "CANVAS_API_TOKEN", "tok")
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return _resp([{"id": 1}], '<https://canvas.example.edu/api/v1/courses?page=2>; rel="next"')
        return _resp([{"id": 2}])

    monkeypatch.setattr(canvas_api.requests, "get", fake_get)

    result = canvas_api._paginated_get("https://canvas.example.edu/api/v1/courses")

    assert calls == [
        "https://canvas.example.edu/api/v1/courses",
        "https://canvas.example.edu/api/v1/courses?page=2",
    ]
    assert result == [{"id": 1}, {"id": 2}]
