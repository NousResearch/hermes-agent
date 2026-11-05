"""Tests for the har-derived-api-client optional skill.

Runs the real har_to_client.py logic against a synthetic HAR fixture and
asserts it derives the endpoint, collapses id path segments, filters static
assets, and surfaces the User-Agent replay hint. Stdlib + pytest, no network.
"""

import importlib.util
import json
from pathlib import Path

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "web-development"
    / "har-derived-api-client"
)
DERIVE = SKILL_DIR / "scripts" / "har_to_client.py"

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --- behavioral: derivation logic -----------------------------------------

def _make_har() -> dict:
    return {
        "log": {
            "entries": [
                {  # a JSON API call we want derived, with an id path segment
                    "_resourceType": "fetch",
                    "request": {
                        "method": "GET",
                        "url": "https://api.example.com/v1/items/12345/reviews?limit=5",
                        "queryString": [{"name": "limit", "value": "5"}],
                        "headers": [
                            {"name": "User-Agent", "value": "Mozilla/5.0 TestBrowser/1.0"},
                            {"name": "accept", "value": "application/json"},
                            {"name": "referer", "value": "https://example.com/"},
                        ],
                    },
                    "response": {
                        "status": 200,
                        "content": {
                            "mimeType": "application/json",
                            "text": '{"reviews":[{"id":1}]}',
                        },
                    },
                },
                {  # a static asset we must filter out by default
                    "_resourceType": "script",
                    "request": {
                        "method": "GET",
                        "url": "https://cdn.example.com/app.js",
                        "queryString": [],
                        "headers": [{"name": "User-Agent", "value": "Mozilla/5.0 TestBrowser/1.0"}],
                    },
                    "response": {"status": 200, "content": {"mimeType": "application/javascript"}},
                },
            ]
        }
    }

def test_derives_endpoint_and_filters_static(tmp_path, capsys):
    mod = _load_module(DERIVE, "har_to_client_undertest")
    har = tmp_path / "t.har"
    har.write_text(json.dumps(_make_har()), encoding="utf-8")

    import sys

    argv = sys.argv
    try:
        sys.argv = ["har_to_client.py", str(har), "--host", "example.com"]
        rc = mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out

    assert rc == 0
    # id path segment collapsed to {id}
    assert "GET https://api.example.com/v1/items/{id}/reviews" in out
    # query param surfaced
    assert "limit = 5" in out
    # static JS filtered out
    assert "app.js" not in out
    # boring header dropped, useful one absent from list but UA promoted to hints
    assert "referer" not in out
    # replay hint carries the browser UA
    assert "User-Agent (send this): Mozilla/5.0 TestBrowser/1.0" in out

def test_path_template_collapses_ids():
    mod = _load_module(DERIVE, "har_to_client_undertest2")
    assert mod.path_template("/v1/items/12345/x") == "/v1/items/{id}/x"
    assert mod.path_template("/v1/items/abc/x") == "/v1/items/abc/x"
