"""Tests for the har-derived-api-client optional skill.

Two layers, both stdlib + pytest, no network:
  1. Structural / frontmatter contract on SKILL.md (matches the maintainer
     review checklist for optional skills).
  2. Behavioral: run the real har_to_client.py logic against a synthetic HAR
     fixture and assert it derives the endpoint, collapses id path segments,
     filters static assets, and surfaces the User-Agent replay hint.
"""

import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "web-development"
    / "har-derived-api-client"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
CAPTURE = SKILL_DIR / "scripts" / "har_capture.py"
CAPTURE_CDP = SKILL_DIR / "scripts" / "har_capture_cdp.py"
DERIVE = SKILL_DIR / "scripts" / "har_to_client.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- structural contract ---------------------------------------------------


def test_skill_files_exist():
    assert SKILL_MD.is_file()
    assert CAPTURE.is_file()
    assert CAPTURE_CDP.is_file()
    assert DERIVE.is_file()


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n")
    assert skill_text.count("---") >= 2


def test_description_under_sixty_chars(skill_text: str):
    m = re.search(r"^description: (.*)$", skill_text, re.MULTILINE)
    assert m, "no description field"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars (>60): {desc!r}"
    assert desc.endswith("."), "description should end with a period"


def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"


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


def test_preserves_scheme_parses_url_query_and_redacts_credentials(tmp_path, capsys):
    mod = _load_module(DERIVE, "har_to_client_security_undertest")
    fixture = _make_har()
    request = fixture["log"]["entries"][0]["request"]
    request["url"] = "http://api.example.com/v1/items/12345/reviews?q=one&blank="
    request["queryString"] = []
    request["headers"].extend([
        {"name": "Authorization", "value": "Bearer secret-auth-value"},
        {"name": "X-Goog-API-Key", "value": "secret-api-key-value"},
        {"name": "X-Custom-Token", "value": "secret-token-value"},
    ])
    har = tmp_path / "sensitive.har"
    har.write_text(json.dumps(fixture), encoding="utf-8")

    argv = sys.argv
    try:
        sys.argv = ["har_to_client.py", str(har), "--host", "example.com"]
        rc = mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out

    assert rc == 0
    assert "GET http://api.example.com/v1/items/{id}/reviews" in out
    assert "q = one" in out
    assert "blank = " in out
    assert out.count("[REDACTED]") == 3
    for secret in ("secret-auth-value", "secret-api-key-value", "secret-token-value"):
        assert secret not in out


def test_redacts_query_and_structured_body_credentials(tmp_path, capsys):
    mod = _load_module(DERIVE, "har_to_client_body_security_undertest")
    fixture = _make_har()
    entry = fixture["log"]["entries"][0]
    entry["request"]["url"] += "&access_token=query-secret"
    entry["request"]["queryString"].append(
        {"name": "access_token", "value": "query-secret"}
    )
    entry["request"]["method"] = "POST"
    entry["request"]["postData"] = {
        "mimeType": "application/json",
        "text": json.dumps({
            "password": "request-secret",
            "profile": {"token": "nested-secret", "token_count": 7},
        }),
    }
    entry["response"]["content"]["text"] = json.dumps({
        "access_token": "response-secret",
        "items": [{"client_secret": "deep-secret", "name": "kept"}],
    })
    har = tmp_path / "structured-secrets.har"
    har.write_text(json.dumps(fixture), encoding="utf-8")

    argv = sys.argv
    try:
        sys.argv = ["har_to_client.py", str(har)]
        rc = mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out

    assert rc == 0
    assert "access_token = [REDACTED]" in out
    assert '"password": "[REDACTED]"' in out
    assert '"token": "[REDACTED]"' in out
    assert '"token_count": 7' in out
    assert '"client_secret": "[REDACTED]"' in out
    assert '"name": "kept"' in out
    for secret in (
        "query-secret", "request-secret", "nested-secret",
        "response-secret", "deep-secret",
    ):
        assert secret not in out


def test_redacts_form_credentials_without_hiding_non_secret_token_fields():
    mod = _load_module(DERIVE, "har_to_client_form_security_undertest")
    sample = mod.body_sample(
        "application/x-www-form-urlencoded",
        "username=alice&password=form-secret&token_count=3",
        600,
    )
    assert "password=%5BREDACTED%5D" in sample
    assert "token_count=3" in sample
    assert "form-secret" not in sample


def test_cdp_capture_populates_query_string(monkeypatch):
    sync_api = types.ModuleType("playwright.sync_api")
    setattr(sync_api, "sync_playwright", object())
    playwright = types.ModuleType("playwright")
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    mod = _load_module(CAPTURE_CDP, "har_capture_cdp_query_undertest")

    class Request:
        method = "GET"
        url = "https://example.com/api?q=one&q=two&blank="
        headers = {}
        post_data = None
        resource_type = "xhr"

    entry = mod._har_entry(Request(), None)
    assert entry["request"]["queryString"] == [
        {"name": "q", "value": "one"},
        {"name": "q", "value": "two"},
        {"name": "blank", "value": ""},
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_cdp_capture_writes_owner_only_har(monkeypatch, tmp_path):
    sync_api = types.ModuleType("playwright.sync_api")
    setattr(sync_api, "sync_playwright", object())
    playwright = types.ModuleType("playwright")
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    mod = _load_module(CAPTURE_CDP, "har_capture_cdp_permissions_undertest")
    path = tmp_path / "private.har"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o644)

    mod._write_private_har(str(path), {"log": {"entries": []}})

    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"log": {"entries": []}}


def test_path_template_collapses_ids():
    mod = _load_module(DERIVE, "har_to_client_undertest2")
    assert mod.path_template("/v1/items/12345/x") == "/v1/items/{id}/x"
    assert mod.path_template("/v1/items/abc/x") == "/v1/items/abc/x"


def test_capture_actions_parse_ok():
    # har_capture imports playwright at module top; only assert the file is
    # syntactically valid and exposes run_action without importing playwright.
    src = CAPTURE.read_text(encoding="utf-8")
    compile(src, str(CAPTURE), "exec")
    assert "def run_action(" in src
    assert 'record_har_content="embed"' in src


def test_cdp_capture_is_valid_and_attaches_not_launches():
    # Covers the CDP pathway (cloud backends / /browser connect). Syntax-check
    # without importing playwright, and assert it attaches (connect_over_cdp)
    # and does NOT close a browser it doesn't own.
    src = CAPTURE_CDP.read_text(encoding="utf-8")
    compile(src, str(CAPTURE_CDP), "exec")
    assert "connect_over_cdp(" in src
    assert 'page.on("request"' in src and 'page.on("response"' in src
    # must not tear down a browser it merely attached to
    assert "browser.close()" not in src


def test_skill_documents_all_browser_pathways(skill_text: str):
    # The skill must route every Hermes browser backend to the right capturer.
    for token in ("Browserbase", "Browser-Use", "Firecrawl", "browser connect",
                  "har_capture_cdp.py", "connect_over_cdp"):
        assert token in skill_text, f"pathway coverage missing: {token}"
