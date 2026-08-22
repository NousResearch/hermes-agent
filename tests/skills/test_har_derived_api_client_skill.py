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
import re
import sys
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "web-development"
    / "har-derived-api-client"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS = SKILL_DIR / "scripts"
CAPTURE = SCRIPTS / "har_capture.py"
CAPTURE_CDP = SCRIPTS / "har_capture_cdp.py"
ACTIONS = SCRIPTS / "har_actions.py"
DERIVE = SCRIPTS / "har_to_client.py"


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
    assert ACTIONS.is_file()
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


def _run_derive(mod, har_dict, tmp_path, capsys, *extra):
    har = tmp_path / "t.har"
    har.write_text(json.dumps(har_dict), encoding="utf-8")
    argv = sys.argv
    try:
        sys.argv = ["har_to_client.py", str(har), *extra]
        rc = mod.main()
    finally:
        sys.argv = argv
    return rc, capsys.readouterr().out


def test_path_template_collapses_ids():
    mod = _load_module(DERIVE, "har_to_client_undertest2")
    assert mod.path_template("/v1/items/12345/x") == "/v1/items/{id}/x"
    assert mod.path_template("/v1/items/abc/x") == "/v1/items/abc/x"


def test_form_params_and_null_postdata(tmp_path, capsys):
    mod = _load_module(DERIVE, "har_to_client_postdata")
    fixture = {
        "log": {
            "entries": [
                {
                    "_resourceType": "xhr",
                    "request": {
                        "method": "POST",
                        "url": "https://api.example.com/login",
                        "queryString": [],
                        "headers": [{"name": "content-type", "value": "application/x-www-form-urlencoded"}],
                        "postData": {
                            "mimeType": "application/x-www-form-urlencoded",
                            "params": [
                                {"name": "user", "value": "ada"},
                                {"name": "pass", "value": "secret"},
                            ],
                        },
                    },
                    "response": {
                        "status": 200,
                        "content": {"mimeType": "application/json", "text": '{"ok":true}'},
                    },
                },
                {
                    "_resourceType": "xhr",
                    "request": {
                        "method": "GET",
                        "url": "https://api.example.com/v1/search?q=hi",
                        "queryString": [{"name": "q", "value": "hi"}],
                        "headers": [{"name": "User-Agent", "value": "Mozilla/5.0 TestBrowser/1.0"}],
                        "postData": None,
                    },
                    "response": {
                        "status": 200,
                        "content": {"mimeType": "application/json", "text": "{}"},
                    },
                },
            ]
        }
    }
    rc, out = _run_derive(mod, fixture, tmp_path, capsys)
    assert rc == 0
    assert "user=ada" in out
    assert "pass=secret" in out
    assert "GET https://api.example.com/v1/search" in out


def test_decodes_base64_response_body(tmp_path, capsys):
    mod = _load_module(DERIVE, "har_to_client_b64")
    fixture = _make_har()
    fixture["log"]["entries"][0]["response"]["content"] = {
        "mimeType": "application/json",
        "encoding": "base64",
        "text": "eyJvayI6dHJ1ZX0=",  # {"ok":true}
    }
    rc, out = _run_derive(mod, fixture, tmp_path, capsys, "--host", "example.com")
    assert rc == 0
    assert '{"ok":true}' in out
    assert "eyJvayI6dHJ1ZX0=" not in out


def test_run_action_validates_specs_and_drives_page():
    mod = _load_module(ACTIONS, "har_actions_undertest")

    class Page:
        def __init__(self):
            self.calls = []

        def fill(self, sel, text):
            self.calls.append(("fill", sel, text))

        def press(self, sel, key):
            self.calls.append(("press", sel, key))

        def click(self, sel):
            self.calls.append(("click", sel))

        def goto(self, url):
            self.calls.append(("goto", url))

    page = Page()
    mod.run_action(page, "fill:#q:hello:world")
    mod.run_action(page, "goto:https://ex.com:8080/a")
    mod.run_action(page, "click:button.submit")
    assert page.calls == [
        ("fill", "#q", "hello:world"),
        ("goto", "https://ex.com:8080/a"),
        ("click", "button.submit"),
    ]
    for spec in ("fill:onlysel", "fill::text", "click", "sleep:", "sleep:1:30",
                 "press:sel", "press::Enter", "goto", "nope:x"):
        with pytest.raises(ValueError):
            mod.run_action(page, spec)


def test_cdp_goto_opens_new_page_and_flushes_pending():
    mod = _load_module(ACTIONS, "har_actions_cdp")

    class Context:
        def __init__(self, pages):
            self.pages = list(pages)
            self.created = []

        def new_page(self):
            page = object()
            self.created.append(page)
            self.pages.append(page)
            return page

    existing = object()
    ctx = Context([existing])
    driven = mod.choose_drive_page(ctx, new_page=True)
    assert driven is not existing
    assert ctx.created == [driven]
    assert mod.choose_drive_page(ctx, new_page=False) is driven

    class Req:
        # Playwright's request.response() blocks until the response arrives.
        # A leftover request after --wait may never get one, so the flush
        # must not call it at all.
        def response(self):
            raise AssertionError("flush_pending must not call the waiting response()")

    reqs = [Req(), Req()]
    pending = {1: reqs[0], 2: reqs[1]}
    entries = []
    mod.flush_pending(pending, entries, lambda req, resp: (req, resp))
    assert pending == {}
    assert [req for req, _ in entries] == reqs
    assert all(resp is None for _, resp in entries)


def test_skill_documents_all_browser_pathways(skill_text: str):
    # The skill must route every Hermes browser backend to the right capturer.
    for token in ("Browserbase", "Browser-Use", "Firecrawl", "browser connect",
                  "har_capture_cdp.py", "connect_over_cdp"):
        assert token in skill_text, f"pathway coverage missing: {token}"
