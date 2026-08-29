"""Tests for the optional Context7 HTTP skill (no live network)."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request


REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "software-development" / "context7"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "context7.py"


class FakeResponse:
    def __init__(self, body: str, content_type: str = "application/json") -> None:
        self._body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.status = 200

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def load_module():
    assert SCRIPT_PATH.is_file(), f"missing Context7 helper: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("context7_skill", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lookup_resolves_library_then_returns_text_context():
    module = load_module()
    calls: list[tuple[str, dict[str, str]]] = []

    def opener(request, timeout):
        calls.append((request.full_url, dict(request.header_items())))
        if "/libs/search?" in request.full_url:
            return FakeResponse(
                json.dumps({"results": [{"id": "/reactjs/react.dev", "title": "React"}]})
            )
        return FakeResponse("React useState documentation", "text/plain; charset=utf-8")

    result = module.lookup(
        "react",
        "How do I use useState?",
        response_type="txt",
        opener=opener,
    )

    assert result == "React useState documentation"
    assert len(calls) == 2
    assert "/api/v2/libs/search?" in calls[0][0]
    assert "libraryName=react" in calls[0][0]
    assert "/api/v2/context?" in calls[1][0]
    assert "libraryId=%2Freactjs%2Freact.dev" in calls[1][0]


def test_context_follows_context7_library_redirect_once():
    module = load_module()
    calls: list[str] = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            payload = json.dumps(
                {
                    "error": "library_redirected",
                    "message": "Library moved",
                    "redirectUrl": "/react/react",
                }
            ).encode("utf-8")
            raise HTTPError(request.full_url, 301, "Moved", Message(), io.BytesIO(payload))
        return FakeResponse("redirected documentation", "text/plain; charset=utf-8")

    result = module.get_context(
        "/facebook/react",
        "useState",
        response_type="txt",
        opener=opener,
    )

    assert result == "redirected documentation"
    assert len(calls) == 2
    assert "libraryId=%2Freact%2Freact" in calls[1]


def test_context_stops_after_one_library_redirect():
    module = load_module()
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        payload = json.dumps(
            {
                "error": "library_redirected",
                "message": "Library moved again",
                "redirectUrl": f"/react/react-v{calls}",
            }
        ).encode("utf-8")
        raise HTTPError(request.full_url, 301, "Moved", Message(), io.BytesIO(payload))

    try:
        module.get_context("/facebook/react", "useState", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 301
    else:
        raise AssertionError("a second Context7 redirect must be surfaced")

    assert calls == 2


def test_network_errors_are_wrapped_with_a_readable_message():
    module = load_module()

    def opener(request, timeout):
        raise URLError("DNS unavailable")

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 0
        assert "DNS unavailable" in str(exc)
    else:
        raise AssertionError("network failures must be wrapped as Context7Error")


def test_http_redirect_handler_never_forwards_authorization():
    module = load_module()
    request = Request(
        "https://context7.com/api/v2/context",
        headers={"Authorization": "Bearer ctx7sk-test-secret"},
    )
    headers = Message()
    headers["Location"] = "https://attacker.example/collect"

    redirected = module._NoRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        headers,
        headers["Location"],
    )

    assert redirected is None


def test_default_requests_use_the_redirect_blocking_opener():
    module = load_module()
    calls: list[str] = []

    def safe_opener(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(json.dumps({"results": []}))

    setattr(module, "_open_without_redirects", safe_opener)
    result = module.search_libraries("react", "hooks", opener=None)

    assert result == {"results": []}
    assert len(calls) == 1


def test_cli_exposes_search_context_and_lookup_commands():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "{search,context,lookup}" in result.stdout
    assert "CONTEXT7_API_KEY" in result.stdout


def test_skill_metadata_and_workflow_are_complete():
    assert SKILL_MD.is_file(), f"missing skill definition: {SKILL_MD}"
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: context7" in text
    description_line = next(line for line in text.splitlines() if line.startswith("description:"))
    description = description_line.partition(":")[2].strip().strip('"')
    assert len(description) <= 60
    assert description.endswith(".")
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in text
    assert "/api/v2/libs/search" in text
    assert "/api/v2/context" in text
    assert "scripts/context7.py" in text
