"""Tests for the optional Context7 HTTP skill (no live network)."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from email.message import Message
from http.client import IncompleteRead
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

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]

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


def test_api_key_is_stripped_and_sent_only_in_authorization_header():
    module = load_module()
    calls: list[tuple[str, dict[str, str]]] = []

    def opener(request, timeout):
        calls.append((request.full_url, dict(request.header_items())))
        return FakeResponse(json.dumps({"results": []}))

    module.search_libraries(
        "react",
        "hooks",
        api_key="  ctx7sk-test-secret  ",
        opener=opener,
    )

    assert len(calls) == 1
    url, headers = calls[0]
    assert "ctx7sk-test-secret" not in url
    assert headers["Authorization"] == "Bearer ctx7sk-test-secret"


def test_whitespace_api_key_is_not_sent():
    module = load_module()
    headers: dict[str, str] = {}

    def opener(request, timeout):
        headers.update(dict(request.header_items()))
        return FakeResponse(json.dumps({"results": []}))

    module.search_libraries("react", "hooks", api_key="   ", opener=opener)

    assert "Authorization" not in headers


def test_non_object_http_error_payload_is_wrapped():
    module = load_module()

    def opener(request, timeout):
        payload = json.dumps(["unexpected", "shape"]).encode("utf-8")
        raise HTTPError(request.full_url, 429, "Too Many Requests", Message(), io.BytesIO(payload))

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 429
        assert "unexpected error payload" in str(exc).lower()
    else:
        raise AssertionError("non-object error JSON must be wrapped as Context7Error")


def test_excessively_nested_http_error_json_is_wrapped():
    module = load_module()
    nested_json = "[" * 1100 + "0" + "]" * 1100

    def opener(request, timeout):
        raise HTTPError(
            request.full_url,
            500,
            "Server Error",
            Message(),
            io.BytesIO(nested_json.encode("utf-8")),
        )

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 500
        assert "invalid error json" in str(exc).lower()
    else:
        raise AssertionError("excessively nested HTTP error JSON must be wrapped")


def test_api_key_is_redacted_from_http_errors():
    module = load_module()
    api_key = "ctx7sk-test-secret"

    def opener(request, timeout):
        payload = json.dumps({"message": f"upstream echoed {api_key}"}).encode("utf-8")
        raise HTTPError(request.full_url, 500, "Server Error", Message(), io.BytesIO(payload))

    try:
        module.search_libraries("react", "hooks", api_key=api_key, opener=opener)
    except module.Context7Error as exc:
        assert api_key not in str(exc)
        assert "[REDACTED]" in str(exc)
    else:
        raise AssertionError("API keys must be redacted from HTTP errors")


def test_api_key_is_redacted_from_nested_http_error_payloads():
    module = load_module()
    api_key = "ctx7sk-test-secret"

    def opener(request, timeout):
        payload = json.dumps({"message": ["upstream echoed", {"key": api_key}]}).encode("utf-8")
        raise HTTPError(request.full_url, 500, "Server Error", Message(), io.BytesIO(payload))

    try:
        module.search_libraries("react", "hooks", api_key=api_key, opener=opener)
    except module.Context7Error as exc:
        assert api_key not in str(exc)
        assert "[REDACTED]" in str(exc)
    else:
        raise AssertionError("API keys must be redacted recursively")


def test_api_key_redaction_bounds_nested_payload_depth():
    module = load_module()
    api_key = "ctx7sk-test-secret"
    payload: object = api_key
    for _ in range(1100):
        payload = [payload]

    redacted = module._redact_secret(payload, api_key)

    assert "nested value omitted" in repr(redacted)
    assert api_key not in repr(redacted)


def test_api_key_is_redacted_from_json_object_keys():
    module = load_module()
    api_key = "ctx7sk-test-secret"

    redacted = module._redact_secret({api_key: "echo"}, api_key)

    assert api_key not in repr(redacted)
    assert "[REDACTED]" in repr(redacted)


def test_http_error_messages_are_bounded():
    module = load_module()
    setattr(module, "MAX_ERROR_MESSAGE_CHARS", 32)

    def opener(request, timeout):
        payload = json.dumps({"message": "x" * 100}).encode("utf-8")
        raise HTTPError(request.full_url, 500, "Server Error", Message(), io.BytesIO(payload))

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert len(str(exc)) <= 35
        assert str(exc).endswith("...")
    else:
        raise AssertionError("HTTP error messages must be bounded")


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


def test_json_context_rejects_non_json_content_type():
    module = load_module()

    def opener(request, timeout):
        return FakeResponse(json.dumps({"snippets": []}), "text/plain")

    try:
        module.get_context(
            "/react/react",
            "hooks",
            response_type="json",
            opener=opener,
        )
    except module.Context7Error as exc:
        assert "content type" in str(exc).lower()
    else:
        raise AssertionError("JSON context must declare a JSON content type")


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


def test_context_rejects_invalid_library_redirect_ids():
    module = load_module()

    for redirect_value in ({"id": "/react/react"}, "https://attacker.example/library"):
        calls = 0

        def opener(request, timeout):
            nonlocal calls
            calls += 1
            payload = json.dumps(
                {
                    "error": "library_redirected",
                    "message": "Library moved",
                    "redirectUrl": redirect_value,
                }
            ).encode("utf-8")
            raise HTTPError(request.full_url, 301, "Moved", Message(), io.BytesIO(payload))

        try:
            module.get_context("/facebook/react", "useState", opener=opener)
        except module.Context7Error as exc:
            assert exc.status == 301
            assert "redirect" in str(exc).lower()
        else:
            raise AssertionError("invalid redirect IDs must be rejected")

        assert calls == 1


def test_context_rejects_library_redirect_self_loop():
    module = load_module()
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        payload = json.dumps(
            {
                "error": "library_redirected",
                "message": "Library moved",
                "redirectUrl": "/facebook/react",
            }
        ).encode("utf-8")
        raise HTTPError(request.full_url, 301, "Moved", Message(), io.BytesIO(payload))

    try:
        module.get_context("/facebook/react", "useState", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 301
        assert "self" in str(exc).lower()
    else:
        raise AssertionError("redirect self-loops must be rejected")

    assert calls == 1


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


def test_timeouts_are_wrapped_with_a_readable_message():
    module = load_module()

    def opener(request, timeout):
        raise TimeoutError("read timed out")

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 0
        assert "timed out" in str(exc).lower()
    else:
        raise AssertionError("timeouts must be wrapped as Context7Error")


def test_http_error_body_timeouts_are_wrapped():
    module = load_module()

    class TimeoutBody(io.BytesIO):
        def read(self, amount: int | None = -1) -> bytes:
            raise TimeoutError("error body read timed out")

    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "Server Error", Message(), TimeoutBody())

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 500
        assert "timed out" in str(exc).lower()
    else:
        raise AssertionError("HTTP error-body timeouts must be wrapped")


def test_response_body_size_is_bounded():
    module = load_module()
    setattr(module, "MAX_RESPONSE_BYTES", 8)

    def opener(request, timeout):
        return FakeResponse("x" * 9, "text/plain")

    try:
        module.get_context("/react/react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert exc.status == 0
        assert "response exceeds" in str(exc).lower()
    else:
        raise AssertionError("oversized responses must be rejected")


def test_incomplete_response_reads_are_wrapped():
    module = load_module()

    class IncompleteResponse(FakeResponse):
        def read(self, amount: int = -1) -> bytes:
            raise IncompleteRead(b"partial", 10)

    def opener(request, timeout):
        return IncompleteResponse("ignored")

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "read" in str(exc).lower()
        assert "incomplete" in str(exc).lower()
    else:
        raise AssertionError("incomplete response reads must be wrapped")


def test_os_response_read_errors_are_wrapped():
    module = load_module()

    class BrokenResponse(FakeResponse):
        def read(self, amount: int = -1) -> bytes:
            raise OSError("connection reset")

    def opener(request, timeout):
        return BrokenResponse("ignored")

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "read" in str(exc).lower()
        assert "connection reset" in str(exc).lower()
    else:
        raise AssertionError("OS response read errors must be wrapped")


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


def test_search_rejects_non_object_json():
    module = load_module()

    def opener(request, timeout):
        return FakeResponse(json.dumps(["unexpected", "shape"]))

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "search response" in str(exc).lower()
        assert "json object" in str(exc).lower()
    else:
        raise AssertionError("search responses must be JSON objects")


def test_search_wraps_excessively_nested_json():
    module = load_module()
    nested_json = "[" * 1100 + "0" + "]" * 1100

    def opener(request, timeout):
        return FakeResponse(nested_json)

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "invalid json" in str(exc).lower()
    else:
        raise AssertionError("excessively nested JSON must be wrapped")


def test_search_rejects_non_json_content_type():
    module = load_module()

    def opener(request, timeout):
        return FakeResponse(json.dumps({"results": []}), "text/plain")

    try:
        module.search_libraries("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "content type" in str(exc).lower()
    else:
        raise AssertionError("search responses must declare a JSON content type")


def test_lookup_rejects_missing_library_id():
    module = load_module()

    def opener(request, timeout):
        return FakeResponse(json.dumps({"results": [{"title": "React"}]}))

    try:
        module.lookup("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "library id" in str(exc).lower()
    else:
        raise AssertionError("lookup results must contain a valid library ID")


def test_lookup_rejects_absolute_library_id():
    module = load_module()
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse(
            json.dumps(
                {
                    "results": [
                        {"id": "https://attacker.example/library", "title": "React"}
                    ]
                }
            )
        )

    try:
        module.lookup("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "library id" in str(exc).lower()
    else:
        raise AssertionError("absolute URLs must not be accepted as library IDs")

    assert calls == 1


def test_library_id_validation_rejects_malformed_paths():
    module = load_module()
    malformed_ids = (
        "/",
        "//attacker.example/library",
        "/has space/library",
        "/owner/repo?query=value",
        "/owner/repo#fragment",
        "/owner\\repo",
        "/owner/../repo",
        "/owner/\x00repo",
    )

    for library_id in malformed_ids:
        try:
            module._validate_library_id(library_id, status=0, label="test")
        except module.Context7Error:
            continue
        raise AssertionError(f"malformed library ID was accepted: {library_id!r}")


def test_context_rejects_invalid_direct_library_ids_before_request():
    module = load_module()
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse("must not be requested", "text/plain")

    for library_id in ("https://attacker.example/library", "/owner/../repo"):
        try:
            module.get_context(library_id, "hooks", opener=opener)
        except module.Context7Error as exc:
            assert "library id" in str(exc).lower()
        else:
            raise AssertionError("invalid direct library IDs must be rejected")

    assert calls == 0


def test_lookup_rejects_non_list_results():
    module = load_module()

    def opener(request, timeout):
        return FakeResponse(json.dumps({"results": {"id": "/react/react"}}))

    try:
        module.lookup("react", "hooks", opener=opener)
    except module.Context7Error as exc:
        assert "results" in str(exc).lower()
        assert "list" in str(exc).lower()
    else:
        raise AssertionError("lookup results must be a list")


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
