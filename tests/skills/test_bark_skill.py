"""The bark skill ships a usable helper and honest setup metadata.

No network: every request goes through an injected opener or a monkeypatched
``urllib.request.urlopen``.
"""

from __future__ import annotations

import email.message
import http.client
import importlib.util
import io
import json
import re
import urllib.error
from pathlib import Path

import hermes_yaml as yaml
import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "bark"
)
SCRIPT = SKILL_DIR / "scripts" / "bark_push.py"
MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge)\b", re.IGNORECASE
)
SECRET = "s3cr3t-device-key"


def _skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


def _frontmatter() -> dict:
    text = _skill_text()
    assert text.startswith("---"), "SKILL.md must start with frontmatter"
    end = re.search(r"\n---\s*\n", text[3:])
    assert end, "unclosed frontmatter"
    return yaml.safe_load(text[3 : end.start() + 3])


def _load_helper():
    spec = importlib.util.spec_from_file_location("bark_push", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bark = _load_helper()


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_frontmatter_matches_the_shipped_directory():
    meta = _frontmatter()
    assert meta["name"] == "bark"
    assert meta["version"] and meta["author"] and meta["license"] == "MIT"


def test_description_stays_within_the_hardline():
    description = _frontmatter()["description"]
    assert len(description) <= 60, len(description)
    assert description.endswith(".")
    assert not MARKETING.search(description)


def test_platforms_cover_every_host_the_helper_runs_on():
    assert set(_frontmatter()["platforms"]) == {"macos", "linux", "windows"}


def test_bark_token_is_declared_as_a_required_secret():
    entries = _frontmatter()["required_environment_variables"]
    token = next(entry for entry in entries if entry["name"] == "BARK_TOKEN")
    assert token["prompt"] and token["help"] and token["required_for"]


def test_server_address_is_skill_config_not_an_environment_variable():
    config = _frontmatter()["metadata"]["hermes"]["config"]
    entry = next(item for item in config if item["key"] == "bark.server")
    assert entry["default"] == "https://api.day.app"
    assert entry["description"] and entry["prompt"]


def test_body_keeps_the_modern_section_order():
    headings = re.findall(r"^## (.+)$", _skill_text(), re.MULTILINE)
    expected = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    assert headings[: len(expected)] == expected


def test_payload_drops_unset_fields():
    assert bark.build_payload(body="done", badge=0) == {"body": "done", "badge": 0}


def test_markdown_replaces_the_plain_body():
    payload = bark.build_payload(body="ignored", markdown="**done**")
    assert payload == {"markdown": "**done**"}


def test_payload_requires_content():
    with pytest.raises(ValueError, match="--body or --markdown"):
        bark.build_payload(title="no body")


def test_payload_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="unsupported level"):
        bark.build_payload(body="x", level="loud")


def test_withdrawing_requires_an_id():
    with pytest.raises(ValueError, match="--delete needs --id"):
        bark.build_payload(body="x", delete=True)


def test_call_uses_the_bark_string_flag():
    assert bark.build_payload(body="x", call=True)["call"] == "1"


def test_withdraw_payload_carries_only_id_and_delete():
    assert bark.build_payload(push_id="build-42", delete=True) == {
        "id": "build-42",
        "delete": "1",
    }


def test_withdraw_dry_run_succeeds_without_a_body(capsys):
    assert bark.main(["--id", "build-42", "--delete", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"id": "build-42", "delete": "1"}


def test_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("BARK_TOKEN", "abc123")
    assert bark.resolve_token() == "abc123"


def test_there_is_no_device_key_flag():
    """A key on the command line would land in history, the process list and transcripts."""
    with pytest.raises(SystemExit):
        bark.build_parser().parse_args(["--device-key", "abc123", "--body", "x"])


def test_missing_token_names_the_fix(monkeypatch):
    monkeypatch.delenv("BARK_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="BARK_TOKEN"):
        bark.resolve_token()


def test_server_defaults_and_trims_trailing_slash():
    assert bark.resolve_server() == "https://api.day.app"
    assert bark.resolve_server("https://bark.example.com/") == "https://bark.example.com"


def test_server_environment_variable_is_not_read(monkeypatch):
    monkeypatch.setenv("BARK_SERVER", "https://env.example.com")
    assert bark.resolve_server() == "https://api.day.app"


def test_send_posts_json_to_the_device_endpoint():
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(b'{"code":200,"message":"success"}')

    result = bark.send_push({"body": "你好"}, "abc123", opener=opener)

    assert result == {"code": 200, "message": "success"}
    request = captured["request"]
    assert request.full_url == "https://api.day.app/abc123"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"body": "你好"}
    assert request.headers["Content-type"] == "application/json; charset=utf-8"


def test_send_surfaces_the_server_error_body():
    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            email.message.Message(),
            io.BytesIO(b'{"code":400,"message":"failed to get device token"}'),
        )

    with pytest.raises(bark.BarkError, match="failed to get device token"):
        bark.send_push({"body": "x"}, "abc123", opener=opener)


def test_send_reports_an_unreachable_server():
    def opener(request, timeout):
        raise urllib.error.URLError("nodename nor servname provided")

    with pytest.raises(bark.BarkError, match="cannot reach"):
        bark.send_push({"body": "x"}, "abc123", opener=opener)


def test_dry_run_prints_without_touching_the_network(capsys, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("dry run must not send")

    monkeypatch.setattr(bark, "send_push", explode)
    assert bark.main(["--body", "hello", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"body": "hello"}


def test_dry_run_never_prints_the_device_key(capsys, monkeypatch):
    monkeypatch.setenv("BARK_TOKEN", SECRET)
    assert bark.main(["--body", "hello", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err


def test_a_rejected_push_exits_non_zero(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", "abc123")
    monkeypatch.setattr(bark, "send_push", lambda *a, **k: {"code": 400, "message": "bad key"})
    assert bark.main(["--body", "hello"]) == 1
    assert "bad key" in capsys.readouterr().out


def test_http_errors_never_print_the_device_key(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            email.message.Message(),
            io.BytesIO(f'{{"code":400,"message":"bad key {SECRET}"}}'.encode()),
        )

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert bark.REDACTED in captured.err


def test_network_errors_never_print_the_device_key(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        raise urllib.error.URLError(f"failed to reach {request.full_url}")

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert bark.REDACTED in captured.err


def test_unexpected_responses_never_print_the_device_key(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        return _Response(f"<html>bad request for {request.full_url}</html>".encode())

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert bark.REDACTED in captured.err


def test_read_timeouts_are_reported_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 1
    captured = capsys.readouterr()
    assert "timed out" in captured.err
    assert "Traceback" not in captured.err
    assert SECRET not in captured.err


def test_http_protocol_errors_are_reported_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        raise http.client.BadStatusLine(f"bad status for {request.full_url}")

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 1
    captured = capsys.readouterr()
    assert "BadStatusLine" in captured.err
    assert "Traceback" not in captured.err
    assert SECRET not in captured.err
    assert bark.REDACTED in captured.err


def test_successful_output_never_echoes_the_device_key(monkeypatch, capsys):
    monkeypatch.setenv("BARK_TOKEN", SECRET)

    def opener(request, timeout):
        return _Response(json.dumps({"code": 200, "echo": SECRET}).encode())

    monkeypatch.setattr(bark.urllib.request, "urlopen", opener)
    assert bark.main(["--body", "hello"]) == 0
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
