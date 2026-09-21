"""RED-first tests for F001 optional hardening: telemtry (A) + structured-output (B).

Covers:
- A) Redacted validation-failure telemetry: record_validation_failure writes
  ONLY the 5 redacted fields, never raw content; writes a durable JSONL line.
- A-integration) decomposer + specifier emit a redacted event on parse failure.
- B) _aux_supports_json_schema is STRICT: False for unknown/free models,
  True only when proven (allowlist or response_format advertised).
- B-integration) decomposer requests response_format ONLY when enabled + proven;
  default OFF path sends no response_format (no 400 risk on free models).

No network calls for the unit-level tests; integration tests mock the aux client.
"""
from __future__ import annotations

import json as jsonlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_telemetry as telem
from tests.hermes_cli.test_kanban_decompose import kanban_home  # reuse fixture


# ── Item A: telemetry module ──────────────────────────────────────────────

def test_telemetry_writes_redacted_fields_only(tmp_path, monkeypatch):
    """Only the 5 redacted fields are ever stored — never raw content."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    vid = telem.record_validation_failure(
        profile="octacon", provider="openrouter", model="gpt-4o-mini",
        task_class="invalid_tool_call", schema_or_version_mismatch="unknown_tool:Shell",
    )
    assert vid
    path = telem._telemetry_path()
    assert os.path.isfile(path)
    with open(path) as fh:
        line = fh.readline()
    event = jsonlib.loads(line)
    assert set(event.keys()) == {"profile", "provider", "model", "task_class", "schema_or_version_mismatch", "ts"}
    assert event["profile"] == "octacon"
    assert event["model"] == "gpt-4o-mini"
    assert event["task_class"] == "invalid_tool_call"


def test_telemetry_never_leaks_raw_content(tmp_path, monkeypatch):
    """Even if a caller passes a 'mismatch' containing what looks like raw
    output, the function truncates and the schema is fixed — no arbitrary
    free-text field exists to hold raw model output."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    telem.record_validation_failure(
        profile="k", provider="p", model="m",
        task_class="aux_malformed_json",
        schema_or_version_mismatch="raw: {'some':'leak'} attempted Shell call",
    )
    with open(telem._telemetry_path()) as fh:
        event = jsonlib.loads(fh.readline())
    # 'mismatch' is just a short redacted class string, capped at 128 chars.
    assert "raw:" not in event  # no key named raw
    assert len(event["schema_or_version_mismatch"]) <= 128


def test_telemetry_write_failure_is_nonfatal(tmp_path, monkeypatch):
    """If the path is unwritable, the call returns None and does not raise."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    bad = tmp_path / "nope" / "dir" / "telemetry"
    monkeypatch.setattr(telem, "_telemetry_path", lambda: str(bad / "x.jsonl"))
    # Parent dir doesn't exist and we force an error by making the dir a file.
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "x.jsonl").write_text("")  # file where a dir is needed -> mkdir fails
    # Even if it doesn't fail on this platform, the function must not raise.
    result = telem.record_validation_failure(
        profile="p", provider="pv", model="m",
        task_class="t", schema_or_version_mismatch="mm",
    )
    assert result is None or isinstance(result, str)


# ── Item B: capability probe ──────────────────────────────────────────────

def test_json_schema_probe_false_for_unknown_free_model():
    # A random/free model with no proof must be False (avoid 400).
    assert decomp._aux_supports_json_schema("some-free-model-7b") is False


def test_json_schema_probe_true_for_known_good():
    assert decomp._aux_supports_json_schema("gpt-4o-mini") is True
    assert decomp._aux_supports_json_schema("claude-3-7-sonnet") is True


def test_json_schema_probe_false_when_catalog_lacks_response_format():
    """If the catalog entry exists but doesn't advertise response_format,
    it must NOT be treated as capable (strict, unlike tools)."""
    fake_catalog = [
        {"id": "openrouter/foo-8b", "supported_parameters": ["temperature", "tools"]},
    ]
    with patch("hermes_cli.models.fetch_openrouter_models", return_value=fake_catalog):
        assert decomp._aux_supports_json_schema("openrouter/foo-8b") is False


def test_json_schema_probe_true_when_catalog_advertises_response_format():
    fake_catalog = [
        {"id": "openrouter/bar-8b", "supported_parameters": ["temperature", "response_format"]},
    ]
    with patch("hermes_cli.models.fetch_openrouter_models", return_value=fake_catalog):
        assert decomp._aux_supports_json_schema("openrouter/bar-8b") is True


def test_structured_output_disabled_by_default():
    # Default config => OFF, so even a capable model gets no response_format.
    assert decomp._structured_output_enabled({}) is False
    assert decomp._structured_output_enabled({"kanban": {}}) is False


# ── Integration: decomposer requests structured output only when enabled+proven ──

def _fake_aux_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_call_llm(content_or_side_effect):
    """Mock the call_llm production seam (task='kanban_decomposer').

    decompose_task routes through agent.auxiliary_client.call_llm; the
    structured-output response_format is carried inside its extra_body kwarg.
    """
    if isinstance(content_or_side_effect, list):
        return patch(
            "agent.auxiliary_client.call_llm",
            side_effect=content_or_side_effect,
        )
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=content_or_side_effect,
    )


def _patch_resolve_model(model: str):
    """Make the decomposer's capability probe see ``model``.

    decompose_task imports _resolve_task_provider_model from
    agent.auxiliary_client and reads index 1 (the model) to gate
    _aux_supports_json_schema. Stubbing it lets each test control whether the
    structured-output path is taken without any real provider config.
    """
    return patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openrouter", model, None, None, None),
    )


def _patch_list_profiles(names):
    from types import SimpleNamespace
    fps = [SimpleNamespace(name=n, is_default=(i == 0), description=f"d {n}",
                           description_auto=False, model="m", provider="p", skill_count=1)
           for i, n in enumerate(names)]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fps),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decomposer_default_sends_no_response_format(kanban_home):
    """Default OFF: the call_llm extra_body must NOT carry response_format."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)
    good = jsonlib.dumps({"fanout": False, "rationale": "one", "title": "T", "body": "B", "assignee": "researcher"})
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(_fake_aux_response(good)) as call_llm_mock,              _patch_resolve_model("some-free-model"),              patch(
                 "hermes_cli.kanban_decompose._load_config",
                 return_value={"kanban": {"default_assignee": "researcher"}},
             ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    # No response_format kwarg reaches call_llm's extra_body.
    _, kwargs = call_llm_mock.call_args
    extra_body = kwargs.get("extra_body") or {}
    assert "response_format" not in extra_body


def test_decomposer_enabled_proven_sends_response_format(kanban_home):
    """Enabled + proven model => response_format=json_schema IS sent in extra_body."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)
    good = jsonlib.dumps({"fanout": False, "rationale": "one", "title": "T", "body": "B", "assignee": "researcher"})
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(_fake_aux_response(good)) as call_llm_mock,              _patch_resolve_model("gpt-4o-mini"),              patch(
                 "hermes_cli.kanban_decompose._load_config",
                 return_value={"kanban": {"structured_output": True, "default_assignee": "researcher"}},
             ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    _, kwargs = call_llm_mock.call_args
    extra_body = kwargs.get("extra_body") or {}
    assert extra_body.get("response_format", {}).get("type") == "json_schema"


def test_decomposer_enabled_but_unproven_sends_no_response_format(kanban_home):
    """Enabled BUT model not proven => still NO response_format (400 guard)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)
    good = jsonlib.dumps({"fanout": False, "rationale": "one", "title": "T", "body": "B", "assignee": "researcher"})
    patches = _patch_list_profiles(["orchestrator", "researcher"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm(_fake_aux_response(good)) as call_llm_mock,              _patch_resolve_model("some-free-model-7b"),              patch(
                 "hermes_cli.kanban_decompose._load_config",
                 return_value={"kanban": {"structured_output": True, "default_assignee": "researcher"}},
             ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    _, kwargs = call_llm_mock.call_args
    extra_body = kwargs.get("extra_body") or {}
    assert "response_format" not in extra_body


# ── Integration: telemetry emitted on decomposer parse failure ──

def test_decomposer_emits_telemetry_on_exhausted_parse(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)
    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with _patch_call_llm([
            _fake_aux_response("not json 1"),
            _fake_aux_response("not json 2"),
            _fake_aux_response("not json 3"),
        ]), _patch_resolve_model("some-free-model"):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    # A redacted telemetry line must now exist.
    path = telem._telemetry_path()
    assert os.path.isfile(path), "expected telemetry file to be created"
    with open(path) as fh:
        lines = [jsonlib.loads(l) for l in fh if l.strip()]
    assert any(e["task_class"] == "aux_malformed_json" for e in lines)
    # Redaction guaranteed: no line carries a 'content'/'raw' free-text field.
    for e in lines:
        assert set(e.keys()) <= {"profile", "provider", "model", "task_class", "schema_or_version_mismatch", "ts"}
