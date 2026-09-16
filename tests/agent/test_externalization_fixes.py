"""Externalization verification fixes for transcript sanitation validation.

Two fail-open paths in ``agent.conversation_sanitation._validation_views``
(issues 4018592961 and 4018592967):

- An already-externalized transcript (marker content byte-identical on both
  sides) was permanently blocked from further sanitation: verified-expansion
  rewrote only the candidate view to the sidecar secret while the original
  view kept the marker, so structural validation rejected the whole
  candidate. A marker that survives byte-identical from original to candidate
  must pass through without expansion.
- A nested externalization marker under a ``content`` part validated through
  the string externalization grant without sidecar verification, so a
  missing/invalid sidecar still committed and destroyed the last recoverable
  copy of the payload. Nested markers must verify through the same loader
  contract as top-level markers and fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from agent.conversation_sanitation import (
    SanitationChanges,
    validate_sanitation_candidate,
)


def _tool_marker(secret: str, tool_call_id: str = "call-real") -> str:
    digest = hashlib.sha256(secret.encode()).hexdigest()[:12]
    return (
        f"[Externalized tool output: tool_call_id={tool_call_id}; "
        f"chars={len(secret)}; bytes={len(secret.encode())}; "
        f"ref=20260915_{tool_call_id}_{digest}_abc123.json]"
    )


def _payload_marker(secret: str, role: str) -> str:
    digest = hashlib.sha256(secret.encode()).hexdigest()[:12]
    return (
        "[Externalized payload: kind=raw_payload; "
        f"role={role}; chars={len(secret)}; bytes={len(secret.encode())}; "
        f"ref=20260915_raw_payload_{role}_{digest}_abc123.json]"
    )


def _tool_payload(secret: str, tool_call_id: str = "call-real") -> dict[str, Any]:
    return {
        "kind": "tool_result",
        "tool_call_id": tool_call_id,
        "content": secret,
        "content_chars": len(secret),
        "content_bytes": len(secret.encode()),
    }


def _payload_payload(secret: str, role: str) -> dict[str, Any]:
    return {
        "kind": "raw_payload",
        "role": role,
        "content": secret,
        "content_chars": len(secret),
        "content_bytes": len(secret.encode()),
    }


def _placeholder(pattern: str, secret: str) -> str:
    return (
        f"[LCM sensitive redaction: name={pattern}; "
        f"chars={len(secret)}; bytes={len(secret.encode())}; "
        f"sha256={hashlib.sha256(secret.encode()).hexdigest()[:16]}]"
    )


# ---- bug 4018592961: unchanged verified markers must pass through ----


def test_unchanged_verified_marker_transcript_still_validates():
    """Second-pass sanitation is not blocked by a verified marker already in
    the transcript: the byte-identical marker passes through while other
    fields still validate."""
    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [
        {"role": "tool", "tool_call_id": "call-real", "content": marker},
        {"role": "user", "content": "password= hunter2"},
    ]
    candidate = copy.deepcopy(original)
    candidate[1]["content"] = "password= " + _placeholder("password", "hunter2")

    changes = validate_sanitation_candidate(
        original,
        candidate,
        externalized_payload_loader=lambda _ref: _tool_payload(secret),
    )

    assert changes is not None
    assert changes.placeholders == 1
    assert changes.changed_fields == 1


def test_unchanged_verified_marker_transcript_passes_without_new_changes():
    """The pass-through must not count the unchanged marker as a new change:
    the no-progress guard compares rough token sizes and a phantom change
    would not satisfy it, so the commit site needs a real delta elsewhere."""
    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [{"role": "tool", "tool_call_id": "call-real", "content": marker}]
    candidate = copy.deepcopy(original)

    changes = validate_sanitation_candidate(
        original,
        candidate,
        externalized_payload_loader=lambda _ref: _tool_payload(secret),
    )

    assert changes == SanitationChanges()


def test_unchanged_verified_marker_passes_through_even_without_sidecar():
    """The pass-through must not require the sidecar: an unchanged marker was
    verified when it was committed, and re-verification would permanently
    block sanitation once the sidecar becomes unreadable."""
    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [{"role": "tool", "tool_call_id": "call-real", "content": marker}]
    candidate = copy.deepcopy(original)

    for loader in (lambda _ref: None, lambda _ref: (_ for _ in ()).throw(RuntimeError("gone"))):
        assert (
            validate_sanitation_candidate(
                original,
                candidate,
                externalized_payload_loader=loader,
            )
            == SanitationChanges()
        )


def test_unchanged_verified_payload_marker_transcript_still_validates():
    secret = "sensitive output"
    marker = _payload_marker(secret, "assistant")
    original = [
        {"role": "assistant", "content": marker},
        {"role": "user", "content": "token: hunter2"},
    ]
    candidate = copy.deepcopy(original)
    candidate[1]["content"] = "token: " + _placeholder("bearer_token", "hunter2")

    changes = validate_sanitation_candidate(
        original,
        candidate,
        externalized_payload_loader=lambda _ref: _payload_payload(secret, "assistant"),
    )

    assert changes is not None
    assert changes.placeholders == 1


# ---- bug 4018592967: nested markers must verify against the sidecar ----


def test_nested_marker_requires_verifiable_sidecar():
    """A nested marker replacing a secret must fail closed when the sidecar is
    missing, malformed, unreadable, or no loader is wired."""
    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [
        {
            "role": "tool",
            "tool_call_id": "call-real",
            "content": [{"type": "text", "text": secret}],
        }
    ]
    candidate = copy.deepcopy(original)
    candidate[0]["content"][0]["text"] = marker

    assert validate_sanitation_candidate(original, candidate) is None
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: None,
        )
        is None
    )
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: {"kind": "tool_result"},
        )
        is None
    )
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: (_ for _ in ()).throw(
                RuntimeError("cannot read sidecar")
            ),
        )
        is None
    )
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: _tool_payload(secret),
        )
        is not None
    )


def test_nested_marker_requires_verifiable_sidecar_for_payload_role():
    secret = "sensitive output"
    marker = _payload_marker(secret, "user")
    original = [
        {"role": "user", "content": [{"type": "text", "text": secret}]},
    ]
    candidate = copy.deepcopy(original)
    candidate[0]["content"][0]["text"] = marker

    assert validate_sanitation_candidate(original, candidate) is None
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: None,
        )
        is None
    )
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: _payload_payload(secret, "user"),
        )
        is not None
    )


def test_nested_marker_sidecar_identity_must_match_message():
    secret = "sensitive output"
    original = [
        {
            "role": "tool",
            "tool_call_id": "call-other",
            "content": [{"type": "text", "text": secret}],
        }
    ]

    # A marker minted for another tool call must not validate against this
    # message even when a sidecar for this message exists.
    foreign = copy.deepcopy(original)
    foreign[0]["content"][0]["text"] = _tool_marker(secret, tool_call_id="call-real")
    assert (
        validate_sanitation_candidate(
            original,
            foreign,
            externalized_payload_loader=lambda _ref: _tool_payload(secret, "call-other"),
        )
        is None
    )

    # A marker for this message verifies against its matching sidecar.
    candidate = copy.deepcopy(original)
    candidate[0]["content"][0]["text"] = _tool_marker(secret, tool_call_id="call-other")
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: _tool_payload(secret, "call-other"),
        )
        is not None
    )


def test_nested_marker_unchanged_passes_through_without_sidecar():
    """An unchanged nested marker (already committed) must not block a second
    sanitation pass, mirroring the top-level pass-through."""
    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [
        {
            "role": "tool",
            "tool_call_id": "call-real",
            "content": [{"type": "text", "text": marker}],
        },
        {"role": "user", "content": "password= hunter2"},
    ]
    candidate = copy.deepcopy(original)
    candidate[1]["content"] = "password= " + _placeholder("password", "hunter2")

    assert (
        validate_sanitation_candidate(original, candidate)
        is not None
    )
    assert (
        validate_sanitation_candidate(
            original,
            candidate,
            externalized_payload_loader=lambda _ref: None,
        )
        is not None
    )


def test_nested_marker_fresh_externalization_keeps_marker_and_counts_change():
    """A verified nested externalization is accepted without expanding: the
    candidate keeps the marker (the payload lives in the sidecar) and the
    change is counted through the string externalization contract."""
    import agent.conversation_sanitation as sanitation

    secret = "sensitive output"
    marker = _tool_marker(secret)
    original = [
        {
            "role": "tool",
            "tool_call_id": "call-real",
            "content": [{"type": "text", "text": secret}],
        }
    ]
    candidate = copy.deepcopy(original)
    candidate[0]["content"][0]["text"] = marker

    original_view, candidate_view, verified, failed = sanitation._validation_views(
        copy.deepcopy(original),
        copy.deepcopy(candidate),
        lambda _ref: _tool_payload(secret),
    )

    assert failed is False
    assert verified == 0
    assert candidate_view[0]["content"][0]["text"] == marker

    changes = validate_sanitation_candidate(
        original,
        candidate,
        externalized_payload_loader=lambda _ref: _tool_payload(secret),
    )

    assert changes is not None
    assert changes.changed_fields == 1
    assert changes.placeholders == 1


# ---- E2E: real DB + file-backed sidecar ----


class _FileSidecarEngine:
    """Minimal context-engine double whose sidecars live on disk."""

    name = "fixture-file-sidecar-engine"
    _last_compress_aborted = False
    _last_summary_error = None
    _last_compression_made_progress = True
    _last_summary_fallback_used = False
    _last_feasibility_skip = False
    compression_count = 1
    last_compression_rough_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    awaiting_real_usage_after_compression = False
    emit_automatic_compaction_status = False

    def __init__(self, sidecar_dir: str):
        self.sidecar_dir = sidecar_dir
        self.candidate: list[dict] = []
        self.current_operation = "sanitize"
        self.calls = 0

    def prepare_compression_operation(
        self, messages, *, session_id=None, attempt_generation=None
    ):
        if self.current_operation != "sanitize":
            return None
        return "sanitize", object()

    def compress(self, _messages, **kwargs):
        self.calls += 1
        return copy.deepcopy(self.candidate), kwargs["operation_claim"]

    def load_externalized_payload_sidecar(self, ref: str):
        with open(os.path.join(self.sidecar_dir, ref), "r", encoding="utf-8") as fh:
            return json.load(fh)


def _write_sidecar(sidecar_dir: str, ref: str, payload: dict[str, Any]) -> None:
    os.makedirs(sidecar_dir, exist_ok=True)
    with open(os.path.join(sidecar_dir, ref), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _make_agent(tmp_path, messages):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id="externalization-two-pass",
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    agent._ensure_db_session()
    agent._flush_messages_to_session_db(messages, [])
    return agent, db


def _without_persistence_markers(messages):
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"_db_persisted", "_row_id", "timestamp"}
        }
        for message in messages
    ]


def test_two_pass_externalization_commits_with_real_db_and_file_sidecar(tmp_path):
    """E2E: pass 1 externalizes a tool output behind a file-backed sidecar and
    commits; pass 2 redacts another secret on the now marker-bearing
    transcript and must not be blocked by the unchanged marker."""
    import agent.conversation_compression as compression

    secret = "tool-sensitive output"
    sidecar_dir = str(tmp_path / "externalized")
    digest = hashlib.sha256(secret.encode()).hexdigest()[:12]
    ref = f"20260915_call-two-pass_{digest}_abc123.json"
    marker = (
        "[Externalized tool output: tool_call_id=call-two-pass; "
        f"chars={len(secret)}; bytes={len(secret.encode())}; "
        f"ref={ref}]"
    )
    _write_sidecar(
        sidecar_dir,
        ref,
        {
            "kind": "tool_result",
            "tool_call_id": "call-two-pass",
            "content": secret,
            "content_chars": len(secret),
            "content_bytes": len(secret.encode()),
        },
    )

    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "please inspect the payload"},
        {
            "role": "assistant",
            "content": "inspecting",
            "tool_calls": [
                {
                    "id": "call-two-pass",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-two-pass", "content": secret},
        {"role": "user", "content": "api_key= hunter2secret"},
    ]
    agent, db = _make_agent(tmp_path, messages)

    candidate = copy.deepcopy(messages)
    candidate[3]["content"] = marker
    engine = _FileSidecarEngine(sidecar_dir)
    engine.candidate = candidate
    agent.context_compressor = engine

    returned, _ = compression.compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100_000,
    )

    assert engine.calls == 1
    assert agent._last_compaction_in_place is True
    durable = db.get_messages_as_conversation(agent.session_id)
    assert durable[3]["content"] == marker
    assert returned[3]["content"] == marker
    assert secret not in json.dumps(
        _without_persistence_markers(durable), default=str
    )

    # ---- pass 2: the marker-bearing transcript is still sanitatable ----
    candidate2 = copy.deepcopy(returned)
    candidate2[4]["content"] = "api_key= " + _placeholder("api_key", "hunter2secret")
    engine2 = _FileSidecarEngine(sidecar_dir)
    engine2.candidate = candidate2
    agent.context_compressor = engine2

    returned2, _ = compression.compress_context(
        agent,
        returned,
        "system",
        approx_tokens=100_000,
    )

    assert engine2.calls == 1
    assert agent._last_compaction_in_place is True
    assert returned2[3]["content"] == marker
    assert returned2[4]["content"] == "api_key= " + _placeholder(
        "api_key", "hunter2secret"
    )
    durable_after = db.get_messages_as_conversation(agent.session_id)
    assert durable_after[3]["content"] == marker
    durable_values = json.dumps(
        _without_persistence_markers(durable_after), default=str
    )
    assert secret not in durable_values
    assert "hunter2secret" not in durable_values


def test_nested_marker_missing_sidecar_e2e_fails_closed(tmp_path):
    """E2E: a nested marker whose sidecar file is gone fails the commit
    closed — the last recoverable copy (the secret) stays durable."""
    import agent.conversation_compression as compression

    secret = "last-copy-sensitive output"
    sidecar_dir = str(tmp_path / "externalized")
    os.makedirs(sidecar_dir, exist_ok=True)  # sidecar dir exists, file missing
    digest = hashlib.sha256(secret.encode()).hexdigest()[:12]
    ref = f"20260915_call-lost_{digest}_abc123.json"
    marker = (
        "[Externalized tool output: tool_call_id=call-lost; "
        f"chars={len(secret)}; bytes={len(secret.encode())}; "
        f"ref={ref}]"
    )

    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "please inspect"},
        {
            "role": "assistant",
            "content": "inspecting",
            "tool_calls": [
                {
                    "id": "call-lost",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-lost",
            "content": [{"type": "text", "text": secret}],
        },
        {"role": "user", "content": "summary request"},
    ]
    agent, db = _make_agent(tmp_path, messages)

    candidate = copy.deepcopy(messages)
    candidate[3]["content"][0]["text"] = marker
    engine = _FileSidecarEngine(sidecar_dir)
    engine.candidate = candidate
    agent.context_compressor = engine

    durable_before = db.get_messages_as_conversation(agent.session_id)
    returned, _ = compression.compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100_000,
    )

    assert returned is messages
    assert agent._last_compaction_in_place is False
    assert _without_persistence_markers(
        db.get_messages_as_conversation(agent.session_id)
    ) == _without_persistence_markers(durable_before)
    assert secret in json.dumps(
        _without_persistence_markers(durable_before), default=str
    )
