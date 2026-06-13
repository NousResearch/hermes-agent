from __future__ import annotations

import json

import pytest

from agent.context_packet import (
    ContextPacket,
    ContextPacketValidationError,
    parse_context_packet,
)


def _valid_packet_dict() -> dict:
    return {
        "role": "steve",
        "source": "bob",
        "goal": "Implement typed handoffs.",
        "non_goals": ["Do not push"],
        "safety": "No secrets or production mutations.",
        "allowed_actions": ["read files", "edit repo files"],
        "forbidden_actions": ["push", "deploy"],
        "context": ["Worktree: /tmp/hermes"],
        "evidence": ["Prior discovery found no ContextPacket model"],
        "verification": ["pytest tests/agent/test_context_packet.py -q"],
        "output_contract": "Report changed files, verification, risks.",
        "residual_risks": ["Not all handoff surfaces integrated"],
    }


def test_context_packet_validates_full_field_contract() -> None:
    packet = ContextPacket.from_dict(_valid_packet_dict())

    assert packet.role == "steve"
    assert packet.source == "bob"
    assert packet.goal == "Implement typed handoffs."
    assert packet.non_goals == ("Do not push",)
    assert packet.allowed_actions == ("read files", "edit repo files")
    assert packet.forbidden_actions == ("push", "deploy")
    assert packet.context == ("Worktree: /tmp/hermes",)
    assert packet.evidence == ("Prior discovery found no ContextPacket model",)
    assert packet.verification == ("pytest tests/agent/test_context_packet.py -q",)
    assert packet.output_contract == "Report changed files, verification, risks."
    assert packet.residual_risks == ("Not all handoff surfaces integrated",)


def test_context_packet_markdown_rendering_is_deterministic() -> None:
    packet = ContextPacket.from_dict(_valid_packet_dict())

    assert packet.render_markdown() == """# Context Packet

- role: steve
- source: bob
- goal: Implement typed handoffs.
- safety: No secrets or production mutations.
- output_contract: Report changed files, verification, risks.

## Non-goals
- Do not push

## Allowed actions
- read files
- edit repo files

## Forbidden actions
- push
- deploy

## Context
- Worktree: /tmp/hermes

## Evidence
- Prior discovery found no ContextPacket model

## Verification
- pytest tests/agent/test_context_packet.py -q

## Residual risks
- Not all handoff surfaces integrated"""


def test_context_packet_parses_json_and_fenced_block() -> None:
    raw_json = json.dumps(_valid_packet_dict())
    assert ContextPacket.from_json(raw_json).goal == "Implement typed handoffs."

    fenced = f"handoff follows\n```context-packet\n{raw_json}\n```\nthanks"
    parsed = parse_context_packet(fenced)
    assert parsed is not None
    assert parsed.source == "bob"


def test_context_packet_missing_required_fields_fail_clearly() -> None:
    data = _valid_packet_dict()
    data.pop("verification")

    with pytest.raises(ContextPacketValidationError) as exc:
        ContextPacket.from_dict(data)

    assert "missing required field: verification" in str(exc.value)


def test_context_packet_redacts_likely_secrets() -> None:
    data = _valid_packet_dict()
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    data["context"] = [f"OPENAI_API_KEY={secret}"]
    packet = ContextPacket.from_dict(data)

    rendered = packet.render_markdown()
    assert secret not in rendered
    assert "OPENAI_API_KEY=" in rendered


def test_plain_natural_language_is_not_forced_into_context_packet() -> None:
    assert parse_context_packet("Please fix the tests and report back.") is None
