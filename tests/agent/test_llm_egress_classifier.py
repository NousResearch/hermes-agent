from __future__ import annotations

from agent.llm_egress_classifier import _typed_payload
from agent.llm_egress_firewall import LiteralSegment, SanitizedSegment


def test_typed_payload_classifies_protocol_literals_and_text() -> None:
    typed = _typed_payload(
        {"role": "user", "content": "hello"},
        (),
        {},
        sanitized_cap=128,
    )

    assert isinstance(typed["role"], LiteralSegment)
    assert typed["role"].text == "user"
    assert isinstance(typed["content"], SanitizedSegment)
    assert typed["content"].text == "hello"
