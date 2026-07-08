"""ACP refusal-marker surfacing in the in-harness delegate_task -> external ACP path
(2026-07-07).

When the harness delegates to an external ACP agent (e.g. Claude Code) via
delegate_task(acp_command=...), the ACP session/prompt result carries a stopReason.
"refusal" means the delegated agent DECLINED on content policy — previously the
stopReason was discarded and the refusal text came back looking like an ordinary
answer. _apply_acp_refusal_marker prepends an unmistakable marker so the delegating
agent surfaces it.
"""

from agent.copilot_acp_client import _apply_acp_refusal_marker, _ACP_REFUSAL_MARKER


def test_refusal_stopreason_prepends_marker():
    out = _apply_acp_refusal_marker("(empty)", "refusal")
    assert out.startswith(_ACP_REFUSAL_MARKER)
    assert "stopReason=refusal" in out
    assert out.endswith("(empty)")


def test_refusal_case_insensitive_and_trimmed():
    assert _apply_acp_refusal_marker("x", " Refusal ").startswith(_ACP_REFUSAL_MARKER)


def test_normal_stopreasons_pass_through_untouched():
    for sr in ("end_turn", "max_tokens", "cancelled", "stop", "", None):
        assert _apply_acp_refusal_marker("real answer", sr) == "real answer"  # type: ignore[arg-type]


def test_marker_is_prefix_not_replacement():
    # The original text must be preserved (the refusal explanation is useful).
    body = "API Error: ...Usage Policy..."
    out = _apply_acp_refusal_marker(body, "refusal")
    assert body in out
    assert out != body
