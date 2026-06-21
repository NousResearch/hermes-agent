from gateway.orchestrator.redaction import redact_for, redact_text
from gateway.orchestrator.registry import AgentKind, AgentSpec


def test_redact_text_masks_common_secret_shapes_and_is_idempotent():
    raw = (
        "OPENAI_API_KEY=sk-1234567890abcdef "
        "github=ghp_abcdefghijklmnopqrstuvwxyz123456 "
        "aws=AKIA1234567890ABCDEF "
        "hex=0123456789abcdef0123456789abcdef "
        "password=supersecretvalue "
        "normal words stay"
    )

    redacted = redact_text(raw)

    assert "sk-123" not in redacted
    assert "ghp_" not in redacted
    assert "AKIA" not in redacted
    assert "0123456789abcdef0123456789abcdef" not in redacted
    assert "supersecretvalue" not in redacted
    assert "normal words stay" in redacted
    assert redact_text(redacted) == redacted


def test_redact_for_suppresses_sensitive_agent_output_entirely():
    spec = AgentSpec("ccm", AgentKind.SHELL_FUNCTION, secrets=True)

    assert redact_for(spec, "MINIMAX_API_KEY=abc output") == "<suppressed>"


def test_redact_for_non_sensitive_agent_redacts_but_preserves_context():
    spec = AgentSpec("ccd", AgentKind.SHELL_FUNCTION)

    redacted = redact_for(spec, "result ok token=abc123")

    assert "result ok" in redacted
    assert "abc123" not in redacted
