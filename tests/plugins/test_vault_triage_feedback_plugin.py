from __future__ import annotations

from plugins.vault_triage_feedback import register


class _DummyContext:
    def __init__(self):
        self.calls = []

    def register_command(self, name, handler, description="", args_hint=""):
        self.calls.append(
            {
                "name": name,
                "handler": handler,
                "description": description,
                "args_hint": args_hint,
            }
        )


def test_register_exposes_para_feedback_command():
    ctx = _DummyContext()
    register(ctx)

    assert len(ctx.calls) == 1
    call = ctx.calls[0]
    assert call["name"] == "para-feedback"
    assert "correct <entry_id> <target>" in call["args_hint"]
    assert callable(call["handler"])
