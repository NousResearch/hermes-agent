from pathlib import Path


def test_gateway_clarify_callback_assigned_after_definition():
    """Regression for Telegram-wide UnboundLocalError on every message.

    A nested ``def _clarify_callback_sync`` inside ``run_sync`` makes that name
    local to the function. If ``agent.clarify_callback = _clarify_callback_sync``
    appears before the nested def, Python raises UnboundLocalError before any
    agent call runs.
    """

    source = Path("gateway/run.py").read_text()
    run_sync_start = source.index("        def run_sync():")
    bridge_comment = source.index("            # Clarify callback: present a clarify prompt", run_sync_start)
    first_assignment = source.index("agent.clarify_callback = _clarify_callback_sync", run_sync_start)
    nested_def = source.index("            def _clarify_callback_sync(question: str, choices) -> str:", run_sync_start)

    assert nested_def < first_assignment
    assert bridge_comment < nested_def
