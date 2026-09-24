"""Auto-reset is a full conversation boundary: per-session caches must not
survive it (#58403)."""
from __future__ import annotations

from gateway import run as gateway_run


def test_auto_reset_cleanup_clears_last_resolved_model():
    """Regression test for #58403.

    Auto-reset is a full conversation boundary and now routes through the
    `_clear_conversation_scope` funnel. The funnel must clear
    `_last_resolved_model` — without it, the fresh auto-reset session could
    serve a model cached before the reset on a transient config-cache miss.
    Behavioral: exercises the real funnel instead of pinning source shape.
    """
    runner = object.__new__(gateway_run.GatewayRunner)
    key = "agent:main:telegram:dm:58403"
    runner._last_resolved_model = {key: "stale/model", "other": "keep/me"}
    runner._clear_conversation_scope(key, reason="auto_reset")
    assert key not in runner._last_resolved_model, (
        "the conversation-boundary funnel must pop the session's entry from "
        "`_last_resolved_model` (#58403) — auto-reset routes through it"
    )
    assert runner._last_resolved_model.get("other") == "keep/me"
