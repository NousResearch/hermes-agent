"""Resume-time orphan-recovery wording by interrupt provenance (#84207).

When an interrupted side-effecting tool survives into replay history, the
recovered note should phrase the cause precisely: a deliberate stop is the
user's own action, a dropped connection is a transport failure.  The
structured ``stop_kind`` stamped on the tool result at execution time
carries that provenance across the restart boundary.
"""

from agent.replay_cleanup import strip_interrupted_tool_tails


def _user(text):
    return {"role": "user", "content": text}


def _assistant_tc():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ],
    }


def _interrupted_tool(stop_kind=None):
    msg = {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "terminal",
        "content": "\n[Command interrupted]\nExit Code: 130",
    }
    if stop_kind is not None:
        msg["stop_kind"] = stop_kind
    return msg


def _recover(stop_kind=None):
    out = strip_interrupted_tool_tails([_user("run it"), _assistant_tc(), _interrupted_tool(stop_kind)])
    assert len(out) == 3, f"expected assistant+recovered tool to survive, got {out!r}"
    return out[-1]


def test_orphan_recovery_without_provenance_keeps_legacy_wording():
    recovered = _recover()
    assert recovered["effect_disposition"] == "unknown"
    assert "interrupted side-effecting tool may have executed" in recovered["content"]
    assert "you stopped" not in recovered["content"].lower()
    assert "connection dropped" not in recovered["content"].lower()


def test_orphan_recovery_user_stop_wording():
    recovered = _recover("user_stop")
    assert recovered["effect_disposition"] == "unknown"
    assert "you stopped" in recovered["content"].lower()
    # The side-effect UNKNOWN warning is preserved — a deliberate stop still
    # leaves the tool's effect unknown.
    assert "unknown" in recovered["content"].lower()


def test_orphan_recovery_client_disconnect_wording():
    recovered = _recover("client_disconnect")
    assert recovered["effect_disposition"] == "unknown"
    assert "connection dropped" in recovered["content"].lower()
    assert "unknown" in recovered["content"].lower()