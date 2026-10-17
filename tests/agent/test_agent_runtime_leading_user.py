"""Leading-user-turn invariant on outbound API payloads (#131382).

``ensure_user_leads_api_messages`` (agent/agent_runtime_leading_user.py) is the
send-time guard called from ``assemble_api_request``: Qwen-derived chat templates
(LM Studio and other local OpenAI-compatible gateways) 400 with
"No user query found in messages." on any payload whose first conversational turn
is not role=user, and Anthropic rejects the same shape. Persisted repair stays
shrink-only (the re-anchor contract, fe21a4d2f0): the two guards are tested together
here so the division of labour stays pinned in one place.
"""

from run_agent import AIAgent

from agent.agent_runtime_leading_user import (
    _LEADING_USER_BRIDGE,
    ensure_user_leads_api_messages,
)


def _bare_agent():
    return AIAgent.__new__(AIAgent)


def test_repair_does_not_bridge_leading_assistant_in_history():
    """Repair is shrink-only (the re-anchor contract pins
    ``recorded_idx >= len(messages)`` after repair, fe21a4d2f0): it never
    INSERTS a row. A leading-assistant lineage is left as-is in persisted
    history; the send-time guard owns the leading-user invariant."""
    agent = _bare_agent()
    messages = [
        {"role": "system", "content": "SOUL"},
        {"role": "assistant", "content": "[PRIOR CONTEXT ...]",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "todo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
        {"role": "user", "content": "move forward"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert len(messages) == 4
    assert all(m.get("content") != _LEADING_USER_BRIDGE for m in messages)


def test_repair_leaves_lost_opening_user_row_history_untouched():
    """The #131382 shape: in-memory history lost its opening user row after a
    mid-chat model switch. Persisted repair must not grow the list; the
    send-time guard re-leads the payload without touching history."""
    agent = _bare_agent()
    messages = [
        {"role": "system", "content": "SOUL"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]

    assert AIAgent._repair_message_sequence(agent, messages) == 0
    assert messages == [
        {"role": "system", "content": "SOUL"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def test_ensure_user_leads_model_switch_marker_cases():
    """The report's trigger: the model_switch marker is a display-only user
    row (display_kind=model_switch). On the wire copy it leads with
    role=user, so no bridge; a payload that lost BOTH the first user row and
    the marker (the dumped #131382 payload opened system, assistant, ...) gets
    the bridge ahead of the leading assistant row."""
    marker_leads = [
        {"role": "system", "content": "SOUL"},
        {"role": "user", "content": "[System: The active model changed]",
         "display_kind": "model_switch"},
        {"role": "user", "content": "next question"},
    ]
    assert ensure_user_leads_api_messages(marker_leads) == 0

    dropped = [
        {"role": "system", "content": "SOUL"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "next question"},
    ]
    assert ensure_user_leads_api_messages(dropped) == 1
    assert dropped[1]["role"] == "user"
    assert dropped[1]["content"] == _LEADING_USER_BRIDGE


def test_ensure_user_leads_skips_session_meta_rows():
    """Hermes transcripts open with a session_meta row; the bridge must go
    after it, and a transcript already leading with user stays untouched."""
    meta_then_user = [
        {"role": "session_meta", "tools": []},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    assert ensure_user_leads_api_messages(meta_then_user) == 0

    meta_then_assistant = [
        {"role": "session_meta", "tools": []},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "next question"},
    ]
    assert ensure_user_leads_api_messages(meta_then_assistant) == 1
    assert meta_then_assistant[0]["role"] == "session_meta"
    assert meta_then_assistant[1]["content"] == _LEADING_USER_BRIDGE


def test_ensure_user_leads_leading_orphan_tool_gets_bridge():
    """A wire payload opening with an orphan tool result: the bridge goes
    ahead of it, preserving whatever follows."""
    api_messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "x", "content": "{}"},
        {"role": "user", "content": "q"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 1
    assert api_messages[1]["role"] == "user"
    assert api_messages[1]["content"] == _LEADING_USER_BRIDGE


def test_ensure_user_leads_no_bridge_without_any_user_turn():
    """A payload with no user turn at all (e.g. an ACP session holding a
    single assistant message) is not the resumed-lineage case — inserting
    a bridge would fabricate a user turn the session never had."""
    api_messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "greeting only"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 0
    assert [m["role"] for m in api_messages] == ["system", "assistant"]


def test_ensure_user_leads_noop_on_well_formed_history():
    api_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "y"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 0
    assert len(api_messages) == 3
    assert all(m.get("content") != _LEADING_USER_BRIDGE for m in api_messages)


def test_ensure_user_leads_idempotent():
    """Re-running the guard must not insert a second bridge."""
    api_messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "summary",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
        {"role": "user", "content": "go"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 1
    assert ensure_user_leads_api_messages(api_messages) == 0
    assert sum(1 for m in api_messages if m.get("content") == _LEADING_USER_BRIDGE) == 1


def test_ensure_user_leads_bridges_leading_assistant():
    api_messages = [
        {"role": "system", "content": "SOUL"},
        {"role": "assistant", "content": "[PRIOR CONTEXT ...]",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "todo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
        {"role": "user", "content": "move forward with all changes"},
    ]

    inserted = ensure_user_leads_api_messages(api_messages)

    assert inserted == 1
    assert api_messages[0]["role"] == "system"
    assert api_messages[1]["role"] == "user"            # bridge now leads
    assert api_messages[2]["role"] == "assistant"       # assistant->tool intact
    assert api_messages[2]["tool_calls"][0]["id"] == "t1"
    assert api_messages[3]["tool_call_id"] == "t1"      # pairing preserved


def test_ensure_user_leads_well_formed_payload_is_noop():
    api_messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 0
    assert len(api_messages) == 2


def test_ensure_user_leads_no_system_leading_assistant():
    api_messages = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "q"},
    ]
    assert ensure_user_leads_api_messages(api_messages) == 1
    assert api_messages[0]["role"] == "user"


def test_ensure_user_leads_system_only_and_empty_are_noops():
    system_only = [{"role": "system", "content": "a"}]
    assert ensure_user_leads_api_messages(system_only) == 0
    assert system_only == [{"role": "system", "content": "a"}]
    assert ensure_user_leads_api_messages([]) == 0
