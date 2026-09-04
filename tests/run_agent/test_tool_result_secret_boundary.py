"""PRD-279 integration coverage for durable/provider tool-result boundaries."""

import copy
import json
import os
from unittest.mock import patch


def _make_agent(session_db):
    from run_agent import AIAgent

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id="prd-279-test",
            skip_context_files=True,
            skip_memory=True,
        )


def _tool_exchange(secret):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_prd279",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_prd279",
            "tool_name": "terminal",
            "content": (
                f"state = running\nTHIRD_PARTY_TOKEN => {secret}\n"
                f"status=https://example.test/check?access_token={secret}&view=summary"
            ),
        },
    ]


def test_synthetic_secret_is_absent_from_stored_tool_session_record(tmp_path):
    from hermes_state import SessionDB

    secret = "syntheticOpaqueCredentialValue123456"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _make_agent(db)

        assert agent._flush_messages_to_session_db(_tool_exchange(secret), []) is True

        rows = db.get_messages("prd-279-test")
        stored_tool = next(row for row in rows if row["role"] == "tool")
        assert secret not in stored_tool["content"]
        assert "state = running" in stored_tool["content"]
        assert "view=summary" in stored_tool["content"]
    finally:
        db.close()


def test_synthetic_secret_is_absent_from_provider_request_payload(tmp_path):
    from hermes_state import SessionDB

    secret = "syntheticOpaqueCredentialValue123456"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _make_agent(db)
        api_messages = agent._sanitize_api_messages(copy.deepcopy(_tool_exchange(secret)))
        payload = agent._build_api_kwargs(api_messages, tools_for_api=[])

        serialized = json.dumps(payload, default=str)
        assert secret not in serialized
        assert "state = running" in serialized
        assert "view=summary" in serialized
    finally:
        db.close()