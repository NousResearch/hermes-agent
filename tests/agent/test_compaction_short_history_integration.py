"""Real agent, loopback model transport and durable resume for a short token-heavy history."""

import json

import pytest

from agent.model_metadata import estimate_request_tokens_rough
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.fakes.providers.chat_variants import CError, CText, FakeChatVariantServer, MODEL_ID


@pytest.mark.parametrize("protect_first", [0, 1, 3])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("limit_source", ["provider", "preflight"])
def test_actual_agent_compacts_then_recalls_after_database_reopen(tmp_path, monkeypatch, protect_first, in_place, limit_source):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    summaries = []
    rejected_sessions = []

    def respond(record):
        body = record["body"]
        messages = body["messages"]
        rendered = json.dumps(messages)
        tokens = estimate_request_tokens_rough(messages, tools=body.get("tools"))
        if str(messages[-1].get("content", "")).startswith("New current request remains verbatim."):
            rejected_sessions.append(resumed.session_id)
            if limit_source == "provider":
                return CError(400, f"maximum context length is 32768 tokens; input has {tokens} tokens",
                              "context_length_exceeded")
        if tokens > 65_536:
            return CError(400, "maximum context length exceeded", "context_length_exceeded")
        if "You are a summarization agent creating a context checkpoint" in rendered:
            summaries.append(body)
            return CText("## Facts\nQA_TOKEN=CEDAR_409.\n## Completed Actions\nThe earlier filler was acknowledged.",
                         prompt_tokens=tokens)
        if "Recall QA_TOKEN" in str(messages[-1].get("content")):
            return CText("CEDAR_409" if "QA_TOKEN=CEDAR_409" in rendered else "MISSING", prompt_tokens=tokens)
        return CText("SAVED", prompt_tokens=tokens)

    with FakeChatVariantServer(respond, auxiliary=respond) as server:
        compression = {"in_place": in_place}
        if protect_first != 3:
            compression["protect_first_n"] = protect_first
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": "custom", "default": MODEL_ID, "base_url": server.base_url,
                      "context_length": 65_536},
            "compression": compression,
            "auxiliary": {"title_generation": {"enabled": False}},
        }))
        path = home / "state.db"
        database = SessionDB(db_path=path)

        def open_agent(db, session):
            return AIAgent(
                model=MODEL_ID, provider="custom", api_mode="chat_completions", base_url=server.base_url, api_key="fixture-key",
                enabled_toolsets=[], quiet_mode=True, skip_memory=True, skip_context_files=True,
                skip_background_review=True, session_db=db, session_id=session, max_iterations=4,
            )

        agent = open_agent(database, "short-history")
        try:
            assert agent.context_compressor.protect_first_n == protect_first
            assert agent.context_compressor.context_length == 65_536
            filler = "ordinary timber measurement is seventeen units.\n" * 3500
            first_prompt = "Remember QA_TOKEN=CEDAR_409.\n" + filler
            first = agent.run_conversation(first_prompt, system_message="Answer concisely.")
            assert first["final_response"] == "SAVED"
            second_prompt = "Keep QA_TOKEN from the earlier turn.\n" + filler
            second = agent.run_conversation(second_prompt, system_message="Answer concisely.",
                                            conversation_history=first["messages"])
            assert second["final_response"] == "SAVED"
            assert summaries, "the actual preflight and auxiliary transport must reach summary generation"
            assert any("QA_TOKEN=CEDAR_409" in json.dumps(call["messages"]) for call in summaries)
            assert agent.context_compressor.compression_count >= 1
            assert any(row.get("content") == second_prompt for row in second["messages"])
            session = agent.session_id
            archived = database.get_messages(session, include_compacted=True, include_ancestors=True)
            assert any(row.get("content") == first_prompt for row in archived)
        finally:
            agent.close()
            database.close()

        reopened = SessionDB(db_path=path)
        resumed = open_agent(reopened, session)
        try:
            history = reopened.get_messages_as_conversation(session)
            result = resumed.run_conversation("Recall QA_TOKEN. Reply only with its value.",
                                              system_message="Answer concisely.", conversation_history=history)
            assert result["final_response"] == "CEDAR_409"
            prior_summaries = [row["content"] for row in reopened.get_messages(
                session, include_compacted=True, include_ancestors=True,
            ) if "## Facts\nQA_TOKEN=CEDAR_409." in str(row.get("content"))]
            assert prior_summaries
            # Exercise both the provider's proven smaller window and the existing definite
            # configured-window refusal. The latter sends no oversized provider request.
            giant = "New current request remains verbatim.\n" + filler * (3 if limit_source == "preflight" else 1)
            rejected = resumed.run_conversation(giant, system_message="Answer concisely.",
                                                conversation_history=result["messages"])
            assert rejected["failure_reason"] == "request_uncompressible"
            assert rejected["compression_deferred"] is True
            assert not rejected.get("compression_exhausted")
            # Non-in-place compression may legitimately fork before the rejection; the
            # provider's rejected session must then survive without an exhaustion reset.
            if limit_source == "provider":
                assert rejected_sessions
                assert resumed.session_id == rejected_sessions[-1]
                assert "provider's" in rejected["final_response"]
            else:
                assert not rejected_sessions
                assert "configured model context window" in rejected["final_response"]
            assert (resumed.session_id == session) is in_place
            persisted = reopened.get_messages(resumed.session_id, include_compacted=True, include_ancestors=True)
            assert any(row.get("content") == first_prompt for row in persisted)
            assert any(row.get("content") == giant for row in persisted)
            assert all(any(row.get("content") == summary for row in persisted) for summary in prior_summaries)
            assert any("QA_TOKEN=CEDAR_409" in str(row.get("content")) for row in rejected["messages"])
            assert all(request["host"].startswith("127.0.0.1:") for request in server.requests)
            assert all(request["body"]["model"] == MODEL_ID for request in server.requests)
        finally:
            resumed.close()
            reopened.close()


@pytest.mark.parametrize("protect_first", [0, 1, 3])
@pytest.mark.parametrize("status", [400, 413])
def test_actual_agent_preserves_an_impossible_current_request_without_session_reset(tmp_path, monkeypatch, protect_first, status):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    rejection = (CError(413, "request payload too large", "request_too_large") if status == 413
                 else CError(400, "maximum context length exceeded", "context_length_exceeded"))
    reject = lambda record: rejection
    with FakeChatVariantServer(reject, auxiliary=reject) as server:
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": "custom", "default": MODEL_ID, "base_url": server.base_url,
                      "context_length": 65_536},
            "compression": {"protect_first_n": protect_first},
            "auxiliary": {"title_generation": {"enabled": False}},
        }))
        path = home / "state.db"
        database = SessionDB(db_path=path)
        agent = AIAgent(
            model=MODEL_ID, provider="custom", api_mode="chat_completions", base_url=server.base_url,
            api_key="fixture-key", enabled_toolsets=[], quiet_mode=True, skip_memory=True,
            skip_context_files=True, skip_background_review=True, session_db=database,
            session_id="giant-current", max_iterations=4,
        )
        current = "Current request must remain verbatim.\n" + "ordinary timber measurement is seventeen units.\n" * 8000
        try:
            result = agent.run_conversation(current, system_message="Answer concisely.")
            assert result["failure_reason"] == "request_uncompressible"
            assert result["failure_retryable"] is False
            assert result["compression_deferred"] is True
            assert result["failed"] is False
            assert result["completed"] is False
            assert not result.get("compression_exhausted")
            assert agent.session_id == "giant-current"
            assert any(row.get("content") == current for row in result["messages"])
            assert "Shorten" in result["final_response"]
        finally:
            agent.close()
            database.close()
        reopened = SessionDB(db_path=path)
        try:
            assert any(row.get("content") == current for row in reopened.get_messages("giant-current"))
        finally:
            reopened.close()


def test_compaction_acknowledgement_remains_model_only_after_sqlite_reopen_and_export(tmp_path, monkeypatch):
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, MODEL_ONLY_DISPLAY_METADATA_KEY

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with FakeChatVariantServer(lambda record: CText("## Facts\nHistorical facts remain available."),
                               auxiliary=lambda record: CText("## Facts\nHistorical facts remain available.")) as server:
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": "custom", "default": MODEL_ID, "base_url": server.base_url,
                      "context_length": 65_536},
            "auxiliary": {"title_generation": {"enabled": False}},
        }))
        path = home / "state.db"
        database = SessionDB(db_path=path)
        agent = AIAgent(
            model=MODEL_ID, provider="custom", api_mode="chat_completions", base_url=server.base_url,
            api_key="fixture-key", enabled_toolsets=[], quiet_mode=True, skip_memory=True,
            skip_context_files=True, skip_background_review=True, session_db=database, session_id="acknowledgement",
        )
        try:
            messages = [{"role": "system", "content": "System prompt."}]
            for index in range(10):
                messages.append({"role": "user" if index % 2 == 0 else "assistant",
                                 "content": f"turn {index} " + "z" * (32_000 if index < 3 else 1200)})
            agent.context_compressor.tail_token_budget = 500
            result = agent.context_compressor.compress(messages, current_tokens=60_000)
            assert sum(bool(row.get(COMPRESSED_SUMMARY_METADATA_KEY)) for row in result) == 1
            acknowledgements = [row for row in result if (row.get("display_metadata") or {}).get(
                MODEL_ONLY_DISPLAY_METADATA_KEY)]
            assert len(acknowledgements) == 1
            acknowledgement = acknowledgements[0]["content"]
            assert acknowledgements[0]["role"] == "assistant"
            agent._persist_session(result, [])
        finally:
            agent.close()
            database.close()

    reopened = SessionDB(db_path=path)
    try:
        replay = reopened.get_messages_as_conversation("acknowledgement")
        assert any(row.get("content") == acknowledgement and (row.get("display_metadata") or {}).get(
            MODEL_ONLY_DISPLAY_METADATA_KEY) for row in replay)
        raw_export = reopened.export_session("acknowledgement")
        assert any(row.get("content") == acknowledgement and (row.get("display_metadata") or {}).get(
            MODEL_ONLY_DISPLAY_METADATA_KEY) for row in raw_export["messages"])
        display_export = reopened.export_session("acknowledgement", include_compacted=True)
        assert not any(row.get("content") == acknowledgement for row in display_export["messages"])
        assert any(row.get("content") == messages[-1]["content"] for row in display_export["messages"])
    finally:
        reopened.close()
