import json
from pathlib import Path

from plugins.memory.memory_fragmentation import (
    MemoryFragmentationProvider,
    _load_memory_fragmentation_config,
    _save_memory_fragmentation_config,
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_load_and_save_config_round_trip(tmp_path):
    _save_memory_fragmentation_config(
        {
            "enabled": True,
            "max_recall_items": 3,
            "summary_budget_chars": 240,
            "min_turn_chars": 42,
        },
        str(tmp_path),
    )

    cfg = _load_memory_fragmentation_config(str(tmp_path))

    assert cfg["schema_version"] == "v1"
    assert cfg["enabled"] is True
    assert cfg["function"] == "key-summary-full-memory-fragmentation"
    assert cfg["canonical_key"] == "raw human-readable title"
    assert cfg["identity_policy"] == "raw_key_is_canonical; tokenizer_views_are_auxiliary"
    assert cfg["max_recall_items"] == 3
    assert cfg["summary_budget_chars"] == 240
    assert cfg["min_turn_chars"] == 42


def test_post_setup_creates_config_and_activates_provider(tmp_path, monkeypatch):
    saved_configs = []

    def fake_save_config(config):
        saved_configs.append(config)

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)
    provider = MemoryFragmentationProvider()
    config = {"memory": {"provider": ""}}

    provider.post_setup(str(tmp_path), config)

    assert config["memory"]["provider"] == "memory_fragmentation"
    assert saved_configs == [config]
    cfg = _load_memory_fragmentation_config(str(tmp_path))
    assert cfg["enabled"] is True
    assert (tmp_path / "memory_fragmentation" / "config.json").exists()


def test_sync_turn_writes_key_summary_full_fragment(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="user-1",
        agent_identity="coder",
    )

    provider.sync_turn(
        "Please build a quant strategy with momentum and volatility signals.",
        "Completed the quant strategy work. Touched src/strategies/momentum.py and reports/performance.md. CAGR 24%, Sortino 2.1, max drawdown -9%.",
        session_id="session-1",
    )

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["session_id"] == "session-1"
    assert record["user_id"] == "user-1"
    assert record["memory_type"] == "conversation_round"
    assert record["status"] == "active"
    assert record["raw_key"]
    assert not record["raw_key"].startswith("mem_")
    assert "quant" in record["raw_key"].lower()
    assert record["summary_short"]
    assert record["summary_medium"]
    assert record["full_content_ref"] == f"full/{record['record_id']}.md"
    assert not Path(record["full_content_ref"]).is_absolute()
    assert record["source_spans"] == ["user", "assistant"]
    assert "quant" in record["tags"]
    assert any("src/strategies/momentum.py" in artifact for artifact in record["artifacts"])

    full_path = tmp_path / "memory_fragmentation" / record["full_content_ref"]
    assert full_path.exists()
    full_text = full_path.read_text(encoding="utf-8")
    assert "[role: user]" in full_text
    assert "[role: assistant]" in full_text
    assert "momentum and volatility" in full_text


def test_sync_turn_skips_trivial_exchanges(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn("ok", "sure", session_id="session-1")

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    assert records == []


def test_sync_turn_masks_sensitive_text_in_record_and_full_content(tmp_path):
    secret = "sk-testSECRETSECRET123456"
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn(
        f"Please remember api_key={secret} for the demo service.",
        "I will not store the raw key, but I configured the demo service notes.",
        session_id="session-1",
    )

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    assert len(records) == 1
    serialized = json.dumps(records[0], sort_keys=True)
    assert secret not in serialized
    assert records[0]["sensitivity_labels"]

    full_text = (tmp_path / "memory_fragmentation" / records[0]["full_content_ref"]).read_text(encoding="utf-8")
    assert secret not in full_text
    assert "[REDACTED]" in full_text


def test_prefetch_returns_only_key_summary_ladder_by_default(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Develop a quant strategy with RSI and momentum filters.",
        "Finished the strategy. Touched src/strategies/rsi_momentum.py. Performance: CAGR 18%, Sortino 1.7, max drawdown -8%.",
        session_id="session-1",
    )

    context = provider.prefetch("What did we do for the quant strategy?", session_id="session-1")

    assert "Memory Fragmentation Context" in context
    assert "Injected level: summary" in context
    assert "Raw key:" in context
    assert "Summary:" in context
    assert "Record ID:" in context
    assert "Full content ref:" not in context
    assert str(tmp_path) not in context
    assert "[role: user]" not in context
    assert "[role: assistant]" not in context


def test_prefetch_can_expand_to_full_for_exact_detail_requests(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Develop a quant strategy with RSI and momentum filters.",
        "Finished the strategy. Touched src/strategies/rsi_momentum.py. Performance: CAGR 18%, Sortino 1.7, max drawdown -8%.",
        session_id="session-1",
    )

    context = provider.prefetch("Which files changed in the quant strategy work?", session_id="session-1")

    assert "Injected level: full" in context
    assert "src/strategies/rsi_momentum.py" in context
    assert "[role: assistant]" in context


def test_full_retrieval_preserves_exact_output_formatting(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Save exact output formatting for the migration report.",
        "Exact output:\nline one\n  indented line\nline three",
        session_id="session-1",
    )
    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")

    payload = _tool_payload(
        provider,
        "memory_fragmentation_get",
        {"record_id": records[0]["record_id"], "detail_level": "full"},
    )
    assert "\nExact output:\nline one\n  indented line\nline three\n" in payload["full_content"]

    context = provider.prefetch("exact output migration report", session_id="session-1")
    assert "    Exact output:" in context
    assert "      indented line" in context


def _tool_payload(provider: MemoryFragmentationProvider, tool_name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(tool_name, args))


def test_prefetch_filters_by_user_and_agent_scope_before_scoring(tmp_path):
    alice = MemoryFragmentationProvider()
    alice.initialize(
        "session-alice",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="alice",
        agent_identity="coder",
    )
    alice.sync_turn(
        "Develop a quant strategy with private alpha filters.",
        "Completed private alpha quant strategy notes in reports/alice_alpha.md.",
        session_id="session-alice",
    )
    assert "private alpha" in alice.prefetch("quant private alpha", session_id="session-alice")

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    record_id = records[0]["record_id"]

    bob = MemoryFragmentationProvider()
    bob.initialize(
        "session-bob",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="bob",
        agent_identity="coder",
    )
    assert bob.prefetch("quant private alpha", session_id="session-bob") == ""
    assert "error" in _tool_payload(
        bob,
        "memory_fragmentation_get",
        {"record_id": record_id, "detail_level": "full"},
    )

    other_agent = MemoryFragmentationProvider()
    other_agent.initialize(
        "session-alice-researcher",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="alice",
        agent_identity="researcher",
    )
    assert other_agent.prefetch("quant private alpha", session_id="session-alice-researcher") == ""
    assert "error" in _tool_payload(
        other_agent,
        "memory_fragmentation_get",
        {"record_id": record_id, "detail_level": "full"},
    )


def test_prefetch_and_get_filter_by_gateway_conversation_scope(tmp_path):
    chat_a = MemoryFragmentationProvider()
    chat_a.initialize(
        "session-chat-a",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="alice",
        agent_identity="coder",
        chat_id="chat-a",
        chat_type="private",
        thread_id="thread-1",
        gateway_session_key="telegram:chat-a:thread-1",
    )
    chat_a.sync_turn(
        "Develop a quant strategy with chat-specific private alpha filters.",
        "Completed chat-specific alpha notes in reports/chat_a_alpha.md.",
        session_id="session-chat-a",
    )
    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    record_id = records[0]["record_id"]
    assert records[0]["conversation_scope"] == "gateway:telegram:chat-a:thread-1"
    assert records[0]["chat_id"] == "chat-a"
    assert "chat-specific alpha" in chat_a.prefetch("quant chat-specific alpha")

    chat_b = MemoryFragmentationProvider()
    chat_b.initialize(
        "session-chat-b",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="alice",
        agent_identity="coder",
        chat_id="chat-b",
        chat_type="private",
        thread_id="thread-2",
        gateway_session_key="telegram:chat-b:thread-2",
    )

    assert chat_b.prefetch("quant chat-specific alpha") == ""
    assert "error" in _tool_payload(
        chat_b,
        "memory_fragmentation_get",
        {"record_id": record_id, "detail_level": "full"},
    )

    local_cli = MemoryFragmentationProvider()
    local_cli.initialize(
        "session-cli",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="alice",
        agent_identity="coder",
    )
    assert local_cli.prefetch("quant chat-specific alpha") == ""


def test_prefetch_unrelated_query_returns_empty_even_for_important_records(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Please remember this decision and preference for the dashboard project.",
        "Implemented the decision and wrote notes to reports/dashboard_decision.md.",
        session_id="session-1",
    )

    assert provider.prefetch("weather paris forecast tomorrow", session_id="session-1") == ""


def test_broad_details_query_stays_summary_only(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Develop a quant strategy with RSI and momentum filters.",
        "Finished the strategy. Touched src/strategies/rsi_momentum.py.",
        session_id="session-1",
    )

    context = provider.prefetch("Show me details about the quant strategy", session_id="session-1")

    assert "Injected level: summary" in context
    assert "Injected level: full" not in context
    assert "[role: user]" not in context
    assert "[role: assistant]" not in context


def test_search_and_get_shape_payloads_by_detail_level(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli", user_id="user-1")
    provider.sync_turn(
        "Develop a quant strategy with RSI and momentum filters.",
        "Finished the strategy. Touched src/strategies/rsi_momentum.py.",
        session_id="session-1",
    )

    key_payload = _tool_payload(
        provider,
        "memory_fragmentation_search",
        {"query": "quant strategy", "detail_level": "key"},
    )
    key_record = key_payload["results"][0]
    assert set(key_record) == {"record_id", "raw_key"}

    summary_payload = _tool_payload(
        provider,
        "memory_fragmentation_search",
        {"query": "quant strategy", "detail_level": "summary"},
    )
    summary_record = summary_payload["results"][0]
    assert "summary_short" in summary_record
    assert "summary_medium" in summary_record
    assert "tags" in summary_record
    assert "artifacts" in summary_record
    assert "user_id" not in summary_record
    assert "full_content_ref" not in summary_record

    record_id = summary_record["record_id"]
    full_payload = _tool_payload(
        provider,
        "memory_fragmentation_get",
        {"record_id": record_id, "detail_level": "full"},
    )
    assert "full_content" in full_payload
    assert "[role: assistant]" in full_payload["full_content"]
    assert "full_content_ref" not in full_payload
    assert "user_id" not in full_payload


def test_sensitive_bearer_password_and_jwt_are_redacted_and_excluded_from_recall(tmp_path):
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturepart"
    password = "SuperSecretPassword123"
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn(
        f"Configure demo service with Authorization: {bearer}; jwt={jwt}; my password is {password}.",
        "Stored safe demo notes without raw credentials.",
        session_id="session-1",
    )

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    assert len(records) == 1
    serialized = json.dumps(records[0], sort_keys=True)
    full_text = (tmp_path / "memory_fragmentation" / records[0]["full_content_ref"]).read_text(encoding="utf-8")
    for secret in (bearer, jwt, password):
        assert secret not in serialized
        assert secret not in full_text
    assert records[0]["sensitivity_labels"]
    assert "[REDACTED]" in serialized
    assert provider.prefetch("demo service credentials", session_id="session-1") == ""
    assert "error" in _tool_payload(
        provider,
        "memory_fragmentation_get",
        {"record_id": records[0]["record_id"], "detail_level": "full"},
    )


def test_tampered_full_content_ref_cannot_escape_provider_storage(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Develop a quant strategy with escape-test filters.",
        "Finished the strategy with reports/escape_safe.md.",
        session_id="session-1",
    )
    records_path = tmp_path / "memory_fragmentation" / "fragments.jsonl"
    records = _read_jsonl(records_path)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("OUTSIDE-SECRET-SHOULD-NOT-BE-READ", encoding="utf-8")
    records[0]["full_content_ref"] = str(outside)
    records_path.write_text(json.dumps(records[0], ensure_ascii=False) + "\n", encoding="utf-8")

    payload = _tool_payload(
        provider,
        "memory_fragmentation_get",
        {"record_id": records[0]["record_id"], "detail_level": "full"},
    )

    assert "OUTSIDE-SECRET-SHOULD-NOT-BE-READ" not in payload.get("full_content", "")
    assert "escape-test filters" in payload.get("full_content", "")


def test_session_switch_updates_cached_session_for_full_source(tmp_path):
    provider = MemoryFragmentationProvider()
    provider.initialize("old-session", hermes_home=str(tmp_path), platform="cli")

    provider.on_session_switch("new-session", parent_session_id="old-session", reset=True)
    provider.sync_turn(
        "Develop a quant strategy after session switch.",
        "Finished switched-session strategy notes.",
    )

    records = _read_jsonl(tmp_path / "memory_fragmentation" / "fragments.jsonl")
    assert records[0]["session_id"] == "new-session"
    full_text = (tmp_path / "memory_fragmentation" / records[0]["full_content_ref"]).read_text(encoding="utf-8")
    assert "Session: new-session" in full_text
