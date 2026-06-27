from agent.prompt_cache_diagnostics import (
    build_prompt_cache_diagnostics,
    format_prompt_cache_diagnostics,
    stable_hash,
)


def test_prompt_cache_diagnostics_hashes_are_stable_for_equivalent_payloads():
    messages_a = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "current"},
    ]
    messages_b = [
        {"content": "stable system", "role": "system"},
        {"content": "first", "role": "user"},
        {"content": "second", "role": "assistant"},
        {"content": "current", "role": "user"},
    ]
    tools = [{"function": {"name": "demo", "parameters": {"type": "object"}}, "type": "function"}]

    first = build_prompt_cache_diagnostics(messages_a, tools=tools, provider="anthropic", model="claude")
    second = build_prompt_cache_diagnostics(messages_b, tools=tools, provider="anthropic", model="claude")

    assert first.system_prompt_hash == second.system_prompt_hash
    assert first.tools_hash == second.tools_hash
    assert first.message_prefix_hash == second.message_prefix_hash
    assert first.system_prompt_tokens_estimate > 0
    assert first.tools_tokens_estimate > 0


def test_prompt_cache_diagnostics_reports_only_possible_bust_causes():
    previous = build_prompt_cache_diagnostics(
        [{"role": "system", "content": "stable"}, {"role": "user", "content": "one"}],
        tools=[{"name": "old"}],
        provider="anthropic",
        model="claude",
        session_id="s1",
    )

    current = build_prompt_cache_diagnostics(
        [{"role": "system", "content": "stable"}, {"role": "user", "content": "two"}],
        tools=[{"name": "new"}],
        provider="anthropic",
        model="claude",
        session_id="s1",
        previous=previous,
    )

    assert "tool schema hash changed" in current.possible_bust_causes
    assert "system prompt rebuilt or changed" not in current.possible_bust_causes
    rendered = format_prompt_cache_diagnostics(current)
    assert "possible bust causes" in rendered
    assert "tools=" in rendered


def test_stable_hash_omits_obvious_secret_attrs():
    class Obj:
        def __init__(self):
            self.name = "tool"
            self.api_key = "redacted"
            self.api_token = "redacted"
            self.auth_header = "redacted"

    assert stable_hash(Obj()) == stable_hash({"name": "tool"})
