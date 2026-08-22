"""Per-turn timestamp injection tests.

The per-turn clock stamp (``_current_turn_timestamp``) rides on the
ephemeral system prompt, appended AFTER the byte-stable cached prefix at
API-call time. One shared composer
(``agent.prompt_builder.compose_effective_system_tail``) serves all three
request-only system assemblies: the main API build, the failover sync, and
the max-iterations summary request. These tests pin that contract so a
future refactor cannot silently drop the stamp from one site.
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import (
    handle_max_iterations,
    rewrite_prompt_model_identity,
)
from agent.conversation_loop import _sync_failover_system_message
from agent.prompt_builder import compose_effective_system_tail

_TS = "Current time: Sunday 2026-08-02 14:42 BST"


def _agent_with_timestamp(prompt="SYS\nModel: gpt-5.4-mini\nProvider: openai-codex", ephemeral=None):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        ephemeral_system_prompt=ephemeral,
        _current_turn_timestamp=_TS,
    )


class TestComposeEffectiveSystemTail:
    """The shared composer: base prompt preserved byte-identically, stamp
    and ephemeral prompt appended after it, in that order."""

    def test_timestamp_and_ephemeral_after_base(self):
        agent = _agent_with_timestamp(ephemeral="EPHEMERAL")
        out = compose_effective_system_tail(agent, "SYS")
        assert out == "SYS\n\nCurrent time: Sunday 2026-08-02 14:42 BST\n\nEPHEMERAL"

    def test_timestamp_alone(self):
        agent = _agent_with_timestamp(ephemeral=None)
        assert compose_effective_system_tail(agent, "SYS") == "SYS\n\nCurrent time: Sunday 2026-08-02 14:42 BST"

    def test_ephemeral_without_timestamp_preserves_legacy_layout(self):
        agent = SimpleNamespace(
            _cached_system_prompt="SYS", ephemeral_system_prompt="EPHEMERAL"
        )
        assert compose_effective_system_tail(agent, "SYS") == "SYS\n\nEPHEMERAL"

    def test_neither_appends_nothing(self):
        agent = SimpleNamespace(_cached_system_prompt="SYS", ephemeral_system_prompt=None)
        assert compose_effective_system_tail(agent, "SYS") == "SYS"

    def test_base_prompt_untouched_verbatim(self):
        """The byte-stable cached prefix must never be mutated — the
        provider prefix cache depends on it."""
        agent = _agent_with_timestamp(ephemeral="EPHEMERAL")
        base = "PRECISE\nBYTES\nLINE"
        out = compose_effective_system_tail(agent, base)
        assert base in out
        assert out.startswith(base)

    def test_empty_base_with_stamp(self):
        agent = _agent_with_timestamp(ephemeral=None)
        assert compose_effective_system_tail(agent, "") == _TS


class TestTimestampThroughFailoverSync:
    """_sync_failover_system_message rewrites the in-flight system message
    after provider failover; the stamp must survive that rewrite."""

    def test_stamp_preserved_after_failover_rewrite(self):
        agent = _agent_with_timestamp()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        api_messages = [
            {"role": "system", "content": agent._cached_system_prompt},
            {"role": "user", "content": "what model are you?"},
        ]
        _sync_failover_system_message(agent, api_messages, agent._cached_system_prompt)
        content = api_messages[0]["content"]
        assert _TS in content
        assert "Model: gemma4:e2b-mlx" in content
        # rewrite_prompt_model_identity updates the cached prompt in place
        # (that is its contract); the stamp must ride the rewritten copy.
        assert "Model: gemma4:e2b-mlx" in agent._cached_system_prompt


class TestTimestampThroughMaxIterationsSummary:
    """handle_max_iterations builds its own system message for the forced
    summary request. It must carry the same per-turn stamp as the main
    loop, or the timestamp disappears exactly when the loop ends."""

    def test_summary_request_includes_timestamp(self):
        from unittest.mock import patch

        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        agent._current_turn_timestamp = _TS

        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "RAW-RESPONSE"

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        transport = SimpleNamespace(
            normalize_response=lambda _r: SimpleNamespace(content="SUMMARY")
        )

        messages = [{"role": "user", "content": "q1"}]
        with patch.object(
            agent, "_ensure_primary_openai_client", return_value=client
        ), patch.object(agent, "_get_transport", return_value=transport):
            out = handle_max_iterations(agent, messages, 5)

        assert out == "SUMMARY"
        system = [m for m in captured["messages"] if m.get("role") == "system"]
        assert system, "summary request must carry a system message"
        assert _TS in system[0]["content"]
        assert system[0]["content"].startswith("SYS")

    def test_summary_without_timestamp_still_works(self):
        """Older agent stubs (no _current_turn_timestamp attribute) must
        behave exactly as before — stamp degrades gracefully."""
        from unittest.mock import patch

        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        # No _current_turn_timestamp set.

        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "RAW-RESPONSE"

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        transport = SimpleNamespace(
            normalize_response=lambda _r: SimpleNamespace(content="SUMMARY")
        )

        with patch.object(
            agent, "_ensure_primary_openai_client", return_value=client
        ), patch.object(agent, "_get_transport", return_value=transport):
            out = handle_max_iterations(agent, [{"role": "user", "content": "q1"}], 5)

        assert out == "SUMMARY"
        system = [m for m in captured["messages"] if m.get("role") == "system"]
        assert system[0]["content"] == "SYS"
