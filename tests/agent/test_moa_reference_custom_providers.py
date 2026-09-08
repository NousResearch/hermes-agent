"""Regression test: MoA's reference fan-out must thread the agent's custom_providers into
context-length resolution.

``ContextCompressor._resolve_context_length`` had this exact bug (71516214c3, "thread
custom_providers into context-length resolution") — a per-model ``context_length`` override in
``custom_providers`` was invisible because ``get_model_context_length()`` was called without it,
so the hardcoded catalog default won instead. ``agent/moa_loop.py``'s
``_reference_context_length`` (used by ``_trim_messages_for_reference`` to fit an advisor request
into its own window, issue #60345) has the same call shape and the same gap: ``agent`` is already
threaded down to ``_run_references_parallel`` (for cache-policy and interrupt checks), but
``agent._custom_providers`` was never extracted or passed to the reference/trim chain below it.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent import moa_loop


class TestReferenceContextLengthCustomProviders:
    def test_reference_context_length_passes_custom_providers_to_resolver(self):
        captured = {}

        def fake_get_model_context_length(**kwargs):
            captured.update(kwargs)
            return 42

        custom_providers = [{"base_url": "https://custom.example.com/v1", "models": {}}]
        with patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_model_context_length):
            result = moa_loop._reference_context_length(
                slot={"model": "big-window-model", "provider": "custom"},
                runtime={"base_url": "https://custom.example.com/v1", "api_key": ""},
                cache=None,
                custom_providers=custom_providers,
            )

        assert result == 42
        assert captured.get("custom_providers") == custom_providers

    def test_reference_context_length_defaults_custom_providers_to_none(self):
        """Callers that never pass custom_providers (e.g. pre-existing tests) must not break."""
        captured = {}

        def fake_get_model_context_length(**kwargs):
            captured.update(kwargs)
            return 7

        with patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_model_context_length):
            result = moa_loop._reference_context_length(
                slot={"model": "m", "provider": "p"}, runtime={"base_url": "", "api_key": ""}, cache=None,
            )

        assert result == 7
        assert captured.get("custom_providers") is None


class TestTrimMessagesForReferenceCustomProviders:
    def test_custom_provider_context_override_avoids_needless_trim(self):
        """Without the override, the resolver would report a tiny catalog window and the advisor
        request gets truncated; with it (the real per-model context_length the user configured),
        the same request fits and is left untouched — reproducing issue #60345's failure mode for
        custom-provider reference models specifically."""
        slot = {"model": "big-window-model", "provider": "custom"}
        runtime = {"base_url": "https://custom.example.com/v1", "api_key": ""}
        # Several old turns plus a trailing question: fits a 999999-token window comfortably,
        # but a 100-token window forces the oldest turns to drop (there's more than the
        # always-kept "trailing user turn plus one preceding turn" to trim from).
        messages = [{"role": "system", "content": "advisory system prompt"}]
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"turn {i} " + ("word " * 500)})
        messages.append({"role": "user", "content": "final question " + ("word " * 500)})

        def fake_get_model_context_length(**kwargs):
            # 10_000 leaves only ~800 usable tokens after the default 8192 output reserve —
            # far below this fixture's ~4-5k estimated tokens, forcing a trim.
            return 999999 if kwargs.get("custom_providers") else 10_000

        with patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_model_context_length):
            trimmed_without_override = moa_loop._trim_messages_for_reference(
                [dict(m) for m in messages], slot, runtime, custom_providers=None,
            )
            trimmed_with_override = moa_loop._trim_messages_for_reference(
                [dict(m) for m in messages], slot, runtime,
                custom_providers=[{"base_url": runtime["base_url"], "models": {slot["model"]: {"context_length": 999999}}}],
            )

        assert trimmed_without_override != messages, "tiny catalog window should force a trim"
        assert trimmed_with_override == messages, "the real (large) custom-provider window must not be trimmed"


class TestRunReferencesParallelThreadsCustomProviders:
    def test_extracts_agent_custom_providers_and_passes_to_run_reference(self):
        custom_providers = [{"base_url": "https://custom.example.com/v1", "models": {"m": {"context_length": 999999}}}]
        agent = SimpleNamespace(_custom_providers=custom_providers, _cache_disabled=None, _cache_ttl=None)
        calls = []

        def fake_run_reference(slot, ref_messages, **kwargs):
            calls.append(kwargs)
            return (moa_loop._slot_label(slot), "text", None)

        with patch.object(moa_loop, "_run_reference", side_effect=fake_run_reference):
            moa_loop._run_references_parallel(
                [{"provider": "custom", "model": "m"}], [{"role": "user", "content": "hi"}], agent=agent,
            )

        assert len(calls) == 1
        assert calls[0].get("custom_providers") == custom_providers

    def test_missing_agent_passes_none_without_raising(self):
        calls = []

        def fake_run_reference(slot, ref_messages, **kwargs):
            calls.append(kwargs)
            return (moa_loop._slot_label(slot), "text", None)

        with patch.object(moa_loop, "_run_reference", side_effect=fake_run_reference):
            moa_loop._run_references_parallel(
                [{"provider": "custom", "model": "m"}], [{"role": "user", "content": "hi"}], agent=None,
            )

        assert len(calls) == 1
        assert calls[0].get("custom_providers") is None
