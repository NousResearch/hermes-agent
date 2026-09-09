"""Tests for the /model picker fuzzy filter (C-01).

The filter narrows a provider's concrete model list as the user types, but
selection must still resolve to exactly ONE real model — never an ambiguous
or fuzzy resolution (the "claude → claude-sonnet-3" footgun). These pin the
index-preserving contract of ``_filter_model_picker_entries``.
"""

from cli import HermesCLI


MODELS = [
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.5",
    "x-ai/grok-4.6",
    "deepseek/deepseek-v4-flash",
]


def test_empty_query_returns_all_with_original_indices():
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "")
    assert pairs == list(enumerate(MODELS))


def test_filter_narrows_and_preserves_original_index():
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "grok")
    # Only the grok row matches, and it carries its ORIGINAL index (4) so the
    # selection handler resolves the exact concrete model.
    assert pairs == [(4, "x-ai/grok-4.6")]
    idx, label = pairs[0]
    assert MODELS[idx] == label  # index maps back to the real entry


def test_subsequence_match_case_insensitive():
    # "cs46" is a subsequence of "anthropic/claude-sonnet-4.6"
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "CS46")
    assert ("anthropic/claude-sonnet-4.6") in [e for _i, e in pairs]


def test_no_match_returns_empty():
    assert HermesCLI._filter_model_picker_entries(MODELS, "zzzznope") == []


def test_filter_does_not_reorder_or_pick_a_default():
    # Typing "claude" narrows to the three claude rows in ORIGINAL order — it
    # never silently resolves to one (the anti-ambiguity guarantee). The user
    # still explicitly selects among the concrete matches.
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "claude")
    labels = [e for _i, e in pairs]
    assert labels == [
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
    ]
    # indices are the originals, in order
    assert [i for i, _e in pairs] == [0, 1, 2]


def test_whitespace_query_is_treated_as_empty():
    assert HermesCLI._filter_model_picker_entries(MODELS, "   ") == list(enumerate(MODELS))


class TestModelPickerFallbackNote:
    """The picker's "Current:" row names the CONFIGURED model, which fallback
    activation never rewrites — so with a primary that fails every turn the
    picker advertised a model that was answering nothing while the status bar
    showed the hop. The note is the picker's half of telling the truth.
    """

    @staticmethod
    def _cli(agent):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.agent = agent
        return cli_obj

    def test_note_names_the_live_model_when_a_fallback_is_active(self):
        from types import SimpleNamespace

        cli_obj = self._cli(SimpleNamespace(
            model="deepseek-v4-flash",
            provider="custom:litellm-direct",
            _provider_fallback_active=True,
        ))

        note = cli_obj._model_picker_fallback_note("gpt-6-astra", "openai-codex")

        assert note == (
            "⚠ fallback active — answering on deepseek-v4-flash via custom:litellm-direct"
        )

    def test_no_note_on_a_normal_turn(self):
        from types import SimpleNamespace

        cli_obj = self._cli(SimpleNamespace(
            model="gpt-6-astra", provider="openai-codex", _provider_fallback_active=False,
        ))

        assert cli_obj._model_picker_fallback_note("gpt-6-astra", "openai-codex") == ""

    def test_no_note_when_the_hop_matches_the_configured_model(self):
        """Primary and hop 1 can be the same model on different endpoints; the
        note would then say nothing the "Current:" row does not already say."""
        from types import SimpleNamespace

        cli_obj = self._cli(SimpleNamespace(
            model="deepseek-v4-flash",
            provider="custom:litellm-control",
            _provider_fallback_active=True,
        ))

        assert cli_obj._model_picker_fallback_note(
            "deepseek-v4-flash", "custom:litellm-control") == ""

    def test_no_agent_yet_is_not_an_error(self):
        assert self._cli(None)._model_picker_fallback_note("x", "y") == ""

    def test_one_turn_switch_produces_no_note(self):
        """``_fallback_activated`` alone is the `/model --once` restore plumbing, not a
        demotion — the note must not claim a fallback for a deliberate switch."""
        from types import SimpleNamespace

        cli_obj = self._cli(SimpleNamespace(
            model="glm-5", provider="opencode-go",
            _fallback_activated=True, _provider_fallback_active=False,
        ))

        assert cli_obj._model_picker_fallback_note("gpt-6-astra", "openai-codex") == ""
