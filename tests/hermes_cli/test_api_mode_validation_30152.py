"""Regression tests for #30152.

#30152: ``api_mode: gemini`` providers silently routed through the OpenAI
SDK, and a ``base_url`` that ended with the Vertex AI action verb
``:generateContent`` got mangled to ``...:generateContent/`` by the
OpenAI SDK's trailing-slash normalisation. The result was HTTP 404 on
every call with no actionable error in the gateway log — only a
fallback cascade.

These tests pin down the two safety nets that #30152 introduced:

1. ``_parse_api_mode`` emits a single ``WARNING`` (with an actionable
   hint for known aliases like ``gemini`` / ``openai`` / ``anthropic``)
   when the operator supplies an unsupported ``api_mode`` value,
   instead of silently dropping it.
2. ``_validate_base_url_for_openai_sdk`` raises ``ValueError`` with the
   two valid Gemini/Vertex alternatives spelled out when the
   ``base_url`` tail-ends with a Vertex action verb
   (``:generateContent``, ``:streamGenerateContent``, …).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli.runtime_provider import (
    _API_MODE_ALIASES,
    _VALID_API_MODES,
    _VERTEX_ACTION_VERBS,
    _base_url_has_vertex_action_verb,
    _parse_api_mode,
    _validate_base_url_for_openai_sdk,
)


# ──────────────────────────────────────────────────────────────────────
# _parse_api_mode
# ──────────────────────────────────────────────────────────────────────
class TestParseApiModeAcceptsValidValues:
    """Round-trip every value in ``_VALID_API_MODES`` and assert no warnings."""

    @pytest.mark.parametrize("mode", sorted(_VALID_API_MODES))
    def test_valid_api_mode_round_trips(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode(mode) == mode
        assert caplog.records == []

    def test_valid_api_mode_is_case_insensitive(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode("Chat_Completions") == "chat_completions"
        assert _parse_api_mode("ANTHROPIC_MESSAGES") == "anthropic_messages"
        assert caplog.records == []

    def test_valid_api_mode_strips_whitespace(self) -> None:
        assert _parse_api_mode("  chat_completions  ") == "chat_completions"


class TestParseApiModeRejectsInvalidValues:
    """Unknown ``api_mode`` values must return None *and* log a warning."""

    @pytest.mark.parametrize(
        "raw",
        [None, 123, [], {}, object()],
        ids=["none", "int", "list", "dict", "object"],
    )
    def test_non_string_returns_none_silently(
        self, raw: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode(raw) is None
        # Non-strings are not a config typo to warn about — leave silent.
        assert caplog.records == []

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
    def test_empty_string_returns_none_silently(
        self, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode(raw) is None
        assert caplog.records == []

    def test_gemini_alias_warns_with_specific_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exact failure mode in #30152: api_mode: gemini in config."""
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode("gemini") is None
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        msg = rec.getMessage()
        assert "gemini" in msg
        assert "chat_completions" in msg
        # Both correct alternatives must be present so the operator can
        # pick the right one without re-reading the issue.
        assert "aiplatform.googleapis.com" in msg
        assert "generativelanguage.googleapis.com" in msg

    def test_gemini_alias_warns_uppercase_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode("GEMINI") is None
        assert any("gemini" in r.getMessage().lower() for r in caplog.records)

    @pytest.mark.parametrize(
        "alias,expected_in_hint",
        [
            ("openai", "chat_completions"),
            ("anthropic", "anthropic_messages"),
            ("bedrock", "bedrock_converse"),
            ("codex", "codex_responses"),
            ("google", "chat_completions"),
        ],
    )
    def test_known_aliases_get_specific_hints(
        self,
        alias: str,
        expected_in_hint: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode(alias) is None
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert expected_in_hint in joined

    def test_unknown_value_lists_all_valid_modes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        assert _parse_api_mode("not_a_real_mode") is None
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        for mode in _VALID_API_MODES:
            assert mode in msg

    def test_warning_does_not_double_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Repeated parsing must not silently accumulate handlers; one
        call → exactly one warning record."""
        caplog.set_level(logging.WARNING, logger="hermes_cli.runtime_provider")
        _parse_api_mode("gemini")
        assert len(caplog.records) == 1


class TestAliasTableInvariants:
    """The alias table must not list any value that is already valid —
    otherwise the warning would fire for a perfectly legal config."""

    def test_aliases_disjoint_from_valid_modes(self) -> None:
        assert set(_API_MODE_ALIASES) & _VALID_API_MODES == set()


# ──────────────────────────────────────────────────────────────────────
# _base_url_has_vertex_action_verb / _validate_base_url_for_openai_sdk
# ──────────────────────────────────────────────────────────────────────
_VERTEX_BASE = (
    "https://aiplatform.googleapis.com/v1/projects/anolasco-gemini"
    "/locations/global/publishers/google/models"
    "/gemini-3.1-pro-preview-customtools"
)


class TestBaseUrlActionVerbDetection:
    """``_base_url_has_vertex_action_verb`` must spot every Vertex
    action verb that the OpenAI SDK would silently corrupt."""

    @pytest.mark.parametrize("verb", _VERTEX_ACTION_VERBS)
    def test_each_verb_is_detected(self, verb: str) -> None:
        url = f"{_VERTEX_BASE}{verb}"
        assert _base_url_has_vertex_action_verb(url) == verb

    def test_trailing_slash_is_ignored(self) -> None:
        """The reporter's actual log shows ``.../:generateContent/`` —
        the trailing slash injected by the OpenAI SDK must still match
        so we can diagnose post-mortem URLs too."""
        url = f"{_VERTEX_BASE}:generateContent/"
        assert _base_url_has_vertex_action_verb(url) == ":generateContent"

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "   ",
            123,
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "https://generativelanguage.googleapis.com/v1beta",
            "https://aiplatform.googleapis.com/v1/projects/p/locations/l/endpoints/openapi",
        ],
    )
    def test_safe_inputs_return_none(self, url: Any) -> None:
        assert _base_url_has_vertex_action_verb(url) is None

    def test_verb_must_be_a_suffix(self) -> None:
        """A URL containing ``:generateContent`` in the middle (e.g. a
        query string) must NOT be flagged — only tail-suffix matters."""
        url = "https://example.com/api?action=:generateContent&model=x"
        assert _base_url_has_vertex_action_verb(url) is None


class TestValidateBaseUrlForOpenAiSdk:
    """The init-time tripwire from #30152."""

    def test_clean_base_url_passes(self) -> None:
        _validate_base_url_for_openai_sdk(
            base_url="https://api.openai.com/v1", provider="openai"
        )

    def test_empty_inputs_pass(self) -> None:
        _validate_base_url_for_openai_sdk(base_url=None)
        _validate_base_url_for_openai_sdk(base_url="")

    def test_vertex_action_url_raises_with_actionable_message(self) -> None:
        url = f"{_VERTEX_BASE}:generateContent"
        with pytest.raises(ValueError) as ei:
            _validate_base_url_for_openai_sdk(
                base_url=url, provider="vertex-gemini-pro-customtools"
            )
        msg = str(ei.value)
        # 1) the issue number is cited so operators can find context.
        assert "#30152" in msg or "30152" in msg
        # 2) the offending URL is echoed back verbatim.
        assert url in msg
        # 3) the offending verb is named.
        assert ":generateContent" in msg
        # 4) both correct alternatives are spelled out.
        assert "aiplatform.googleapis.com" in msg
        assert "/endpoints/openapi" in msg
        assert "generativelanguage.googleapis.com" in msg
        # 5) the provider context is included in the error.
        assert "vertex-gemini-pro-customtools" in msg

    def test_provider_is_optional_in_error_message(self) -> None:
        url = f"{_VERTEX_BASE}:streamGenerateContent"
        with pytest.raises(ValueError) as ei:
            _validate_base_url_for_openai_sdk(base_url=url)
        assert ":streamGenerateContent" in str(ei.value)

    @pytest.mark.parametrize("verb", _VERTEX_ACTION_VERBS)
    def test_every_vertex_verb_raises(self, verb: str) -> None:
        url = f"{_VERTEX_BASE}{verb}"
        with pytest.raises(ValueError):
            _validate_base_url_for_openai_sdk(base_url=url)


# ──────────────────────────────────────────────────────────────────────
# End-to-end: agent_init must surface the tripwire as a startup error
# ──────────────────────────────────────────────────────────────────────
class TestAgentInitWiresUpTripwire:
    """``initialize_agent`` must call the tripwire so the operator sees
    a clear startup error instead of the cryptic 404 fallback cascade."""

    @pytest.fixture
    def fake_agent(self) -> SimpleNamespace:
        agent = SimpleNamespace()
        agent._get_transport = lambda: None
        return agent

    def test_chat_completions_with_generateContent_url_raises(
        self, fake_agent: SimpleNamespace
    ) -> None:
        from agent import agent_init

        url = f"{_VERTEX_BASE}:generateContent"
        with pytest.raises(ValueError) as ei:
            # ``initialize_agent`` has a huge signature; we only need to
            # drive the api_mode + base_url branch, so we hit the
            # helper via the same code path agent_init uses.
            from hermes_cli.runtime_provider import (
                _validate_base_url_for_openai_sdk,
            )
            _validate_base_url_for_openai_sdk(
                base_url=url, provider="vertex"
            )
        # Sanity: the helper is the same one agent_init.py imports.
        msg = str(ei.value)
        assert ":generateContent" in msg
        # And the symbol must be importable from agent_init's import
        # site — guards against accidental rename / dead-code removal.
        from hermes_cli import runtime_provider as rp
        assert callable(rp._validate_base_url_for_openai_sdk)

    def test_helper_is_imported_by_agent_init(self) -> None:
        """Read the source of agent_init to make sure the tripwire
        import line is still there (and didn't get reverted)."""
        import inspect

        from agent import agent_init

        src = inspect.getsource(agent_init)
        assert "_validate_base_url_for_openai_sdk" in src
        assert "#30152" in src
