"""Regression tests for #30818 — DeepSeek V4 first message HTTP 400.

The bug
=======

``provider: deepseek`` + ``model: deepseek-v4-flash`` (or any V4 family
model) returns HTTP 400 on the very first message — not on multi-turn
replays, not on tool calls, just a plain "hello" against
``https://api.deepseek.com``.  ``curl`` with the same key/model
succeeds.  Switching to ``provider: custom`` + ``api_mode:
openai-completions`` also succeeds.

Root cause: the DeepSeek profile in
``plugins/model-providers/deepseek/__init__.py`` unconditionally
injected::

    extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

for every V4-family model regardless of the user's
``reasoning_config``.  ``extra_body`` is unwrapped into the request
body by the OpenAI SDK, so DeepSeek saw an unrecognized top-level
``thinking`` field and rejected the request.

The fix
=======

Make the ``extra_body.thinking`` injection opt-in:

* Default path (no ``reasoning_config``, or ``reasoning_config`` without
  an ``enabled`` key) — DON'T emit ``extra_body.thinking``.  The server
  applies its own defaults and the request succeeds.
* Explicit opt-in (``reasoning_config={"enabled": True/False, ...}``) —
  forward the legacy Kimi-style ``thinking`` payload so users who
  depend on the explicit toggle still get it.
* ``reasoning_effort`` is still forwarded when the user sets
  ``reasoning_config.effort`` — that's a separate parameter and is
  not the smoking gun for #30818.

These tests pin every branch of the new contract.
"""

from __future__ import annotations

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _msgs():
    return [{"role": "user", "content": "hello"}]


def _build_kwargs(
    transport: ChatCompletionsTransport,
    *,
    model: str,
    reasoning_config: dict | None = None,
    request_overrides: dict | None = None,
) -> dict:
    """Drive _build_kwargs_from_profile the way the agent runtime does.

    Returns the kwargs dict that would be handed to
    ``client.chat.completions.create()``.
    """
    profile = get_provider_profile("deepseek")
    assert profile is not None, "deepseek profile must be registered"
    return transport.build_kwargs(
        model=model,
        messages=_msgs(),
        tools=None,
        provider_profile=profile,
        max_tokens=None,
        max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
        timeout=300,
        reasoning_config=reasoning_config,
        request_overrides=request_overrides,
        session_id="test-session",
        ollama_num_ctx=None,
    )


# ────────────────────────────────────────────────────────────────────────────
# The default path — the exact scenario the bug reporter hit
# ────────────────────────────────────────────────────────────────────────────


class TestDefaultPathNoThinkingInjected:
    """No ``reasoning_config`` provided → no ``extra_body.thinking``
    emitted → DeepSeek V4 native API accepts the request.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "deepseek-v4-flash",   # exact model from the bug report
            "deepseek-v4-pro",
            "deepseek-v4-experimental",  # forward-compat: future V4-* variants
            "deepseek-v5-flash",   # forward-compat: future V5+ generations
            "deepseek-reasoner",   # legacy R1 thinking model
        ],
    )
    def test_no_extra_body_thinking_when_reasoning_config_is_none(
        self, transport, model
    ):
        kwargs = _build_kwargs(transport, model=model, reasoning_config=None)

        extra_body = kwargs.get("extra_body", {})
        assert "thinking" not in extra_body, (
            f"#30818 regression: ``extra_body.thinking`` injected by default "
            f"for {model!r}; DeepSeek V4 API rejects this with HTTP 400. "
            f"Got extra_body={extra_body!r}"
        )
        assert "reasoning_effort" not in kwargs, (
            "Default path must not set reasoning_effort either — let the "
            "server pick its own default."
        )

    def test_no_extra_body_thinking_when_reasoning_config_is_empty_dict(
        self, transport
    ):
        """Empty dict is the shape ``hermes_cli/config.py`` produces when
        the user has a ``reasoning:`` section but didn't populate
        ``enabled``/``effort``.  Must behave exactly like ``None``.
        """
        kwargs = _build_kwargs(
            transport, model="deepseek-v4-flash", reasoning_config={}
        )
        extra_body = kwargs.get("extra_body", {})
        assert "thinking" not in extra_body
        assert "reasoning_effort" not in kwargs

    def test_no_extra_body_thinking_when_only_effort_configured(
        self, transport
    ):
        """User sets ``reasoning.effort: medium`` only — ``reasoning_effort``
        should be forwarded but ``extra_body.thinking`` must stay absent
        because the user did not opt into the thinking toggle.

        Effort and thinking are independent parameters; the bug was
        coupling them on the request side.
        """
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"effort": "medium"},
        )
        extra_body = kwargs.get("extra_body", {})
        assert "thinking" not in extra_body
        assert kwargs.get("reasoning_effort") == "medium"


# ────────────────────────────────────────────────────────────────────────────
# Opt-in path — legacy Kimi-style toggle still works for explicit users
# ────────────────────────────────────────────────────────────────────────────


class TestExplicitThinkingOptIn:
    """Users who set ``reasoning_config.enabled`` explicitly still get
    the Kimi-style ``extra_body.thinking`` payload, mirroring the
    pre-#30818 behavior for the subset of users who actually configured
    it.
    """

    def test_enabled_true_forwards_thinking_enabled(self, transport):
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"enabled": True},
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_enabled_false_forwards_thinking_disabled(self, transport):
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"enabled": False},
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        # Disabled thinking → reasoning_effort must NOT be forwarded
        # (no point picking an effort level when thinking is off).
        assert "reasoning_effort" not in kwargs

    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        ],
    )
    def test_enabled_true_with_effort_forwards_both(
        self, transport, effort, expected
    ):
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert kwargs["reasoning_effort"] == expected

    def test_enabled_true_without_effort_omits_reasoning_effort(self, transport):
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"enabled": True},
        )
        # Explicit thinking ON but no effort → let server pick default.
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kwargs

    def test_invalid_effort_silently_dropped(self, transport):
        """Unknown effort levels should be silently ignored rather than
        forwarded as-is (which the API would also 400 on).  This was the
        pre-fix behavior and we keep it.
        """
        kwargs = _build_kwargs(
            transport,
            model="deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": "ultra-mega-high"},
        )
        assert "reasoning_effort" not in kwargs


# ────────────────────────────────────────────────────────────────────────────
# V3 / non-thinking models — must remain untouched (no behavior change)
# ────────────────────────────────────────────────────────────────────────────


class TestV3AndNonThinkingModelsUnchanged:
    """The V3 chat model has never had thinking mode.  The fix must
    leave its wire format untouched — no new extras, no behavior change.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "deepseek-chat",   # V3
            "deepseek-v3-pro",  # explicit V3 family
            "deepseek-coder",  # legacy non-thinking
            "",                # empty — defensive
        ],
    )
    def test_no_extras_for_non_thinking_models_default(self, transport, model):
        kwargs = _build_kwargs(transport, model=model, reasoning_config=None)
        extra_body = kwargs.get("extra_body", {})
        assert "thinking" not in extra_body
        assert "reasoning_effort" not in kwargs

    def test_v3_ignores_explicit_thinking_opt_in(self, transport):
        """Even when the user opts into thinking via ``reasoning_config``,
        a V3 model must not get ``extra_body.thinking`` — V3 doesn't
        support it and forwarding the field would re-introduce the same
        class of HTTP 400 the fix avoids for V4.
        """
        kwargs = _build_kwargs(
            transport,
            model="deepseek-chat",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        extra_body = kwargs.get("extra_body", {})
        assert "thinking" not in extra_body
        assert "reasoning_effort" not in kwargs


# ────────────────────────────────────────────────────────────────────────────
# Profile-level pins (catalog stays put, fixture wiring intact)
# ────────────────────────────────────────────────────────────────────────────


class TestProfileMetadataUnchanged:
    """Static pins so a future refactor that breaks the profile's
    identity (rename, accidental deletion, base_url change) fails here
    instead of silently breaking every DeepSeek user.
    """

    def test_profile_registered_under_canonical_name(self):
        profile = get_provider_profile("deepseek")
        assert profile is not None
        assert profile.name == "deepseek"

    def test_profile_resolves_via_chat_alias(self):
        """``deepseek-chat`` is registered as an alias so users who
        configure ``provider: deepseek-chat`` (a common typo) still
        reach the same profile.
        """
        profile = get_provider_profile("deepseek-chat")
        assert profile is not None
        assert profile.name == "deepseek"

    def test_profile_base_url_is_official_v1_endpoint(self):
        profile = get_provider_profile("deepseek")
        assert profile.base_url == "https://api.deepseek.com/v1"

    def test_profile_advertises_deepseek_api_key_env(self):
        profile = get_provider_profile("deepseek")
        assert "DEEPSEEK_API_KEY" in profile.env_vars

    def test_module_helpers_recognise_v4_family(self):
        """The thinking-capability classifier underpins the fix; pin its
        behaviour so a future change there can't silently re-enable
        ``extra_body.thinking`` for non-V4 models.
        """
        from plugins.model_providers.deepseek import _model_supports_thinking  # type: ignore[import-not-found]

        assert _model_supports_thinking("deepseek-v4-flash") is True
        assert _model_supports_thinking("deepseek-v4-pro") is True
        assert _model_supports_thinking("deepseek-v5-flash") is True
        assert _model_supports_thinking("deepseek-reasoner") is True
        assert _model_supports_thinking("deepseek-chat") is False
        assert _model_supports_thinking("deepseek-v3-pro") is False
        assert _model_supports_thinking("") is False
        assert _model_supports_thinking(None) is False


# ────────────────────────────────────────────────────────────────────────────
# Source-level structural pin
# ────────────────────────────────────────────────────────────────────────────


class TestSourceGuards:
    """Read the profile source and assert the structural contract.  A
    future refactor that re-introduces an unconditional
    ``extra_body['thinking']`` assignment would fail here even if it
    happened to pass the behavioral tests above (e.g. by sneaking the
    assignment behind a different gate that the existing tests
    don't exercise).
    """

    def test_thinking_assignment_is_guarded_by_opt_in_helper(self):
        from pathlib import Path

        source = Path("plugins/model-providers/deepseek/__init__.py").read_text()
        # The opt-in helper must exist and must guard the assignment.
        assert "_user_opted_into_thinking_config" in source, (
            "Opt-in helper was removed — re-introduces #30818"
        )
        thinking_idx = source.find("extra_body[\"thinking\"]")
        assert thinking_idx != -1, "thinking assignment disappeared"
        # The 200 chars before the assignment must mention the opt-in
        # guard.  Catches refactors that move the assignment outside
        # the guard.
        preamble = source[max(0, thinking_idx - 200):thinking_idx]
        assert "_user_opted_into_thinking_config" in preamble, (
            "extra_body['thinking'] is no longer guarded by the opt-in "
            "helper — re-introduces #30818."
        )

    def test_docstring_cites_issue_30818(self):
        from pathlib import Path

        source = Path("plugins/model-providers/deepseek/__init__.py").read_text()
        assert "#30818" in source, (
            "Module docstring no longer references #30818; future maintainers "
            "won't understand why ``extra_body.thinking`` is opt-in. Add the "
            "citation back."
        )
