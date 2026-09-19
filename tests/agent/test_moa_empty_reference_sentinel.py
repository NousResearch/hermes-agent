"""A reference that returns NO usable text must not count as a successful advisor.

A reasoning model can spend its ENTIRE output budget on hidden reasoning and return
``content: ""`` (finish_reason=length) while still billing the call. Before this fix
``_run_reference`` converted that into the bare placeholder ``"(empty response)"``,
which ``_is_failed_reference`` did not recognise — so the empty advisor was handed to
the aggregator as though it were real guidance and ``degraded_reference_policy: loud``
never fired. On a 2-advisor preset that silently halves the cross-model diversity MoA
exists to provide, while still billing for it.

These exercise the real functions (``_run_reference`` → ``_guidance_inputs`` /
``aggregate_moa_context``) with a faked ``call_llm``, so they prove the production
logic rather than a re-implementation of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import moa_loop
from agent.usage_pricing import CanonicalUsage


def _response(content, *, completion_tokens=None):
    """A completed response shaped like the OpenAI chat-completions SDK."""
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="length" if not content else "stop")
    usage = None
    if completion_tokens is not None:
        details = SimpleNamespace(cached_tokens=0, cache_write_tokens=0)
        usage = SimpleNamespace(
            prompt_tokens=130, completion_tokens=completion_tokens,
            prompt_tokens_details=details, output_tokens_details=None,
        )
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-model")


def _stub_runtime(monkeypatch):
    """Bare provider/model runtime + no transport, so `_extract_text` uses choices[]."""
    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(
        moa_loop, "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )
    monkeypatch.setattr(
        moa_loop, "_maybe_apply_moa_cache_control", lambda msgs, rt, **kwargs: msgs,
    )


# --- the classifier ---------------------------------------------------------


class TestEmptyReferenceDetection:
    def test_empty_sentinel_is_a_failed_reference(self):
        assert moa_loop._is_failed_reference(moa_loop._EMPTY_REFERENCE_NOTE) is True

    def test_legacy_placeholder_still_classifies_as_failed(self):
        """A fan-out cached by a pre-update process must not leak as guidance."""
        assert moa_loop._is_failed_reference("(empty response)") is True

    def test_blank_and_whitespace_output_is_a_failed_reference(self):
        assert moa_loop._is_failed_reference("") is True
        assert moa_loop._is_failed_reference("   \n\t ") is True

    def test_missing_text_is_a_failed_reference(self):
        assert moa_loop._is_failed_reference(None) is True

    @pytest.mark.parametrize(
        "text", ["[failed: HTTP 500]", "[skipped: interrupted by user]", "  [FAILED: boom]"],
    )
    def test_existing_sentinels_still_detected(self, text):
        assert moa_loop._is_failed_reference(text) is True

    def test_real_advice_is_not_a_failed_reference(self):
        assert moa_loop._is_failed_reference("Use the read_file tool first.") is False
        # Text that merely mentions the words is still real advice.
        assert moa_loop._is_failed_reference("the reference came back empty response") is False


# --- the production path ----------------------------------------------------


class TestEmptyAdvisorIsDisclosed:
    def test_empty_advisor_yields_loud_degraded_notice(self, monkeypatch):
        """2 advisors, one empty → degraded != "" under the loud policy, and the
        empty slot is NOT handed to the aggregator."""
        _stub_runtime(monkeypatch)
        monkeypatch.setattr(
            moa_loop, "call_llm",
            lambda **kw: _response("" if kw.get("provider") == "empty" else "useful advice"),
        )

        outputs = moa_loop._run_references_parallel(
            [
                {"provider": "empty", "model": "silent-advisor"},
                {"provider": "good", "model": "good-advisor"},
            ],
            [{"role": "user", "content": "do the thing"}],
            max_tokens=6000,
        )

        assert outputs[0][1] == moa_loop._EMPTY_REFERENCE_NOTE

        agg_refs, degraded, all_failed = moa_loop._guidance_inputs(outputs, False, "loud")

        assert [label for label, _t, _a in agg_refs] == ["good:good-advisor"]
        assert degraded == "[Reference models unavailable: empty:silent-advisor]"
        assert all_failed is False

    def test_silent_policy_suppresses_the_notice_but_still_filters(self, monkeypatch):
        _stub_runtime(monkeypatch)
        monkeypatch.setattr(
            moa_loop, "call_llm",
            lambda **kw: _response("" if kw.get("provider") == "empty" else "useful advice"),
        )

        outputs = moa_loop._run_references_parallel(
            [{"provider": "empty", "model": "silent-advisor"},
             {"provider": "good", "model": "good-advisor"}],
            [{"role": "user", "content": "go"}],
            max_tokens=6000,
        )
        agg_refs, degraded, all_failed = moa_loop._guidance_inputs(outputs, False, "silent")

        assert len(agg_refs) == 1
        assert degraded == ""
        assert all_failed is False

    def test_all_advisors_empty_skips_aggregator_synthesis(self, monkeypatch):
        """Every advisor empty ⇒ all_failed, so the one-shot path must not pay for a
        synthesis over nothing."""
        _stub_runtime(monkeypatch)
        aggregator_calls = []

        def fake_call_llm(**kw):
            if kw.get("task") == "moa_reference":
                return _response("")
            aggregator_calls.append(kw)
            raise AssertionError("aggregator must not run when every advisor returned nothing")

        monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

        result = moa_loop.aggregate_moa_context(
            user_prompt="do something",
            api_messages=[{"role": "user", "content": "do something"}],
            reference_models=[
                {"provider": "empty-a", "model": "m1"},
                {"provider": "empty-b", "model": "m2"},
            ],
            aggregator={"provider": "openrouter", "model": "aggregator"},
        )

        assert aggregator_calls == []
        assert "all reference models failed" in result
        assert "Reference models unavailable" in result

    def test_billed_empty_advisor_is_still_accounted(self, monkeypatch):
        """The empty call BILLED its tokens — the fix must not drop the spend."""
        _stub_runtime(monkeypatch)
        monkeypatch.setattr(
            moa_loop, "call_llm",
            lambda **kw: _response("", completion_tokens=6000),
        )

        _label, text, acct = moa_loop._run_reference(
            {"provider": "empty", "model": "reasoning-advisor"},
            [{"role": "user", "content": "long prompt"}],
            max_tokens=6000,
        )

        assert text == moa_loop._EMPTY_REFERENCE_NOTE
        assert isinstance(acct, moa_loop._RefAccounting)
        assert acct.usage.output_tokens == 6000
        assert acct.output == moa_loop._EMPTY_REFERENCE_NOTE


class TestFacadeGuidance:
    """The in-agent-loop surface: an empty advisor must not appear as a reference
    block in the acting model's prompt."""

    def _preset_home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            """
moa:
  default_preset: review
  presets:
    review:
      degraded_reference_policy: loud
      reference_models:
        - provider: empty
          model: silent-advisor
        - provider: good
          model: good-advisor
      aggregator:
        provider: openrouter
        model: aggregator
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

    def test_empty_advisor_never_reaches_the_aggregator_prompt(self, tmp_path, monkeypatch):
        self._preset_home(tmp_path, monkeypatch)
        _stub_runtime(monkeypatch)
        monkeypatch.setattr(
            moa_loop, "call_llm",
            lambda **kw: _response("useful advice") if kw.get("provider") == "good" else _response(""),
        )

        facade = moa_loop.MoAChatCompletions("review")
        prepared = facade.create(
            messages=[{"role": "user", "content": "review this"}], tools=[], _moa_prepare_only=True,
        )

        guidance = prepared["guidance"]
        assert guidance is not None
        assert "Reference models unavailable: empty:silent-advisor" in guidance
        assert "silent-advisor:\n[failed: empty response]" not in guidance
        # Only the surviving advisor is listed as a reference.
        assert "References: good:good-advisor" in guidance


class TestSlotMetrics:
    def test_empty_output_is_flagged(self):
        from agent.moa_trace import slot_metrics

        acct = moa_loop._RefAccounting(CanonicalUsage(input_tokens=130, output_tokens=6000))

        flagged = slot_metrics(acct, "empty:silent-advisor", output=moa_loop._EMPTY_REFERENCE_NOTE)
        assert flagged["failed"] is True
        # The billed tokens are still reported — flagged, not hidden.
        assert flagged["usage"]["output_tokens"] == 6000

    def test_real_output_is_not_flagged(self):
        from agent.moa_trace import slot_metrics

        acct = moa_loop._RefAccounting(CanonicalUsage(input_tokens=100, output_tokens=50))
        assert slot_metrics(acct, "good:advisor", output="advice")["failed"] is False
