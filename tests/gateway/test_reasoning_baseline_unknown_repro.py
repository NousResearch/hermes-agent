"""Contract: the reasoning-switch announce never fabricates a baseline (PR #83463).

@Enough1122's finding 1, verified still-live 2026-09-14 at head 56e7fcd9:

    gateway/run.py `_resolved_effort_label` returned the constant "medium" when
    `_resolve_session_reasoning_config` returned None, "on the stated assumption
    that the provider default in that case is medium".

On a provider whose real default is NOT medium that is wrong in both directions:

  * selecting the already-effective value  -> phantom "medium -> high";
  * a genuine high -> medium switch        -> compares "medium == medium" and
    stays SILENT, which is the exact failure this feature exists to remove.

These tests drive the REAL `_resolved_effort_label` and the REAL `/reasoning`
handler through the gateway runner, so reverting the fix turns them red.
"""
from __future__ import annotations

import pytest

from hermes_constants import REASONING_BASELINE_UNKNOWN, VALID_REASONING_EFFORTS

from .test_switch_announce import _make_event, _make_runner, _sent_texts


class TestUnknownReasoningBaseline:
    def test_sentinel_can_never_collide_with_a_real_effort(self):
        """The sentinel must be unselectable and unrenderable as an effort."""
        from hermes_constants import parse_reasoning_effort

        assert REASONING_BASELINE_UNKNOWN not in VALID_REASONING_EFFORTS
        assert parse_reasoning_effort(REASONING_BASELINE_UNKNOWN) is None

    def test_unconfigured_baseline_resolves_to_unknown_not_medium(
        self, tmp_path, monkeypatch
    ):
        """The real resolver must not assert a provider default it cannot know."""
        runner, _ = _make_runner(monkeypatch, tmp_path, "agent: {}\n")

        label = runner._resolved_effort_label(session_key=None, model="")

        assert label == REASONING_BASELINE_UNKNOWN, label
        assert label != "medium", (
            "baseline fabricated as the constant 'medium' for a provider whose "
            "real default is unknown from here"
        )

    @pytest.mark.asyncio
    async def test_unconfigured_switch_does_not_announce_a_guessed_transition(
        self, tmp_path, monkeypatch
    ):
        """With nothing configured, the honest output is silence, not a guess.

        Previously this announced "medium → high" on every provider, including
        providers that were already running high (phantom) — and conversely a
        real high → medium switch compared medium == medium and said nothing.
        """
        runner, adapter = _make_runner(monkeypatch, tmp_path, "agent: {}\n")

        await runner._handle_reasoning_command(_make_event("/reasoning high"))

        texts = _sent_texts(adapter)
        assert not any("🔀 Reasoning:" in text for text in texts), (
            f"announced a transition computed from a fabricated baseline: {texts}"
        )

    @pytest.mark.asyncio
    async def test_configured_switch_still_announces(self, tmp_path, monkeypatch):
        """Guard against over-correcting: a provable change must still announce."""
        runner, adapter = _make_runner(
            monkeypatch, tmp_path, "agent:\n  reasoning_effort: low\n"
        )

        await runner._handle_reasoning_command(_make_event("/reasoning xhigh"))

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: low → xhigh" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_configured_no_op_is_still_silent(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(
            monkeypatch, tmp_path, "agent:\n  reasoning_effort: high\n"
        )

        await runner._handle_reasoning_command(_make_event("/reasoning high"))

        texts = _sent_texts(adapter)
        assert not any("🔀 Reasoning:" in text for text in texts), texts
