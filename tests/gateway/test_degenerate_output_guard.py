"""Regression tests for the cron degenerate-output delivery guard.

Covers the 2026-11-02 incident: a cron job whose model output degenerated into
runaway repetition ("NOW FIN. NOW FIN. ...") was delivered verbatim into a user's group
chat, because the Feishu adapter sets ``splits_long_messages = True`` and the delivery
layer handed the whole 213 KB payload to the adapter to chunk natively.

The guard is statistical (no word blacklist), so the tests also pin the negative side:
ordinary reports — including ones that legitimately repeat a line — must pass untouched.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget, guard_degenerate_output
from gateway.degenerate_output import (
    COLLAPSE_MARKER,
    DEFAULT_MAX_DELIVERY_CHARS,
    collapse_degenerate,
    degeneracy_score,
    degenerate_span,
    is_degenerate,
    tokenize,
    truncate_for_delivery,
)


# ---------------------------------------------------------------------------
# Fixtures: the two incident shapes + a normal report
# ---------------------------------------------------------------------------

# Shape 1 — the "NOW FIN" incident: one phrase repeated to exhaustion.
# 4000 copies is long enough that *both* the exactly-periodic feature and the
# windowed duplicate-gram feature trip.
_NOW_FIN = "NOW FIN. " * 4000

# Shape 2 — the second incident ("Executing the patrol" repeated 537 times): a small
# vocabulary cycled rather than one phrase repeated verbatim.
_PATROL = "Executing the patrol. Invoke the tool. Waiting for result. " * 1200

# A normal report: it repeats a board name a few times (legitimately) and is long enough
# to exercise the windowed duplicate-gram feature, but carries real information.
# A readable preamble followed by a looping tail: the shape the collapse exists for.  Unlike a
# pure loop it HAS salvageable text on both sides, so its collapse carries the marker in the
# middle and is what the delivery lanes re-run the guard on.
_PREAMBLE_AND_LOOP = (
    "Here is the patrol summary I managed to write before the model started looping. "
    + _NOW_FIN
    + " Final state: board A has 2 open cards, board B is idle. All tests passed."
)

_NORMAL_REPORT = """# Patrol report — board watch

## Board A
- card 1: blocked on review
- card 2: in progress
- board A needs one more pass before the next window

## Board B
- board B is idle
- board A and board B both need attention

## Summary
No new failures. board A: 2 open. board B: 0 open.
""" + ("\n- repeated but informative status line\n" * 40)


class _ChunkingAdapter:
    """Adapter that declares ``splits_long_messages = True`` (Feishu, Discord, Telegram...)."""

    splits_long_messages = True

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


class _PlainAdapter:
    """Adapter without native chunking."""

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


def _no_save(content, job_id):
    """Audit-save stub: an already-collapsed payload must never reach the audit path."""
    raise AssertionError("an already-collapsed payload must not be audit-saved as an original")


def _router(tmp_path, monkeypatch, adapter, config=None):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    return DeliveryRouter(config or GatewayConfig(), adapters={Platform.DISCORD: adapter})


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TestDegenerateDetection:
    def test_now_fin_repetition_is_detected(self):
        assert is_degenerate(_NOW_FIN) is True

    def test_patrol_cycle_repetition_is_detected(self):
        """The second incident shape: a small vocabulary cycled, not one phrase verbatim."""
        assert is_degenerate(_PATROL) is True

    def test_normal_report_is_not_detected(self):
        assert is_degenerate(_NORMAL_REPORT) is False

    def test_short_text_is_never_degenerate(self):
        """A terse reply must never trip the detector."""
        assert is_degenerate("Now fin.") is False
        assert is_degenerate("") is False

    def test_repeated_line_in_a_real_report_is_not_degenerate(self):
        """Repetition that carries information is legitimate — the guard is statistical, not lexical."""
        report = "\n".join(
            f"- board-{i % 4}: item {i} complete, verified by the reviewer, closing now"
            for i in range(200)
        )
        assert is_degenerate(report) is False

    def test_detector_uses_no_word_blacklist(self):
        """The incident phrases alone must not be enough — the guard keys on repetition shape."""
        # A single occurrence of the incident vocabulary in a normal report is not degenerate.
        assert is_degenerate(
            "The agent said NOW FIN and then STILL FIN, but that was a single mention. " * 3
        ) is False

    def test_score_reports_both_features(self):
        score = degeneracy_score(_NOW_FIN)
        assert score is not None
        assert score["periodic_run"] >= 25
        assert score["window_dup_ratio"] >= 0.85

    def test_score_is_none_for_short_text(self):
        assert degeneracy_score("too short") is None


class TestCollapse:
    def test_collapse_drops_the_repetitive_span(self):
        collapsed = collapse_degenerate(_NOW_FIN)
        assert len(collapsed) < len(_NOW_FIN)
        assert "NOW FIN" not in collapsed

    def test_collapse_keeps_readable_prefix_and_suffix(self):
        body = (
            "Here is the patrol summary I managed to write before the model started looping. "
            + _NOW_FIN
            + " Final state: board A has 2 open cards, board B is idle. All tests passed."
        )
        collapsed = collapse_degenerate(body)
        assert "patrol summary" in collapsed
        assert "Final state" in collapsed
        assert "NOW FIN" not in collapsed

    def test_collapse_of_a_clean_text_is_identity(self):
        assert collapse_degenerate(_NORMAL_REPORT) == _NORMAL_REPORT

    def test_collapse_keeps_preamble_when_both_features_fire(self):
        """A long exactly-periodic loop trips *both* features.

        The window feature localizes the region with a coarse, window-aligned
        lower bound (up to ``WINDOW_GRAMS`` tokens early); the periodic feature
        reports the exact onset.  Taking the union of the two starts used to
        delete the readable summary that preceded a long loop.  The collapse
        must keep that preamble.
        """
        span = degenerate_span(body := (
            "Here is the patrol summary I managed to write before the model started looping. "
            + _NOW_FIN
            + " Final state: board A has 2 open cards, board B is idle."
        ))
        assert span is not None
        assert span[0] >= len(tokenize(
            "Here is the patrol summary I managed to write before the model started looping."
        )) - 2, f"span start {span[0]} swallowed the readable preamble"
        assert "NOW FIN" not in collapse_degenerate(body)

    def test_collapse_never_keeps_degenerate_prefix(self):
        """When the pre-onset text is itself repetitive, the coarse bound wins.

        Guards the converse of the preamble fix: widening the retained prefix
        must not smuggle degenerate content back into the collapse.
        """
        body = _NOW_FIN + _NOW_FIN
        collapsed = collapse_degenerate(body)
        assert "NOW FIN" not in collapsed

    def test_collapse_result_is_bounded(self):
        """The collapsed message must fit one platform message."""
        collapsed = collapse_degenerate("x" * 200_000 + _NOW_FIN)
        assert len(collapsed) <= 4000

    def test_collapse_is_idempotent(self):
        """A second pass over an already-collapsed payload must not shrink it further.

        The cron scheduler collapses the run's final response before composing its delivery,
        and both delivery lanes then re-run the guard on that same text.  Re-collapsing the
        result mangled it (a 2,826-char delivery came back as 125 chars): the marker is Chinese
        prose and a preserved excerpt still ends with a repetitive tail, which a second pass
        re-detected and cut away.
        """
        once = collapse_degenerate(_PREAMBLE_AND_LOOP)
        assert once != _PREAMBLE_AND_LOOP, "fixture must actually collapse"
        assert COLLAPSE_MARKER in once
        assert collapse_degenerate(once) == once, "the gateway guard is not idempotent"

    def test_an_already_collapsed_payload_is_not_re_saved_as_an_original(self):
        """The reconnect/fallback lanes re-run the delivery guard on collapsed text.

        Passing it through must not audit-save the excerpt as if it were the full original, and
        must not re-collapse it: the guard chain is idempotent so a replay of the collapsed
        payload is byte-identical to what already went out.
        """
        collapsed = collapse_degenerate(_PREAMBLE_AND_LOOP)
        guarded, audit_path = guard_degenerate_output(collapsed, "job1", _no_save)
        assert guarded == collapsed
        assert audit_path is None, "an already-collapsed payload was re-saved for audit"


class TestTruncateForDelivery:
    def test_short_text_is_unchanged(self):
        assert truncate_for_delivery("hello", limit=100) == "hello"

    def test_over_limit_text_is_truncated_with_footer(self):
        out = truncate_for_delivery("x" * 500, limit=100, saved_path="/tmp/out.txt")
        assert len(out) <= 100
        assert "/tmp/out.txt" in out

    def test_zero_limit_disables_the_cap(self):
        assert truncate_for_delivery("x" * 500, limit=0) == "x" * 500


# ---------------------------------------------------------------------------
# Delivery chokepoint
# ---------------------------------------------------------------------------

class TestDeliveryGuard:
    @pytest.mark.asyncio
    async def test_degenerate_payload_is_not_delivered_verbatim(self, tmp_path, monkeypatch):
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        await router._deliver_to_platform(target, _NOW_FIN, metadata={"job_id": "job1"})

        delivered = adapter.calls[0]["content"]
        assert delivered != _NOW_FIN
        assert "NOW FIN" not in delivered
        assert len(delivered) < len(_NOW_FIN)

    @pytest.mark.asyncio
    async def test_chunking_adapter_never_receives_the_full_payload(self, tmp_path, monkeypatch):
        """The regression that mattered: a chunking adapter must not receive the whole 213 KB."""
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        payload = ("NOW FIN. " * 30_000)  # ~270 KB, larger than the 213 KB incident
        await router._deliver_to_platform(target, payload, metadata={"job_id": "job1"})

        delivered = adapter.calls[0]["content"]
        assert len(delivered) <= DEFAULT_MAX_DELIVERY_CHARS
        assert delivered != payload

    @pytest.mark.asyncio
    async def test_oversized_but_normal_payload_is_capped_before_chunking(self, tmp_path, monkeypatch):
        """A non-degenerate oversized payload is still capped before the adapter sees it."""
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        payload = "\n".join(f"status line {i}: board report" for i in range(5000))
        await router._deliver_to_platform(target, payload, metadata={"job_id": "job1"})

        delivered = adapter.calls[0]["content"]
        assert len(delivered) <= DEFAULT_MAX_DELIVERY_CHARS

    @pytest.mark.asyncio
    async def test_normal_report_is_delivered_unchanged(self, tmp_path, monkeypatch):
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        await router._deliver_to_platform(target, _NORMAL_REPORT, metadata={"job_id": "job1"})

        assert adapter.calls[0]["content"] == _NORMAL_REPORT

    @pytest.mark.asyncio
    async def test_short_message_is_delivered_unchanged(self, tmp_path, monkeypatch):
        adapter = _PlainAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        await router._deliver_to_platform(target, "Nightly patrol done — 2 open cards.",
                                          metadata={"job_id": "job1"})

        assert adapter.calls[0]["content"] == "Nightly patrol done — 2 open cards."

    @pytest.mark.asyncio
    async def test_original_is_preserved_in_the_audit_copy(self, tmp_path, monkeypatch):
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        await router._deliver_to_platform(target, _NOW_FIN, metadata={"job_id": "job1"})

        saved = list((tmp_path / "cron" / "output").glob("job1_*.txt"))
        assert len(saved) == 1
        assert saved[0].read_text(encoding="utf-8") == _NOW_FIN

    @pytest.mark.asyncio
    async def test_audit_save_failure_does_not_crash_the_delivery_path(self, tmp_path, monkeypatch):
        """A failed audit save is best-effort: delivery still happens with the collapsed message."""
        adapter = _ChunkingAdapter()
        router = _router(tmp_path, monkeypatch, adapter)
        target = DeliveryTarget.parse("discord:123")

        def _boom(_content, _job_id):
            raise OSError("disk full")

        monkeypatch.setattr(router, "_save_full_output", _boom)

        result = await router._deliver_to_platform(target, _NOW_FIN, metadata={"job_id": "job1"})

        assert result["success"] is True
        delivered = adapter.calls[0]["content"]
        assert "NOW FIN" not in delivered
        assert len(delivered) < len(_NOW_FIN)

    @pytest.mark.asyncio
    async def test_local_delivery_is_not_filtered(self, tmp_path, monkeypatch):
        """Local/file delivery keeps the full artifact — the audit trail must stay intact."""
        monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
        router = DeliveryRouter(GatewayConfig())
        job_id = "localjob"

        result = router._deliver_local(_NOW_FIN, job_id, "job", {})

        written = Path(result["path"]).read_text(encoding="utf-8")
        assert _NOW_FIN in written


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestDeliveryCapConfig:
    def test_default_cap(self):
        assert GatewayConfig().max_delivery_chars == DEFAULT_MAX_DELIVERY_CHARS
        assert DEFAULT_MAX_DELIVERY_CHARS == 20_000

    def test_cap_read_from_cron_section(self):
        config = GatewayConfig.from_dict({"cron": {"max_delivery_chars": 4321}})
        assert config.max_delivery_chars == 4321

    def test_cap_read_from_flat_top_level_key(self):
        assert GatewayConfig.from_dict({"max_delivery_chars": 999}).max_delivery_chars == 999

    def test_zero_disables_the_cap(self):
        assert GatewayConfig.from_dict({"cron": {"max_delivery_chars": 0}}).max_delivery_chars == 0

    def test_malformed_cap_falls_back_to_the_default(self):
        config = GatewayConfig.from_dict({"cron": {"max_delivery_chars": "many"}})
        assert config.max_delivery_chars == DEFAULT_MAX_DELIVERY_CHARS

    def test_config_yaml_cron_section_is_bridged_into_the_gateway_config(self):
        """The gateway loader builds gw_data FLAT; the ``cron`` section is not forwarded as-is."""
        from gateway.config_loader import bridge_toplevel_keys

        gw_data = {}
        bridge_toplevel_keys({"cron": {"max_delivery_chars": 7777}}, None, gw_data)

        assert gw_data.get("max_delivery_chars") == 7777
        assert GatewayConfig.from_dict(gw_data).max_delivery_chars == 7777

    def test_cli_default_config_documents_the_knob(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["cron"]["max_delivery_chars"] == DEFAULT_MAX_DELIVERY_CHARS

    @pytest.mark.asyncio
    async def test_configured_cap_is_enforced(self, tmp_path, monkeypatch):
        adapter = _ChunkingAdapter()
        config = GatewayConfig(max_delivery_chars=500)
        router = _router(tmp_path, monkeypatch, adapter, config=config)
        target = DeliveryTarget.parse("discord:123")

        payload = "\n".join(f"status line {i}: board report" for i in range(200))
        await router._deliver_to_platform(target, payload, metadata={"job_id": "job1"})

        assert len(adapter.calls[0]["content"]) <= 500

    @pytest.mark.asyncio
    async def test_zero_cap_disables_truncation(self, tmp_path, monkeypatch):
        adapter = _ChunkingAdapter()
        config = GatewayConfig(max_delivery_chars=0)
        router = _router(tmp_path, monkeypatch, adapter, config=config)
        target = DeliveryTarget.parse("discord:123")

        payload = "\n".join(f"status line {i}: board report" for i in range(200))
        await router._deliver_to_platform(target, payload, metadata={"job_id": "job1"})

        assert adapter.calls[0]["content"] == payload

    @pytest.mark.asyncio
    async def test_non_numeric_cap_object_falls_back_to_the_default(self, tmp_path, monkeypatch):
        """A config value that is not a real number must not become a tiny cap.

        ``int(<object>)`` can succeed with a surprising value (``int(MagicMock())`` is 1),
        which would truncate every delivery to a single character. Any non-numeric object
        is treated as a configuration error and falls back to the default cap.
        """
        adapter = _ChunkingAdapter()
        config = GatewayConfig(max_delivery_chars=MagicMock())
        router = _router(tmp_path, monkeypatch, adapter, config=config)
        target = DeliveryTarget.parse("discord:123")

        payload = "\n".join(f"status line {i}: board report" for i in range(200))
        await router._deliver_to_platform(target, payload, metadata={"job_id": "job1"})

        # The default cap is far above this payload, so nothing is truncated.
        assert adapter.calls[0]["content"] == payload

    def test_non_numeric_cap_object_is_rejected_by_the_config_normalizer(self):
        from gateway.config import normalize_delivery_char_cap

        assert normalize_delivery_char_cap(MagicMock()) == DEFAULT_MAX_DELIVERY_CHARS
        assert normalize_delivery_char_cap(object()) == DEFAULT_MAX_DELIVERY_CHARS
        assert normalize_delivery_char_cap(True) == DEFAULT_MAX_DELIVERY_CHARS
        assert normalize_delivery_char_cap(1.5) == DEFAULT_MAX_DELIVERY_CHARS

    def test_integer_valued_string_cap_is_accepted(self):
        from gateway.config import normalize_delivery_char_cap

        assert normalize_delivery_char_cap("4321") == 4321
        assert normalize_delivery_char_cap("0") == 0


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_CURRENT_TEST", "")
    raise SystemExit(pytest.main([__file__, "-v"]))
