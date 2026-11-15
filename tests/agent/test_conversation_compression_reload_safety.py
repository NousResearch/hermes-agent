"""Regression tests for ContextVar & singleton preservation across importlib.reload (#CF-79)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import agent.context_compressor as cc_mod
import agent.conversation_compression as conv_comp


def test_reload_mid_summary_dispatch_preserves_contextvar_and_resets_cleanly():
    """Reloading agent.conversation_compression while _run_summary_dispatch is in-flight
    must NOT raise ValueError('<Token ...> was created by a different ContextVar')."""
    orig_var = conv_comp._COMPRESSOR_ATTEMPT_GENERATION
    orig_lock = conv_comp._COMPRESSOR_ATTEMPT_LOCK
    orig_adm_lock = conv_comp._compress_admission_lock

    dummy_compressor = SimpleNamespace()
    dummy_agent = SimpleNamespace(session_id="test_cf79", context_compressor=dummy_compressor)

    seen_inside = {}

    def _fake_compress_fn(messages, **_kwargs):
        seen_inside["gen_before_reload"] = conv_comp._COMPRESSOR_ATTEMPT_GENERATION.get()
        # Simulate or-credit-clamp / hot-reload reloading agent.conversation_compression mid-call
        importlib.reload(conv_comp)
        seen_inside["gen_after_reload"] = conv_comp._COMPRESSOR_ATTEMPT_GENERATION.get()
        seen_inside["same_var"] = conv_comp._COMPRESSOR_ATTEMPT_GENERATION is orig_var
        seen_inside["same_lock"] = conv_comp._COMPRESSOR_ATTEMPT_LOCK is orig_lock
        seen_inside["same_adm_lock"] = conv_comp._compress_admission_lock is orig_adm_lock
        return list(messages)

    out = conv_comp._run_summary_dispatch(
        dummy_agent,
        [{"role": "user", "content": "hi"}],
        _fake_compress_fn,
        {},
        commit_fence=None,
        attempt_generation=42,
        hard_cancel_event=None,
    )
    assert out == [{"role": "user", "content": "hi"}]
    assert seen_inside == {
        "gen_before_reload": 42,
        "gen_after_reload": 42,
        "same_var": True,
        "same_lock": True,
        "same_adm_lock": True,
    }
    assert conv_comp._COMPRESSOR_ATTEMPT_GENERATION.get() is None


def test_pin_summary_route_survives_context_compressor_reload():
    """Reloading agent.context_compressor inside pin_summary_route must reset cleanly."""
    orig_pin_var = cc_mod._SUMMARY_ROUTE_PIN
    route = {"provider": "vertex", "model": "google/gemini-3.8-flash"}
    with cc_mod.pin_summary_route(route):
        importlib.reload(cc_mod)
        assert cc_mod._SUMMARY_ROUTE_PIN is orig_pin_var
        assert cc_mod._SUMMARY_ROUTE_PIN.get() == route
    assert cc_mod._SUMMARY_ROUTE_PIN.get() is None


def test_token_var_reset_survives_forced_contextvar_replacement():
    """Even if _COMPRESSOR_ATTEMPT_GENERATION global is forcibly replaced mid-dispatch
    (pre-CF-79 reload behavior), _attempt_ctx_token.var.reset(_attempt_ctx_token) succeeds
    whereas calling reset() on the replacement ContextVar raises ValueError."""
    import contextvars
    import pytest

    saved_var = conv_comp._COMPRESSOR_ATTEMPT_GENERATION
    dummy_compressor = SimpleNamespace()
    dummy_agent = SimpleNamespace(session_id="test_cf79_forced", context_compressor=dummy_compressor)

    def _force_replace_compress_fn(messages, **_kwargs):
        old_tok = saved_var.set(99)
        replacement_var = contextvars.ContextVar("hermes_compressor_attempt_generation", default=None)
        # Prove negative control: resetting old_tok on replacement_var raises ValueError
        with pytest.raises(ValueError, match="was created by a different ContextVar"):
            replacement_var.reset(old_tok)
        old_tok.var.reset(old_tok)
        conv_comp._COMPRESSOR_ATTEMPT_GENERATION = replacement_var
        return list(messages)

    try:
        out = conv_comp._run_summary_dispatch(
            dummy_agent,
            [{"role": "user", "content": "polarity"}],
            _force_replace_compress_fn,
            {},
            commit_fence=None,
            attempt_generation=7,
            hard_cancel_event=None,
        )
        assert out == [{"role": "user", "content": "polarity"}]
        assert saved_var.get() is None
    finally:
        conv_comp._COMPRESSOR_ATTEMPT_GENERATION = saved_var


def test_or_credit_clamp_reload_if_changed_mtime_gate_polarity(tmp_path, monkeypatch):
    """Verify _reload_if_changed mtime/size gate reloads on first pass / mtime change / force=True
    and skips reload when (st_mtime_ns, st_size) on disk is unchanged."""
    import os
    import sys

    def _reload_if_changed(mod, *, force: bool = False) -> bool:
        if mod is None:
            return False
        mod_file = getattr(mod, "__file__", None)
        sig = None
        if mod_file:
            try:
                st = os.stat(mod_file)
                sig = (st.st_mtime_ns, st.st_size)
            except OSError:
                sig = None
        prev_sig = getattr(mod, "_or_credit_clamp_file_sig", None)
        if not force and sig is not None and prev_sig == sig:
            return False
        importlib.reload(mod)
        if sig is not None:
            mod._or_credit_clamp_file_sig = sig
        return True

    mod_file = tmp_path / "dummy_hotfix_mod.py"
    mod_file.write_text("VAL = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    d_mod = importlib.import_module("dummy_hotfix_mod")
    try:
        assert d_mod.VAL == 1

        # Pass 1: no _or_credit_clamp_file_sig yet -> reloads and stamps sig
        assert _reload_if_changed(d_mod) is True
        # Pass 2: unchanged (st_mtime_ns, st_size) -> skips reload (returns False)
        assert _reload_if_changed(d_mod) is False
        # Pass 3: force=True -> reloads even when unchanged
        assert _reload_if_changed(d_mod, force=True) is True
        # Pass 4: modify file on disk (size + mtime change) -> reloads and updates VAL
        mod_file.write_text("VAL = 222\n", encoding="utf-8")
        st = os.stat(mod_file)
        os.utime(mod_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        importlib.invalidate_caches()
        assert _reload_if_changed(d_mod) is True
        assert d_mod.VAL == 222
        assert _reload_if_changed(d_mod) is False
    finally:
        sys.modules.pop("dummy_hotfix_mod", None)


def test_topological_reload_and_enum_exception_singleton_preservation():
    """Reloading error_classifier, context_compressor, turn_context, hermes_state_rewind,
    and skills_tool_dedup preserves Enum/Exception class identity and tracker state."""
    import sys
    import agent.error_classifier as ec
    import agent.turn_context as tc
    import agent.turn_preflight as tp
    import agent.turn_recovery as tr
    import hermes_state_rewind as hsr
    import tools.skills_tool_dedup as std

    orig_fr = ec.FailoverReason
    orig_shh = cc_mod.StaleHeldHistory
    orig_pcto = tc.PreflightCompressionTimedOut
    orig_rtue = hsr.RewindTargetUnavailableError

    std._skill_view_tracker["test_cf79_sess"] = {("sk", ""): "digest1"}
    try:
        importlib.reload(ec)
        importlib.reload(cc_mod)
        importlib.reload(tc)
        importlib.reload(hsr)
        importlib.reload(std)
        importlib.reload(conv_comp)
        importlib.reload(tp)

        assert ec.FailoverReason is orig_fr
        assert ec.FailoverReason.rate_limit in tr._RATE_LIMIT_REASONS
        assert cc_mod.StaleHeldHistory is orig_shh
        assert tc.PreflightCompressionTimedOut is orig_pcto
        assert hsr.RewindTargetUnavailableError is orig_rtue
        assert "test_cf79_sess" in std._skill_view_tracker
        assert tp.request_exceeds_model_window is conv_comp.request_exceeds_model_window
        assert tp.compression_skipped_due_to_lock is conv_comp.compression_skipped_due_to_lock
        for m in (
            "agent.context_compressor",
            "agent.turn_truncation",
            "agent.turn_api_error",
            "agent.turn_response_check",
            "agent.agent_runtime_helpers",
            "agent.turn_recovery",
        ):
            if m in sys.modules:
                assert getattr(sys.modules[m], "FailoverReason", None) is ec.FailoverReason
    finally:
        std._skill_view_tracker.pop("test_cf79_sess", None)



