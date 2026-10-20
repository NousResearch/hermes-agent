"""Tests for the composable output-guard pipeline (gateway/output_guards.py)."""

import asyncio
import os

import pytest

from gateway.output_guards import (
    GuardContext,
    GuardOutcome,
    OutputGuardPipeline,
    apply_output_guards,
    apply_output_guards_sync,
    get_default_pipeline,
    reset_default_pipeline,
)


@pytest.fixture(autouse=True)
def _reset_pipeline():
    """Isolate the process-wide guard pipeline + env flags for every test.

    The pipeline is a module-level singleton and the guard flags are read from
    env/config, so a test that enables an opt-in guard (em-dash stripping) can
    otherwise leak into unrelated tests that later import the same singleton.
    Clear env FIRST, then reset, on both setup and teardown so the ordering
    can't leave a rebuilt-but-dirty pipeline behind.
    """
    def _clean():
        for k in [k for k in os.environ if k.startswith("HERMES_GUARD_")]:
            del os.environ[k]
        reset_default_pipeline()

    _clean()
    yield
    _clean()


def _tg(**kw):
    kw.setdefault("platform", "telegram")
    kw.setdefault("is_final_response", True)
    return GuardContext(**kw)


# --- pipeline mechanics --------------------------------------------------

def test_empty_pipeline_returns_text_unchanged():
    p = OutputGuardPipeline()
    ctx = _tg()
    assert p.apply_sync("hello", ctx) == "hello"


def test_rewrite_threads_through_chain():
    p = OutputGuardPipeline()
    p.register("upper", lambda t, c: GuardOutcome(text=t.upper()))
    p.register("bang", lambda t, c: GuardOutcome(text=t + "!"))
    assert p.apply_sync("hi", _tg()) == "HI!"


def test_drop_short_circuits():
    seen = []
    p = OutputGuardPipeline()
    p.register("drop", lambda t, c: GuardOutcome(drop=True, reason="x"))
    p.register("after", lambda t, c: seen.append(t))
    assert p.apply_sync("hi", _tg()) is None
    assert seen == []  # second guard never ran


def test_none_outcome_is_noop():
    p = OutputGuardPipeline()
    p.register("noop", lambda t, c: None)
    assert p.apply_sync("hi", _tg()) == "hi"


def test_guard_exception_is_isolated():
    def boom(t, c):
        raise RuntimeError("kaboom")

    p = OutputGuardPipeline()
    p.register("boom", boom)
    p.register("ok", lambda t, c: GuardOutcome(text=t + "-ok"))
    # The raising guard is skipped; the chain continues.
    assert p.apply_sync("hi", _tg()) == "hi-ok"


def test_sync_skips_async_guards():
    async def aguard(t, c):
        return GuardOutcome(text="async-ran")

    p = OutputGuardPipeline()
    p.register("async", aguard)
    # apply_sync must not run async guards.
    assert p.apply_sync("orig", _tg()) == "orig"


def test_async_runs_all_guards():
    async def aguard(t, c):
        return GuardOutcome(text=t + "-async")

    p = OutputGuardPipeline()
    p.register("sync", lambda t, c: GuardOutcome(text=t + "-sync"))
    p.register("async", aguard)
    out = asyncio.run(p.apply("x", _tg()))
    assert out == "x-sync-async"


# --- built-in guards -----------------------------------------------------

def test_default_pipeline_order():
    names = get_default_pipeline().names()
    assert names == [
        "secret",
        "provider-error",
        "em-dash",
        "verify-links",
        "silence-narration",
    ]


def test_secret_redaction_all_platforms():
    # Secret redaction is not telegram-gated. Assert the behaviour contract
    # (the raw credential does not survive) rather than a specific mask
    # marker: the authoritative redactor in agent.redact uses "***" while the
    # gateway's belt-and-suspenders pass uses "[REDACTED]", and which one wins
    # is an upstream implementation detail.
    secret = "sk-" + "a" * 40
    ctx = GuardContext(platform="discord", is_final_response=True)
    out = apply_output_guards_sync(f"key {secret}", ctx)
    assert out is not None
    assert secret not in out


def test_provider_error_rewrite_all_chat_surfaces():
    # Upstream widened this from Telegram (#28533) to every chat surface
    # (#39293): a raw provider envelope can carry a leaked token, so no chat
    # platform should ever see it. Programmatic surfaces are excluded upstream
    # of the pipeline by _GATEWAY_RAW_TEXT_PLATFORMS, not by this guard.
    raw = "HTTP 401 incorrect api key"
    for platform in ("telegram", "discord", "whatsapp", "slack", "signal", "matrix"):
        out = apply_output_guards_sync(raw, GuardContext(platform=platform))
        assert out is not None
        assert "HTTP 401" not in out, f"{platform} leaked the raw provider envelope"
        assert "authentication failed" in out.lower()


def test_normal_text_untouched():
    assert apply_output_guards_sync("Here is your answer.", _tg()) == "Here is your answer."


def test_silence_narration_dropped():
    out = asyncio.run(apply_output_guards("*(silent)*", _tg()))
    assert out is None


def test_em_dash_guard_opt_in():
    text = "This is a test — with a dash."
    # Off by default: unchanged.
    assert apply_output_guards_sync(text, _tg()) == text
    # On via env.
    os.environ["HERMES_GUARD_STRIP_EM_DASHES"] = "1"
    reset_default_pipeline()
    out = apply_output_guards_sync(text, _tg())
    assert "\u2014" not in out
    assert "This is a test, with a dash." == out


def test_em_dash_guard_handles_en_dash():
    os.environ["HERMES_GUARD_STRIP_EM_DASHES"] = "1"
    reset_default_pipeline()
    out = apply_output_guards_sync("range 1\u20135 items", _tg())
    assert "\u2013" not in out
