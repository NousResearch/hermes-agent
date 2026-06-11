"""appraisal.py contract tests against a fake PluginLlm (zero network).

Fakes are built with the host's make_plugin_llm_for_test + _TrustPolicy;
the fake sync_caller returns the (provider, model, response) triple that
PluginLlm._invoke_sync expects, with an OpenAI-shaped response tree.
"""

import json
import time
import types

import pytest

plugin_llm = pytest.importorskip("agent.plugin_llm")

from hexis_appraisal import appraisal  # noqa: E402  (after importorskip)


def _cfg(**overrides):
    cfg = {
        "enabled": True,
        "confidence_threshold": 0.6,
        "deadline_seconds": 2.5,
        "history_chars": 4000,
        "model": None,
        "max_tokens": 700,
    }
    cfg.update(overrides)
    return cfg


def _response(text, prompt_tokens=120, completion_tokens=80):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="fake-model",
    )


def _make_llm(sync_caller, policy=None):
    return plugin_llm.make_plugin_llm_for_test(
        plugin_id="hexis_appraisal",
        policy=policy or plugin_llm._TrustPolicy(plugin_id="hexis_appraisal"),
        sync_caller=sync_caller,
    )


RICH_PAYLOAD = {
    "instincts": [
        {"kind": "caution", "intensity": 0.7, "reason": "deadline tension",
         "confidence": 0.8},
        {"kind": "curiosity", "intensity": 0.5, "reason": "new topic",
         "confidence": 0.7},
    ],
    "salient_observations": [
        {"text": "user switched projects mid-thread", "confidence": 0.9},
    ],
    "contradiction_flags": [
        {"kind": "semantic", "text": "says offline but asks for live calls",
         "confidence": 0.75},
    ],
    "suggested_memory_searches": ["project deadlines", "offline constraints"],
    "gut_reaction": "focused but slightly rushed",
}


@pytest.fixture(autouse=True)
def _fresh_executor():
    yield
    appraisal._reset_executor_for_tests()


# ---------------------------------------------------------------------------
# run_appraisal outcomes
# ---------------------------------------------------------------------------


def test_ok_path_parses_signals_and_records_usage():
    def caller(**_kwargs):
        return "fake", "fake-model", _response(json.dumps(RICH_PAYLOAD))

    result = appraisal.run_appraisal(
        llm=_make_llm(caller),
        user_message="how is the deadline looking?",
        conversation_history=[],
        snapshot=None,
        cfg=_cfg(),
    )
    assert result.outcome == "ok"
    assert result.error is None
    assert result.model == "fake-model"
    assert result.tokens_in == 120
    assert result.tokens_out == 80
    assert result.wall_ms >= 0
    assert len(result.signals["instincts"]) == 2
    assert result.signals["gut_reaction"] == "focused but slightly rushed"


def test_confidence_threshold_drops_low_signals():
    payload = {
        "salient_observations": [
            {"text": "kept", "confidence": 0.7},
            {"text": "dropped", "confidence": 0.5},
        ],
    }

    def caller(**_kwargs):
        return "fake", "fake-model", _response(json.dumps(payload))

    result = appraisal.run_appraisal(
        llm=_make_llm(caller), user_message="msg",
        conversation_history=[], snapshot=None, cfg=_cfg(),
    )
    texts = [o["text"] for o in result.signals["salient_observations"]]
    assert texts == ["kept"]

    # Custom threshold honored: at 0.4 both survive.
    result = appraisal.run_appraisal(
        llm=_make_llm(caller), user_message="msg",
        conversation_history=[], snapshot=None,
        cfg=_cfg(confidence_threshold=0.4),
    )
    texts = [o["text"] for o in result.signals["salient_observations"]]
    assert texts == ["kept", "dropped"]


def test_vocabulary_gating_and_clamping():
    # parse_signals is the defensive layer: exercised directly because the
    # host-side jsonschema validation (when installed) rejects these
    # documents before run_appraisal would ever hand them over.
    payload = {
        "instincts": [
            {"kind": "dominate", "intensity": 0.9, "reason": "x",
             "confidence": 0.9},  # unknown kind -> dropped
            {"kind": "caution", "intensity": 7.5, "reason": "y",
             "confidence": 1.8},  # floats clamped to [0,1]
        ],
        "contradiction_flags": [
            {"kind": "cosmic", "text": "x", "confidence": 0.9},  # dropped
            {"kind": "narrative", "text": "kept", "confidence": 0.8},
        ],
        "suggested_memory_searches": ["a", "b", "c", "d", "x" * 500],
        "gut_reaction": "g" * 999,
    }
    signals = appraisal.parse_signals(payload, 0.6)
    assert [i["kind"] for i in signals["instincts"]] == ["caution"]
    assert signals["instincts"][0]["intensity"] == 1.0
    assert signals["instincts"][0]["confidence"] == 1.0
    assert [c["kind"] for c in signals["contradiction_flags"]] == ["narrative"]
    assert signals["suggested_memory_searches"] == ["a", "b", "c"]  # <= 3
    assert len(signals["gut_reaction"]) == 200

    # Non-dict document -> empty signal set, full five-key shape.
    empty = appraisal.parse_signals(["not", "a", "dict"], 0.6)
    assert empty == {
        "instincts": [],
        "salient_observations": [],
        "contradiction_flags": [],
        "suggested_memory_searches": [],
        "gut_reaction": "",
    }


@pytest.mark.parametrize(
    "content",
    [
        "I cannot produce JSON for that, sorry.",  # non-JSON prose
        None,  # provider content: null under response_format
        json.dumps(["a", "list", "not", "a", "dict"]),  # list root
    ],
)
def test_parse_fail_paths(content):
    def caller(**_kwargs):
        return "fake", "fake-model", _response(content)

    result = appraisal.run_appraisal(
        llm=_make_llm(caller), user_message="msg",
        conversation_history=[], snapshot=None, cfg=_cfg(),
    )
    assert result.outcome == "parse_fail"
    assert result.signals is None


def test_timeout_binds_wall_clock():
    def caller(**_kwargs):
        time.sleep(1.0)
        return "fake", "fake-model", _response(json.dumps(RICH_PAYLOAD))

    start = time.monotonic()
    result = appraisal.run_appraisal(
        llm=_make_llm(caller), user_message="msg",
        conversation_history=[], snapshot=None,
        cfg=_cfg(deadline_seconds=0.2),
    )
    elapsed = time.monotonic() - start
    assert result.outcome == "timeout"
    assert result.signals is None
    assert elapsed < 0.6  # the deadline binds; the call is discarded
    appraisal._reset_executor_for_tests()


def test_trust_fallback_retries_once_without_override():
    calls = []

    def caller(**kwargs):
        calls.append(kwargs.get("model_override"))
        return "fake", "active-model", _response(json.dumps(RICH_PAYLOAD))

    # Restrictive policy: model override raises PluginLlmTrustError BEFORE
    # the caller is invoked; the fallback retry carries no override.
    result = appraisal.run_appraisal(
        llm=_make_llm(caller),
        user_message="msg",
        conversation_history=[],
        snapshot=None,
        cfg=_cfg(model="gpt-4o-mini"),
    )
    assert result.outcome == "trust_fallback"
    assert result.signals is not None
    assert calls == [None]  # gated attempt never reached the caller


def test_llm_error_captured_without_raising():
    def caller(**_kwargs):
        raise RuntimeError("provider exploded")

    result = appraisal.run_appraisal(
        llm=_make_llm(caller), user_message="msg",
        conversation_history=[], snapshot=None, cfg=_cfg(),
    )
    assert result.outcome == "llm_error"
    assert result.signals is None
    assert "provider exploded" in result.error


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def test_build_context_caps_history_keeping_the_end():
    history = [
        {"role": "user", "content": "early " + "x" * 5000},
        {"role": "assistant", "content": "late " + "y" * 200 + " TAIL_MARK"},
    ]
    context = appraisal.build_context("msg", history, None, 300)
    assert "TAIL_MARK" in context  # the END is kept
    assert "early" not in context  # the head is truncated away
    assert "no persisted state" in context


def test_build_context_excludes_sentinel_messages_and_handles_snapshot():
    history = [
        {"role": "system", "content": "[hexis appraisal]\n- instinct: caution"},
        {"role": "user", "content": "real message"},
    ]
    snapshot = {
        "concerns": [{"id": 1, "text": "open concern"}],
        "contradictions": [],
        "trust_scores": {"user:drmani": 0.9},
        "affect_summary": None,
        "turn_log_count": 3,
    }
    context = appraisal.build_context("msg", history, snapshot, 4000)
    assert "instinct: caution" not in context  # echo guard
    assert "real message" in context
    assert "open concern" in context
    assert "user:drmani" in context
    assert len(context) <= 12000


# ---------------------------------------------------------------------------
# Throttle helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "last_norm", "expected"),
    [
        ("", None, "empty"),
        ("   \n ", None, "empty"),
        ("ok", None, "social_close"),
        ("thanks", None, "social_close"),
        ("Deploy   the fix\n", "deploy the fix", "duplicate"),
        ("a genuinely novel question about the store", None, None),
        ("a genuinely novel question", "some other message", None),
    ],
)
def test_should_skip(message, last_norm, expected):
    assert appraisal.should_skip(message, last_norm) == expected


def test_normalize_message():
    assert appraisal.normalize_message("  Hello   WORLD \n") == "hello world"
