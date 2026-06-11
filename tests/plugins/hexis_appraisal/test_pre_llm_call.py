"""pre_llm_call hook contract tests — full path with a fake PluginLlm.

Drives the real registered hook through register(ctx) with a stub ctx.
State is redirected to tmp_path by monkeypatching store.get_db_path (the
deterministic route — get_db_path prefers the host's get_hermes_home, which
may ignore a late env change). Never writes to the real home.
"""

import json
import sqlite3
import types

import pytest

plugin_llm = pytest.importorskip("agent.plugin_llm")

import hexis_appraisal  # noqa: E402
from hexis_appraisal import appraisal, config, store  # noqa: E402


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


RICH_PAYLOAD = {
    "instincts": [
        {"kind": "caution", "intensity": 0.7, "reason": "deadline tension",
         "confidence": 0.8},
    ],
    "salient_observations": [
        {"text": "user switched projects mid-thread", "confidence": 0.9},
    ],
    "contradiction_flags": [],
    "suggested_memory_searches": ["prior deadline discussions"],
    "gut_reaction": "focused urgency",
    "confidence": 0.8,
}

EMPTY_PAYLOAD = {
    "instincts": [],
    "salient_observations": [],
    "contradiction_flags": [],
    "suggested_memory_searches": [],
    "gut_reaction": "",
    "confidence": 0.0,
}


class _CountingCaller:
    def __init__(self, payload):
        self.calls = 0
        self.payload = payload

    def __call__(self, *, messages, model_override=None, **kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return ("openai", model_override or "fake-model",
                _response(json.dumps(self.payload)))


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


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """Wire the hook to tmp state + a controllable cfg + a counting fake LLM."""
    db_path = tmp_path / "hexis_appraisal" / "state.db"
    monkeypatch.setattr(store, "get_db_path", lambda: db_path)
    assert store.ensure_db()

    state = {"cfg": _cfg(), "caller": _CountingCaller(RICH_PAYLOAD)}
    monkeypatch.setattr(config, "get_cfg", lambda force_reload=False: state["cfg"])

    def _delegating_caller(**kwargs):
        # late-bound so tests can swap state["caller"] after fixture setup
        return state["caller"](**kwargs)

    def _ctx():
        llm = plugin_llm.make_plugin_llm_for_test(
            plugin_id="hexis_appraisal",
            policy=plugin_llm._TrustPolicy(plugin_id="hexis_appraisal"),
            sync_caller=_delegating_caller,
        )
        return types.SimpleNamespace(llm=llm, register_hook=lambda *a, **k: None)

    hexis_appraisal.register(_ctx())
    hexis_appraisal._session_state.update(
        {"session_id": None, "last_msg_norm": None}
    )
    yield types.SimpleNamespace(state=state, db_path=db_path)
    appraisal._reset_executor_for_tests()
    hexis_appraisal._ctx = None


def _telemetry_rows(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return list(
            con.execute("SELECT outcome FROM telemetry ORDER BY id")
        )
    finally:
        con.close()


def test_kill_switch_skips_llm(hook_env):
    hook_env.state["cfg"] = _cfg(enabled=False)
    out = hexis_appraisal.pre_llm_call(session_id="s1", user_message="hello there")
    assert out is None
    assert _telemetry_rows(hook_env.db_path) == [("skipped:disabled",)]
    assert hook_env.state["caller"].calls == 0


def test_social_closer_skipped(hook_env):
    out = hexis_appraisal.pre_llm_call(session_id="s1", user_message="ok")
    assert out is None
    assert _telemetry_rows(hook_env.db_path) == [("skipped:social_close",)]
    assert hook_env.state["caller"].calls == 0


def test_duplicate_gate_within_session(hook_env):
    first = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="tell me about the deadline"
    )
    assert first is not None
    second = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="  Tell me about THE deadline "
    )
    assert second is None
    assert _telemetry_rows(hook_env.db_path)[-1] == ("skipped:duplicate",)
    assert hook_env.state["caller"].calls == 1


def test_new_session_resets_duplicate_gate(hook_env):
    hexis_appraisal.pre_llm_call(session_id="s1", user_message="same question")
    out = hexis_appraisal.pre_llm_call(session_id="s2", user_message="same question")
    assert out is not None
    assert hook_env.state["caller"].calls == 2


def test_empty_message_skipped(hook_env):
    out = hexis_appraisal.pre_llm_call(session_id="s1", user_message="   ")
    assert out is None
    assert _telemetry_rows(hook_env.db_path) == [("skipped:empty",)]


def test_happy_path_injects_block(hook_env):
    out = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="how is the migration going?"
    )
    assert isinstance(out, dict) and set(out) == {"context"}
    assert out["context"].startswith("[hexis appraisal]")
    con = sqlite3.connect(f"file:{hook_env.db_path}?mode=ro", uri=True)
    row = con.execute(
        "SELECT outcome, wall_ms, model, tokens_in, tokens_out FROM telemetry"
    ).fetchone()
    con.close()
    assert row[0] == "ok"
    assert row[1] is not None and row[2] == "fake-model"
    assert row[3] == 120 and row[4] == 80


def test_empty_signals_suppressed(hook_env):
    hook_env.state["caller"] = _CountingCaller(EMPTY_PAYLOAD)
    out = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="how is the migration going?"
    )
    assert out is None
    assert _telemetry_rows(hook_env.db_path) == [("ok",)]  # call happened


def test_llm_raise_fails_open(hook_env):
    hook_env.state["caller"] = _CountingCaller(RuntimeError("provider down"))
    out = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="how is the migration going?"
    )
    assert out is None
    rows = _telemetry_rows(hook_env.db_path)
    assert rows and rows[-1][0] == "llm_error"


def test_unwritable_telemetry_is_silent(hook_env, monkeypatch):
    monkeypatch.setattr(
        store, "get_db_path", lambda: hook_env.db_path / "nope" / "state.db"
    )
    out = hexis_appraisal.pre_llm_call(
        session_id="s1", user_message="how is the migration going?"
    )
    assert isinstance(out, dict)  # hook result unaffected by telemetry failure
