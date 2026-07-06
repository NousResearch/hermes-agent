"""Tests for the visual-guardrail plugin (Task C1 code backstop).

The plugin lives at ``~/.hermes/plugins/visual-guardrail/`` (a USER plugin,
outside the fork). These tests import its module directly by path so the
suite can run without depending on HERMES_HOME, then exercise:

  1. the SET hook (post_llm_call) arms the flag only when a visual was owed
     but not rendered, and only on telegram;
  2. the CONSUME hook (pre_llm_call) returns the nudge exactly once and
     clears the flag (no repeat-nudge on the following turn);
  3. the cross-turn handoff (SET turn N -> CONSUME turn N+1) end-to-end;
  4. fail-open + scope guards (non-telegram, empty text, rendered reply,
     exempt reply).

Run: ./venv/bin/python -m pytest tests/plugins/test_visual_guardrail.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the user plugin module directly from ~/.hermes/plugins/visual-guardrail.
# It vendors visual_trigger.py beside it; we load that first as a sibling so
# the package-relative ``from .visual_trigger import classify`` resolves.
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(
    os.environ.get("HERMES_HOME", Path.home() / ".hermes")
) / "plugins" / "visual-guardrail"


def _load_plugin():
    init = _PLUGIN_DIR / "__init__.py"
    if not init.exists():
        pytest.skip(f"visual-guardrail plugin not installed at {_PLUGIN_DIR}")
    # Register a package so the relative import in __init__.py resolves.
    pkg_name = "visual_guardrail_under_test"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init, submodule_search_locations=[str(_PLUGIN_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def vg():
    mod = _load_plugin()
    # Clear the in-memory armed store between tests for isolation.
    with mod._ARMED_LOCK:
        mod._ARMED.clear()
    yield mod
    with mod._ARMED_LOCK:
        mod._ARMED.clear()


# Reply text fixtures -------------------------------------------------------

# A too-wide table that owes a visual (T2), pure text (no MEDIA:, no image md).
_OWED_TABLE = (
    "Here is the detailed matrix that is too wide for Telegram text:\n\n"
    "| Case | Result | Correct | Owner | Risk | Note |\n"
    "|---|---|---|---|---|---|\n"
    "| healthy lead | silent | yes | app | low | ok |\n"
    "| flood event | lost | no | crm | high | the bug |\n"
    "| retry path | sent | yes | api | med | fixed |\n"
)

# Same table but WITH an adequate rendered PNG -> compliant, must NOT arm.
_RENDERED = _OWED_TABLE + "\nMEDIA:/tmp/vrp_table_1.png\n"

# Same table but WITH a text-card -> inadequate, must arm BAD_RENDER.
_BAD_RENDER = _OWED_TABLE + "\nMEDIA:/tmp/vr_card_1.png\n"

# A trivial short reply -> exempt, must NOT arm.
_TRIVIAL = "yes"


# --- SET hook (post_llm_call) ---------------------------------------------


def test_set_arms_when_visual_owed_and_not_rendered(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response=_OWED_TABLE, platform="telegram")
    with vg._ARMED_LOCK:
        assert vg._ARMED.get("s1") == "T2"


def test_set_does_not_arm_when_rendered(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response=_RENDERED, platform="telegram")
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_set_arms_bad_render_when_card_used_for_t2(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response=_BAD_RENDER, platform="telegram")
    with vg._ARMED_LOCK:
        assert vg._ARMED.get("s1") == "BAD_RENDER"
    out = vg._on_pre_llm_call(session_id="s1", platform="telegram")
    assert out is not None and "text-card" in out["context"]


def test_set_does_not_arm_on_exempt_reply(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response=_TRIVIAL, platform="telegram")
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_set_ignored_on_non_telegram(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response=_OWED_TABLE, platform="cli")
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_set_ignored_on_empty_response(vg):
    vg._on_post_llm_call(session_id="s1", assistant_response="   ", platform="telegram")
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_set_clears_stale_flag_when_next_reply_compliant(vg):
    # Arm, then a compliant reply on the same session must disarm.
    vg._on_post_llm_call(session_id="s1", assistant_response=_OWED_TABLE, platform="telegram")
    assert vg._ARMED.get("s1") == "T2"
    vg._on_post_llm_call(session_id="s1", assistant_response=_RENDERED, platform="telegram")
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


# --- CONSUME hook (pre_llm_call) ------------------------------------------


def test_consume_returns_note_when_armed_then_clears(vg):
    vg._arm("s1", "T2")
    out = vg._on_pre_llm_call(session_id="s1", platform="telegram")
    assert isinstance(out, dict) and "context" in out
    assert "T2" in out["context"]
    assert "render" in out["context"].lower()
    # Flag must be cleared after one consume — no repeat nudge.
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_consume_returns_none_when_not_armed(vg):
    assert vg._on_pre_llm_call(session_id="s1", platform="telegram") is None


def test_consume_returns_none_on_non_telegram_even_if_armed(vg):
    vg._arm("s1", "T2")
    assert vg._on_pre_llm_call(session_id="s1", platform="cli") is None
    # Flag should remain (it was a non-telegram turn; we never reached disarm path).
    with vg._ARMED_LOCK:
        assert "s1" in vg._ARMED


def test_consume_no_repeat_nudge_on_following_turn(vg):
    vg._arm("s1", "T2")
    first = vg._on_pre_llm_call(session_id="s1", platform="telegram")
    assert first is not None
    second = vg._on_pre_llm_call(session_id="s1", platform="telegram")
    assert second is None  # cleared after first consume


# --- End-to-end cross-turn handoff ----------------------------------------


def test_end_to_end_set_turn_n_consume_turn_n_plus_1(vg):
    # Turn N: assistant ships an owed-but-text reply.
    vg._on_post_llm_call(session_id="sX", assistant_response=_OWED_TABLE, platform="telegram")
    # Turn N+1 prologue: pre_llm_call injects the nudge.
    out = vg._on_pre_llm_call(session_id="sX", platform="telegram")
    assert out is not None and "context" in out
    # Turn N+2 prologue: no stale nudge.
    assert vg._on_pre_llm_call(session_id="sX", platform="telegram") is None


def test_per_session_isolation(vg):
    vg._on_post_llm_call(session_id="a", assistant_response=_OWED_TABLE, platform="telegram")
    # Different session must not see a's flag.
    assert vg._on_pre_llm_call(session_id="b", platform="telegram") is None
    # a still armed.
    assert vg._on_pre_llm_call(session_id="a", platform="telegram") is not None


def test_armed_store_bounded(vg):
    # Arm more than the cap; oldest should be evicted, store never exceeds cap.
    cap = vg._MAX_SESSIONS
    for i in range(cap + 50):
        vg._arm(f"s{i}", "T2")
    with vg._ARMED_LOCK:
        assert len(vg._ARMED) <= cap
        # The very first session should have been evicted.
        assert "s0" not in vg._ARMED


# --- Fail-open ------------------------------------------------------------


def test_set_swallows_exceptions(vg, monkeypatch):
    # Force the classifier to raise; the hook must not propagate.
    monkeypatch.setattr(vg, "_classify", lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    vg._on_post_llm_call(session_id="s1", assistant_response=_OWED_TABLE, platform="telegram")
    # No crash, nothing armed.
    with vg._ARMED_LOCK:
        assert "s1" not in vg._ARMED


def test_register_wires_both_hooks(vg):
    seen = {}

    class _Ctx:
        def register_hook(self, name, cb):
            seen[name] = cb

    vg.register(_Ctx())
    assert "post_llm_call" in seen
    assert "pre_llm_call" in seen


# --- E2E capstone: the injection channel both prior attempts got wrong -----


def test_e2e_note_rides_api_user_message_and_is_not_persisted(vg, monkeypatch):
    """DECISIVE regression for B1 (cache-bust) + B2 (silent strip).

    Drive a REAL AIAgent.run_conversation with a fake client capturing the
    `messages` arg to chat.completions.create. A pre_llm_call hook returns a
    sentinel context. Assert:
      (a) the sentinel IS in the API USER message  (model sees it),
      (b) the sentinel is NOT in the API SYSTEM message  (B1 impossible),
      (c) the persisted user turn is CLEAN  (B2 / leak impossible).

    This is the contract the plugin depends on; if the host injection seam
    ever moves back into the system prompt or into persist_user_message,
    this test fails loudly.
    """
    from unittest.mock import MagicMock, patch as _patch

    SENTINEL = "ZZ_VG_E2E_SENTINEL_ZZ"
    captured = {}

    def _mock_response(content="ok"):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        msg.reasoning = None
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        return resp

    def _capture_create(*a, **kw):
        captured["messages"] = kw.get("messages")
        return _mock_response("done")

    try:
        from run_agent import AIAgent
        import hermes_cli.plugins as plugins_mod
    except Exception:
        pytest.skip("run_agent / plugins not importable in this environment")

    with (
        _patch("run_agent.get_tool_definitions", return_value=[]),
        _patch("run_agent.check_toolset_requirements", return_value={}),
        _patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            provider="openai",
            api_mode="chat_completions",
            model="gpt-5.5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = _capture_create
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.platform = "telegram"

    mgr = plugins_mod.get_plugin_manager()
    hook = lambda **kw: {"context": SENTINEL}
    mgr._hooks.setdefault("pre_llm_call", []).append(hook)

    persisted = {}

    def _spy_persist(messages, *a, **k):
        for m in messages:
            if m.get("role") == "user":
                persisted["last_user"] = m.get("content")
        return None

    try:
        with (
            _patch.object(agent, "_persist_session", side_effect=_spy_persist),
            _patch.object(agent, "_save_trajectory"),
            _patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("show me the four options")
    finally:
        # Clean the global hook list so other tests aren't polluted.
        try:
            mgr._hooks.get("pre_llm_call", []).remove(hook)
        except ValueError:
            pass

    api_msgs = captured.get("messages") or []
    assert agent.client.chat.completions.create.called, "never reached the API call"

    def _content(role):
        m = next((x for x in api_msgs if x.get("role") == role), None)
        c = (m or {}).get("content", "")
        return str(c) if not isinstance(c, str) else c

    usr = _content("user")
    sysc = _content("system")

    # (a) model sees the note
    assert SENTINEL in usr, "note did NOT ride the API user message"
    # (b) B1 — never in the system prompt (cache-safe)
    assert SENTINEL not in sysc, "note leaked into system prompt (cache-bust regression)"
    # (c) B2 — persisted transcript stays clean
    assert SENTINEL not in (persisted.get("last_user") or ""), "note leaked into persisted transcript"


# --- Vendored-classifier drift guard --------------------------------------


def test_vendored_classifier_matches_source_of_truth():
    """The plugin's vendored visual_trigger.py must stay byte-identical to the
    source of truth at ~/.hermes/scripts/visual_trigger.py.

    Two copies exist by design (the plugin vendors the classifier so it is
    self-contained and survives `hermes update`). If they diverge, the live
    guardrail would nudge on a different rule than the daily audit measures —
    a silent correctness bug. This test makes that drift fail loudly at
    pre-push time. Skips when the install copies aren't present (CI / temp
    HERMES_HOME), since it guards the real environment, not a fixture.
    """
    import hashlib

    # conftest.py forces HERMES_HOME to a temp dir for hermetic tests, but this
    # guard must check the REAL install copies, not the sandbox. _PLUGIN_DIR is
    # already resolved at import time (before conftest's autouse fixture runs),
    # so derive the source path from its real grandparent (~/.hermes).
    real_home = _PLUGIN_DIR.parent.parent  # .../.hermes/plugins/visual-guardrail -> .hermes
    source = real_home / "scripts" / "visual_trigger.py"
    plugin = _PLUGIN_DIR / "visual_trigger.py"
    if not source.exists() or not plugin.exists():
        pytest.skip("visual_trigger source/plugin copies not both present in this env")

    src_h = hashlib.sha256(source.read_bytes()).hexdigest()
    plg_h = hashlib.sha256(plugin.read_bytes()).hexdigest()
    assert src_h == plg_h, (
        "visual_trigger.py DRIFT: plugin copy != source of truth.\n"
        f"  source {source} sha256={src_h}\n"
        f"  plugin {plugin} sha256={plg_h}\n"
        f"  Resync: cp '{source}' '{plugin}' (then commit both repos)."
    )
