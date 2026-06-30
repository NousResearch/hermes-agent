"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881): the historical ``codex_responses``-only path is byte-stable
under the default ``"auto"`` mode, while an explicit ``true``/model-list opt-in
extends the "you announced an action but called no tool — keep going" nudge to
every api_mode and relaxes the codebase/workspace requirement so general
autonomous workflows ("I'll run a health check on the server") are caught.

These are invariant assertions about how the mode string and the detector
gates relate, not snapshots of the marker lists.
"""

from types import SimpleNamespace
from typing import Union

from agent.agent_runtime_helpers import (
    intent_ack_continuation_enabled,
    intent_ack_continuation_mode,
    looks_like_codex_intermediate_ack,
)


def _agent(
    mode: Union[str, bool, list] = "auto",
    api_mode="chat_completions",
    model="anthropic/claude-sonnet-4",
):
    # _strip_think_blocks is a no-op for these plain-text fixtures.
    return SimpleNamespace(
        _intent_ack_continuation=mode,
        api_mode=api_mode,
        model=model,
        _strip_think_blocks=lambda c: c,
    )


# The reporter's exact repro (#27881): server-ops task, no filesystem reference.
REPRO_USER = (
    "check the current status of the server, grab the latest error logs, "
    "and let me know if there's anything critical"
)
REPRO_ACK = "I will start by running a health check command on the server to see its current status."

# The codex-coding case the detector was originally built for.
CODE_USER = "review the codebase in /app"
CODE_ACK = "Let me inspect the repository files first."


# ── mode resolution ────────────────────────────────────────────────────────


def test_auto_is_codex_only():
    assert intent_ack_continuation_mode(_agent("auto", "codex_responses")) == "codex_only"
    assert intent_ack_continuation_mode(_agent("auto", "chat_completions")) == "off"
    assert intent_ack_continuation_mode(_agent("auto", "anthropic")) == "off"


def test_true_is_all_api_modes():
    for am in ("chat_completions", "anthropic", "codex_responses"):
        assert intent_ack_continuation_mode(_agent(True, am)) == "all"
    for s in ("true", "always", "yes", "on", "ON"):
        assert intent_ack_continuation_mode(_agent(s, "chat_completions")) == "all"


def test_false_is_off_even_for_codex():
    assert intent_ack_continuation_mode(_agent(False, "codex_responses")) == "off"
    for s in ("false", "never", "no", "off"):
        assert intent_ack_continuation_mode(_agent(s, "codex_responses")) == "off"


def test_list_matches_model_substring():
    assert intent_ack_continuation_mode(
        _agent(["gemini", "qwen"], "chat_completions", "google/gemini-3-pro")
    ) == "all"
    assert intent_ack_continuation_mode(
        _agent(["gemini", "qwen"], "chat_completions", "anthropic/claude-sonnet-4")
    ) == "off"


def test_unrecognised_value_falls_back_to_auto():
    assert intent_ack_continuation_mode(_agent("garbage", "codex_responses")) == "codex_only"
    assert intent_ack_continuation_mode(_agent("garbage", "chat_completions")) == "off"


def test_missing_attr_defaults_to_auto():
    bare = SimpleNamespace(api_mode="chat_completions", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare) == "off"
    bare_codex = SimpleNamespace(api_mode="codex_responses", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare_codex) == "codex_only"


def test_enabled_is_mode_not_off():
    assert intent_ack_continuation_enabled(_agent(True, "chat_completions")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "codex_responses")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "chat_completions")) is False
    assert intent_ack_continuation_enabled(_agent(False, "codex_responses")) is False


# ── detector: workspace requirement ─────────────────────────────────────────


def test_codex_only_path_requires_workspace():
    a = _agent("auto", "codex_responses")
    msgs = [{"role": "user", "content": CODE_USER}]
    # codebase ack matches workspace markers → fires
    assert looks_like_codex_intermediate_ack(a, CODE_USER, CODE_ACK, msgs, require_workspace=True)
    # server-ops ack has no filesystem reference → does NOT fire (historical scope)
    repro_msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, repro_msgs, require_workspace=True
    )


def test_multipart_user_message_does_not_crash_on_workspace_path():
    """#9562: vision requests forward ``user_message`` as a multi-part list.

    The OpenAI-compat API server passes the raw ``content`` field straight
    through for vision turns, so ``user_message`` reaches the detector as
    ``[{type:"text",...}, {type:"image_url",...}]``. The ``require_workspace``
    path flattened it with ``(user_message or "").strip()`` — a truthy list
    survived and ``.strip()`` raised ``AttributeError``, killing the turn.
    The text part still has to drive workspace detection.
    """
    a = _agent("auto", "codex_responses")
    multipart = [
        {"type": "text", "text": CODE_USER},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    msgs = [{"role": "user", "content": multipart}]
    # No crash, and the text part ("review the codebase in /app") still
    # satisfies the workspace requirement so the ack fires.
    assert looks_like_codex_intermediate_ack(
        a, multipart, CODE_ACK, msgs, require_workspace=True
    )


def test_all_path_drops_workspace_requirement():
    """The #27881 fix: opted-in turns catch non-codebase intent acks."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


# ── detector: guardrails that hold regardless of workspace ───────────────────


def test_real_final_answer_does_not_fire():
    a = _agent(True, "chat_completions")
    final = "Done. The server is healthy and there are no critical errors in the logs."
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(a, REPRO_USER, final, msgs, require_workspace=False)


def test_conversational_reply_without_action_verb_does_not_fire():
    a = _agent(True, "chat_completions")
    brainstorm = "I'll help you think through the tradeoffs here."
    msgs = [{"role": "user", "content": "help me decide"}]
    assert not looks_like_codex_intermediate_ack(
        a, "help me decide", brainstorm, msgs, require_workspace=False
    )


def test_does_not_fire_after_a_tool_already_ran():
    a = _agent(True, "chat_completions")
    msgs = [
        {"role": "user", "content": REPRO_USER},
        {"role": "tool", "content": "health check result"},
    ]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


def test_long_response_is_not_treated_as_an_ack():
    a = _agent(True, "chat_completions")
    long_ack = "I will run the check. " + ("x" * 1300)
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, long_ack, msgs, require_workspace=False
    )


# ── detector: CJK acks (#55664) ──────────────────────────────────────────────


def test_chinese_ack_fires_under_all_api_mode():
    """#55664: the future-ack regex is Latin-only, so a Chinese ack like
    "我先載入 skill 確認用法" never matched and intent-ack continuation was a
    complete no-op for CJK users. Traditional and simplified variants both
    need to fire.
    """
    a = _agent(True, "chat_completions")
    user_msg = "幫我看一下伺服器狀態並抓最新的錯誤紀錄"
    traditional_ack = "我先載入 skill 確認用法，然後幫你檢查伺服器狀態。"
    simplified_ack = "我先加载 skill 确认用法，然后帮你检查服务器状态。"
    msgs = [{"role": "user", "content": user_msg}]
    assert looks_like_codex_intermediate_ack(
        a, user_msg, traditional_ack, msgs, require_workspace=False
    )
    assert looks_like_codex_intermediate_ack(
        a, user_msg, simplified_ack, msgs, require_workspace=False
    )


def test_chinese_final_answer_does_not_fire():
    """A real Chinese final answer (no future-ack phrase) must still not fire —
    the CJK addition only widens the future-ack gate, not the other guardrails.
    """
    a = _agent(True, "chat_completions")
    user_msg = "幫我看一下伺服器狀態"
    final = "伺服器狀態正常，沒有發現任何嚴重錯誤。"
    msgs = [{"role": "user", "content": user_msg}]
    assert not looks_like_codex_intermediate_ack(
        a, user_msg, final, msgs, require_workspace=False
    )
