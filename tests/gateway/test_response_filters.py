from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_notification_dump,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


def test_translated_sentinel_is_silence_in_every_form_the_english_one_is():
    """#110935: a lane that answers the cron instruction in its own language translates the
    sentinel; ``[静默]`` must suppress delivery exactly like ``[SILENT]`` (exact, own-line note,
    reordered lines, bracketless, edge punctuation)."""
    assert is_intentional_silence_response("[静默]")
    assert is_intentional_silence_response("**沉默**")
    assert is_autonomous_silence_response("[静默]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[沉默]")
    assert is_autonomous_silence_response("静默")


def test_prose_mentioning_the_translated_sentinel_is_delivered():
    assert not is_intentional_silence_response("status: 静默 means the lane is quiet")
    assert not is_autonomous_silence_response("the lane said 静默 mid-sentence and kept talking")


def test_autonomous_lane_agrees_with_interactive_lane_on_cjk_punctuation_variants():
    """A Chinese lane emits fullwidth brackets or a trailing ``。``; cron/webhook must suppress
    exactly what the interactive predicate suppresses, or the two lanes drift on the new tokens."""
    for variant in ("【静默】", "静默。", "【沉默】", "沉默。", "**[静默]**", "NO_REPLY."):
        assert is_intentional_silence_response(variant)
        assert is_autonomous_silence_response(variant) == is_intentional_silence_response(variant), variant



# Notification-dump detection (incident: agent pasted its 11K-char
# notification backlog into a group chat).

def test_notification_dump_detects_supervisor_backlog_paste():
    sample = "\n".join(
        [
            "[九门_S01E12] ⏳ preprocess_asr_scene 已跑 5m1s，超阈值 5m0s，仍在继续",
            "[九门_S01E12] ⚠️ preprocess_asr_scene 跑过 1h15m0s，极大超时，自动修复桥唤起 K3 诊断",
            "[九门_S01E12] ✨ 修复完成，重跑管线（断点续跑）",
            "[九门_S01E12] ✅ curate done (49m20s)",
            "[九门_S01E12] 📦 任务失败，请人工介入",
        ]
    )
    assert is_notification_dump(sample)
    # CJK bracket variant
    assert is_notification_dump("【九门_S01E12】 ✅ accept done\n【九门_S01E12】 ▶️ 管线启动\n【九门_S01E12】 ❌ 任务失败")


def test_notification_dump_ignores_normal_replies():
    assert not is_notification_dump("curate 慢是因为 27B 精修逐帧打分，建议 M3 分担初筛。")
    # quoting 1-2 notification lines is not a dump
    assert not is_notification_dump("[九门_S01E12] ✅ curate done\n这条说明 curate 没问题。")
    assert not is_notification_dump("")
    assert not is_notification_dump(None)
    assert not is_notification_dump(123)
