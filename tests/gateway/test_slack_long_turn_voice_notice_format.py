from types import SimpleNamespace

import pytest

from gateway.run import (
    _format_slack_long_turn_voice_notice,
    _sanitize_slack_long_turn_ai_summary,
    _sanitize_slack_long_turn_next_clause,
    _slack_long_turn_link_label,
    _summarize_slack_long_turn_next_clause_ai,
    _summarize_slack_long_turn_request,
    _summarize_slack_long_turn_request_ai,
)


def test_slack_long_turn_notice_matches_javan_format():
    url = "https://javannetwork.slack.com/archives/C0ALSV2R23C/p1779067195283799?thread_ts=1779066938.907289&cid=C0ALSV2R23C"

    assert _format_slack_long_turn_voice_notice("車道自動化腳本修改", url) == (
        "Hi Javan，車道自動化腳本修改，已完成回覆產出，接下來請查看原對話串確認結果。\n"
        f"對話串: <{url}|{url}>"
    )


def test_slack_long_turn_notice_can_include_ai_next_step():
    url = "https://javannetwork.slack.com/archives/C0ALSV2R23C/p1779067195283799"

    assert _format_slack_long_turn_voice_notice(
        "PRD 與 EPIC 規劃調整",
        url,
        "建議繼續進行步驟二",
    ) == (
        "Hi Javan，PRD 與 EPIC 規劃調整，已完成回覆產出，接下來建議繼續進行步驟二。\n"
        f"對話串: <{url}|{url}>"
    )


def test_slack_long_turn_link_label_normalizes_existing_angle_link():
    url = "https://javannetwork.slack.com/archives/C0ALSV2R23C/p1779067195283799"

    assert _slack_long_turn_link_label(f"<{url}>") == f"<{url}|{url}>"
    assert _slack_long_turn_link_label(f"<{url}|old label>") == f"<{url}|{url}>"


def test_summarize_slack_long_turn_request_strips_thread_context_and_sender():
    raw = """[Replying to: \"old notice\"]
[Javan] [Thread context — prior messages in this thread (not yet in conversation history):]
[thread parent] Jarvis: old notice
[End of thread context]

[Javan] 車道自動化腳本修改，請幫我修一下
"""

    assert _summarize_slack_long_turn_request(raw) == "車道自動化腳本修改"


def test_summarize_slack_long_turn_request_turns_question_into_task_title():
    raw = "幫我查一下，我們從上星期五下午三點之後到現在，有做了哪些重要的事情。"

    assert _summarize_slack_long_turn_request(raw) == "上星期五下午三點之後到現在的重要事情匯整"


def test_summarize_slack_long_turn_request_skips_replying_to_quote():
    raw = '''[Replying to: "Hi Javan，幫我查一下，我們從上星期五下午三點之後到現在，有做了哪些重要的事情。，已完成回覆產出，接下來要進行原對話串結果確認。
對話串: <https://example.com|https://example.com>"]

[Javan] 原本這個傳送到 jarvis-speak 的回覆時間超過一分鐘的通知訊息，請幫我修改相關的指示詞或腳本。
'''

    assert _summarize_slack_long_turn_request(raw) == "長任務通知摘要格式修正"


def test_summarize_slack_long_turn_request_avoids_speaking_full_local_paths():
    raw = "[Javan] 這個 /Users/javanxu/Obsidian/Javan/WORK/YU/研究議題/案例整理.md，幫我整理"

    summary = _summarize_slack_long_turn_request(raw)

    assert "/Users/javanxu" not in summary
    assert summary == "案例整理.md 相關任務"


def test_summarize_slack_long_turn_request_semanticizes_gpt55_brief_issue():
    raw = "原訊息中提到的「gpt-5.5 brief 內容只有占位字樣」是什麼狀況? 是沒抓到資料嗎"

    assert _summarize_slack_long_turn_request(raw) == "gpt-5.5 brief 內容只有占位字樣的問題"


def test_summarize_slack_long_turn_request_semanticizes_unreal_ifc_memory_question():
    raw = "[Javan] Jarvis，你還記得我們之前在討論使用 Unreal Engine 來進行 IFC 4D 建築施工影片產生的事嗎?"

    summary = _summarize_slack_long_turn_request(raw)

    assert summary == "Unreal Engine IFC 4D 施工影片討論脈絡回顧"
    assert "Jarvis" not in summary
    assert "你還記得" not in summary
    assert not summary.endswith("…")


def test_sanitize_ai_summary_rejects_urls_and_completion_notice_fragments():
    raw = "幫我查一下，我們從上星期五下午三點之後到現在，有做了哪些重要的事情。"

    assert _sanitize_slack_long_turn_ai_summary(
        "對話串: <https://javannetwork.slack.com/archives/C0ALS…，已完成回覆產出",
        raw,
    ) == "上星期五下午三點之後到現在的重要事情匯整"


@pytest.mark.asyncio
async def test_ai_summary_path_uses_auxiliary_model_for_task_title(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="PRD 與 EPIC 規劃調整"))]
        )

    import agent.auxiliary_client as auxiliary_client

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    raw = "[Javan] 好的，請依照你的建議進行修改"
    summary = await _summarize_slack_long_turn_request_ai(raw)

    assert summary == "PRD 與 EPIC 規劃調整"
    assert calls
    assert calls[0]["task"] == "title_generation"
    assert calls[0]["max_tokens"] == 64
    assert "不要複製原句" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_ai_summary_path_falls_back_when_model_echoes_long_original(monkeypatch):
    def fake_call_llm(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="接下來，請幫我依照這份調整後的 2026-05-18_Word批註回覆與修正意見紀錄.md，幫我重新調整 PRD"
            ))]
        )

    import agent.auxiliary_client as auxiliary_client

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    raw = "[Javan] 接下來，請幫我依照這份調整後的 2026-05-18_Word批註回覆與修正意見紀錄.md，幫我重新調整 PRD / EPICS / 概略規劃原檔"
    summary = await _summarize_slack_long_turn_request_ai(raw)

    assert summary == "PRD 與 EPIC 規劃調整"
    assert not summary.endswith("…")


def test_sanitize_next_clause_rejects_links_and_notice_fragments():
    assert _sanitize_slack_long_turn_next_clause(
        "接下來 對話串: <https://javannetwork.slack.com/archives/C0ALS…>"
    ) == "請查看原對話串確認結果"
    assert _sanitize_slack_long_turn_next_clause("接下來建議繼續進行步驟二。") == "建議繼續進行步驟二"


@pytest.mark.asyncio
async def test_ai_next_clause_uses_auxiliary_model_for_followup(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="建議繼續進行步驟二"))]
        )

    import agent.auxiliary_client as auxiliary_client

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)
    clause = await _summarize_slack_long_turn_next_clause_ai(
        "請依照建議規劃",
        "已完成步驟一。下一步：建議繼續進行步驟二。",
    )

    assert clause == "建議繼續進行步驟二"
    assert calls
    assert calls[0]["task"] == "next_step_generation"
    assert "輸出接在『接下來』後面的短句" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_ai_next_clause_falls_back_without_final_response():
    assert await _summarize_slack_long_turn_next_clause_ai("anything", "") == "請查看原對話串確認結果"
