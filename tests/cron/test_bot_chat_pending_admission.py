"""t_d29e1a7b — ``cron/bot_chat_pending`` 记录「存什么」口径 + admission_conflict 终局判定.

复现的病：同一 id 先 defer 正文、再 drain；旧 ``_dispose`` 会二次 ``bot_chat_message``
包装 → 门以同 id 不同 payload 拒投 ``admission_conflict`` → ``return False`` 留队，
每轮 cron 都重投一次，记录永远 ``queued``（376f9c… 从 2026-10-27 挂到 11-13）。

口径（本卡定，写进 bot_chat_legacy.py / kanban-botchat-deliver.py /
scheduler_delivery.py 三处注释）：**pending 记录的 content 就是最终要投递的正文**；
producer 在 defer 之前包装/排版恰好一次，defer 打上 ``content_format="final"``，
``_dispose`` 认标记、原样投递、绝不二次包装。无标记的是 retired CLI lane 历史记录
（裸 job output），_dispose 兼容地在投递那刻补一次 bot_chat_message。
"""
from __future__ import annotations

import json

from cron import bot_chat_legacy as legacy
from tools import bot_live_delivery as mailbox

KEY = "a" * 64
JOB = {"id": "kanban-botchat-deliver", "name": "机队看板终态回传"}
BODY = "Message from 🤖 franky (@franky): 【交接交付】\n• 任务: t_x\n• 结论: done"


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state.db").write_text("")
    legacy._warned.clear()
    return tmp_path


def _fake_door(monkeypatch, *, conflict: bool = False):
    calls: list[tuple] = []

    def deliver(home, owner, message, *, delivery_id, notification_category="result"):
        calls.append((delivery_id, message, notification_category))
        if conflict:
            raise RuntimeError("admission_conflict")
        return {"status": "queued", "message": message, "delivery_id": delivery_id}

    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", deliver)
    return calls


def _record(home):
    return json.loads((home / "cron" / "bot_chat_pending" / f"{KEY}.json").read_text(encoding="utf-8"))


def _seed_receipt(home, *, message: str, status: str = "queued"):
    root = home / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{KEY}.json").write_text(json.dumps({
        "delivery_id": KEY, "status": status, "admission_id": "f204783e5f4046caac2cbe0748133aee",
        "message": message, "author": {"id": "bot:franky", "is_bot": True, "name": "franky"},
    }), encoding="utf-8")


# ---------------------------------------------------------------- 主回归（要求 4）
def test_defer_then_drain_transfers_verbatim_without_admission_conflict(tmp_path, monkeypatch):
    """同一 id 先 defer 正文、再 drain → transferred，且不二次包装、不抛 admission_conflict。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch)

    legacy.defer(KEY, JOB, BODY, home, for_failure=False)
    assert _record(home)["content_format"] == "final"

    legacy.drain_legacy_pending()

    rec = _record(home)
    assert rec["status"] == "transferred", rec
    assert rec.get("error") is None
    assert calls == [(KEY, BODY, "result")], calls
    assert "Cronjob" not in calls[0][1], "正文被二次包装了"
    # 幂等：第二轮 drain 不再投递（记录已终态）
    legacy.drain_legacy_pending()
    assert len(calls) == 1


# ---------------------------------------------------------------- 要求 2 / 3
def test_same_id_receipt_with_identical_payload_transfers_without_resend(tmp_path, monkeypatch):
    """376f9c… 的形态：回执 message == record.content，已准入 FIFO → transferred，零重投。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch)
    legacy.defer(KEY, JOB, BODY, home, for_failure=False)
    _seed_receipt(home, message=BODY)

    legacy.drain_legacy_pending()

    rec = _record(home)
    assert rec["status"] == "transferred", rec
    assert rec["receipt_status"] == "queued"
    assert calls == [], "已有回执却还调门 → 又会给 admission_conflict 机会"


def test_same_id_receipt_with_different_payload_is_ambiguous_not_queued(tmp_path, monkeypatch):
    """同 id 已有回执且 payload 不一致 → ambiguous（保留证据、不再重投），不是 return False 留队。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch)
    legacy.defer(KEY, JOB, BODY, home, for_failure=False)
    _seed_receipt(home, message="a totally different payload")

    legacy.drain_legacy_pending()

    rec = _record(home)
    assert rec["status"] == "ambiguous", rec
    assert "admission_conflict" in (rec.get("error") or "")
    assert rec["receipt_status"] == "queued"
    assert rec.get("pending_message_sha256")
    assert rec.get("receipt_message_sha256")
    assert calls == []
    # 永久终局：再 drain 也不重投、不改写
    legacy.drain_legacy_pending()
    assert _record(home) == rec


def test_admission_conflict_from_door_is_terminal_ambiguous(tmp_path, monkeypatch):
    """门直接抛 admission_conflict（竞态/identity 不一致）→ 终局 ambiguous，不能当瞬时错误无限重试。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch, conflict=True)
    legacy.defer(KEY, JOB, BODY, home, for_failure=False)

    legacy.drain_legacy_pending()

    rec = _record(home)
    assert rec["status"] == "ambiguous", rec
    assert "admission_conflict" in (rec.get("error") or "")
    legacy.drain_legacy_pending()
    assert len(calls) == 1, "冲突被当瞬时错误重投了"


# ---------------------------------------------------------------- 兼容与守卫
def test_transient_dial_failure_stays_queued_for_next_tick(tmp_path, monkeypatch):
    """护栏：只有 admission_conflict 是终局；普通 dial/owner 错误仍是瞬时重试。"""
    home = _setup(tmp_path, monkeypatch)
    calls = []

    def deliver(home, owner, message, *, delivery_id, notification_category="result"):
        calls.append(delivery_id)
        raise RuntimeError("gateway disconnected; outcome unknown")

    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", deliver)
    legacy.defer(KEY, JOB, BODY, home, for_failure=False)

    legacy.drain_legacy_pending()
    assert _record(home)["status"] == "queued", "瞬时错误被误判成终局"
    legacy.drain_legacy_pending()
    assert len(calls) == 2, "瞬时错误应当允许下一轮重试"


def test_unmarked_retired_cli_record_is_wrapped_exactly_once(tmp_path, monkeypatch):
    """retired CLI lane 留下的无标记历史记录：仍是裸 output，_dispose 补一次包装。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch)
    root = home / "cron" / "bot_chat_pending"
    root.mkdir(parents=True)
    (root / f"{KEY}.json").write_text(json.dumps({
        "id": KEY, "status": "queued", "job": {"id": "j1", "name": "Digest"},
        "content": "the report", "home": str(home), "for_failure": False, "sequence": 1,
    }), encoding="utf-8")

    legacy.drain_legacy_pending()

    assert _record(home)["status"] == "transferred"
    assert 'Cronjob "Digest" output' in calls[0][1]
    assert "the report" in calls[0][1]
    assert len(calls) == 1


def test_unknown_content_format_is_refused_as_evidence(tmp_path, monkeypatch):
    """未知标记不猜、不投：判 ambiguous 留证（前置守卫）。"""
    home = _setup(tmp_path, monkeypatch)
    calls = _fake_door(monkeypatch)
    root = home / "cron" / "bot_chat_pending"
    root.mkdir(parents=True)
    (root / f"{KEY}.json").write_text(json.dumps({
        "id": KEY, "status": "queued", "job": JOB, "content": BODY, "home": str(home),
        "for_failure": False, "sequence": 1, "content_format": "v2-mystery",
    }), encoding="utf-8")

    legacy.drain_legacy_pending()

    rec = _record(home)
    assert rec["status"] == "ambiguous", rec
    assert "content_format" in (rec.get("error") or "")
    assert calls == []
