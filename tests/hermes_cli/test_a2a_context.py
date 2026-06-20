"""A2A 子agent 多轮对话核心(hermes_cli/a2a_context.py)单测。

不触真 LLM:注入 stub agent,只验安全攸关 + 易写错的管道:
  - F1 防劫持:contextId 客户端可填,不同 caller 同 ext_ctx → 不同内部会话;前缀隔离;稳定。
  - 多轮:第二轮能拿到第一轮的 transcript(回灌生效)。
  - source=a2a:会话被标记,便于从本机会话列表/搜索排除(不污染本机对话)。
  - F2:压缩换了 session_id 后,映射跟到新 tip。
"""

import sqlite3

import pytest

from hermes_cli import a2a_context as ac
from hermes_state import SessionDB


@pytest.fixture
def ctx_db(tmp_path):
    ac.set_context_db_path(str(tmp_path / "ctx.sqlite"))
    yield
    ac.set_context_db_path(ac._DEFAULT_DB_PATH)  # noqa: SLF001 — 复位,别留测试连接


@pytest.fixture
def sdb(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")  # SessionDB 要 Path,不是 str


class StubAgent:
    """假 agent:记录收到的 conversation_history,并像 run_conversation 那样把本轮落盘。"""

    instances: list = []

    def __init__(self, session_id, session_db):
        self.session_id = session_id
        self.session_db = session_db
        self.seen_history = "UNSET"
        StubAgent.instances.append(self)

    def run_conversation(self, message, conversation_history=None):
        self.seen_history = conversation_history
        self.session_db.append_message(self.session_id, "user", message)
        reply = f"echo:{message}"
        self.session_db.append_message(self.session_id, "assistant", reply)
        return {"final_response": reply, "messages": []}

    def close(self):
        pass


def _stub_builder(session_id, session_db):
    return StubAgent(session_id, session_db)


def test_contextid_no_hijack(ctx_db):
    # 同一个 ext contextId、不同 caller → 必须落到不同内部会话(防劫持)
    sid_boss = ac.resolve_or_create("boss", "ctxX")
    sid_intruder = ac.resolve_or_create("intruder", "ctxX")
    assert sid_boss != sid_intruder
    assert sid_boss.startswith("a2a:")  # 命名空间隔离本机会话
    # 稳定:同 (caller, ctx) 再解析得同一个
    assert ac.resolve_or_create("boss", "ctxX") == sid_boss


def test_f2_reanchor(ctx_db):
    ac.resolve_or_create("boss", "c1")
    ac.update_internal_id("boss", "c1", "a2a:child999")
    assert ac.current_internal_id("boss", "c1") == "a2a:child999"
    assert ac.resolve_or_create("boss", "c1") == "a2a:child999"  # 之后解析跟到新 tip


def test_multiturn_history_and_source(ctx_db, sdb, tmp_path):
    StubAgent.instances.clear()

    # 首轮:contextId 空 → server 生成并回传;无历史
    ans1, ctx = ac.run_turn("boss", "", "hello", db=sdb, build_agent=_stub_builder)
    assert ans1 == "echo:hello"
    assert ctx
    assert StubAgent.instances[0].seen_history in (None, [])

    # 次轮:带回 contextId → 应看到上一轮的 user("hello")+assistant
    ans2, ctx2 = ac.run_turn("boss", ctx, "again", db=sdb, build_agent=_stub_builder)
    assert ctx2 == ctx
    hist = StubAgent.instances[1].seen_history
    assert hist and len(hist) >= 2
    assert any(m.get("role") == "user" and m.get("content") == "hello" for m in hist)
    assert any(m.get("role") == "assistant" and m.get("content") == "echo:hello" for m in hist)

    # 不同 caller 用同一个 ctx 串不进来(防劫持端到端):它看到的是自己的空历史
    StubAgent.instances.clear()
    ac.run_turn("intruder", ctx, "whoami", db=sdb, build_agent=_stub_builder)
    assert StubAgent.instances[0].seen_history in (None, [])

    # source 标记 = a2a(供 /api/sessions exclude_sources 排除)
    sid = ac.current_internal_id("boss", ctx)
    raw = sqlite3.connect(str(tmp_path / "state.db"))
    try:
        src = raw.execute("SELECT source FROM sessions WHERE id=?", (sid,)).fetchone()
    finally:
        raw.close()
    assert src is not None and src[0] == "a2a"


# --------------------------- A2A-shaped 线协议 ---------------------------
def test_parse_message_send_ok():
    body = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "message/send",
        "params": {
            "message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}], "contextId": "c9"},
            "callerUid": "boss",
        },
    }
    p = ac.parse_message_send(body)
    assert p["rpc_id"] == 7 and p["text"] == "hi" and p["ext_context_id"] == "c9" and p["caller_uid"] == "boss"


def test_parse_message_send_rejects():
    with pytest.raises(ValueError):
        ac.parse_message_send({"method": "tasks/get"})  # 非 message/send
    with pytest.raises(ValueError):
        ac.parse_message_send({"method": "message/send", "params": {"message": {"parts": []}}})  # 无文本


def test_handle_a2a_error_paths():
    assert ac.handle_a2a_request({"id": 1, "method": "x"})["error"]["code"] == -32602
    # 有文本但缺 callerUid → 拒
    no_caller = {"id": 2, "method": "message/send", "params": {"message": {"parts": [{"kind": "text", "text": "hi"}]}}}
    assert ac.handle_a2a_request(no_caller)["error"]["code"] == -32602


def test_handle_a2a_happy(monkeypatch):
    monkeypatch.setattr(ac, "run_turn", lambda caller, ctx, text: (f"got:{text}", ctx or "newctx"))
    req = {
        "id": 3,
        "method": "message/send",
        "params": {"message": {"parts": [{"kind": "text", "text": "yo"}], "contextId": ""}, "callerUid": "boss"},
    }
    r = ac.handle_a2a_request(req)
    assert r["result"]["role"] == "agent"
    assert r["result"]["parts"][0]["text"] == "got:yo"
    assert r["result"]["contextId"] == "newctx"


def test_agent_card_shape():
    card = ac.build_agent_card("http://1.2.3.4:48901")
    assert card["url"] == "http://1.2.3.4:48901/a2a"
    assert card["securitySchemes"]["lanToken"]["name"] == "X-Kari-LAN-Token"
    assert card["skills"] and card["capabilities"]["streaming"] is False


# --------------------------- is_callable(F5)---------------------------
def test_is_callable(monkeypatch, tmp_path):
    monkeypatch.setenv("KARI_RESOURCES_DB", str(tmp_path / "res.sqlite"))
    from tools import kari_resources as kr

    kr.replace_node_resources("subA", "agent", [{"resource_id": "agent", "name": "子A"}])
    kr.add_grant("财务", "subA", "agent", "agent")
    assert kr.is_callable("anyuid", "财务", "subA", "agent", "agent") is True  # 角色授权 + 资源存活

    kr.add_user_grant("bob", "subA", "agent", "agent")
    assert kr.is_callable("bob", "无此角色", "subA", "agent", "agent") is True  # 按账号直授

    assert kr.is_callable("eve", "无此角色", "subA", "agent", "agent") is False  # 无授权

    kr.add_grant("财务", "subGhost", "agent", "agent")  # 授权在但没有该资源
    assert kr.is_callable("x", "财务", "subGhost", "agent", "agent") is False  # F5:资源不存活 → 拒
