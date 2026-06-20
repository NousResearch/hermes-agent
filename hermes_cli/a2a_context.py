"""A2A 子agent 多轮对话:contextId↔内部会话映射 + run_turn 多轮引擎。

线协议照 A2A ``message/send`` + ``contextId`` 形状(见仓库根 easyhermes-subagent-chat-plan.md v3),
但**自维护、零 SDK**。HTTP 入口在 ``org_lan_server``;本模块只管两件事:

  1. **contextId 安全映射(F1 防劫持)**:A2A 的 ``contextId`` 是客户端可见且可伪填的,绝不能直接当
     ``SessionDB`` 的 session_id —— 否则填别人的 contextId 就能读/续他人会话。这里把
     ``(caller_uid, ext_context_id)`` 映射到一个 **server 自己生成、带 ``a2a:`` 前缀**的内部 session_id,
     与本机普通会话命名空间隔离;不同 caller 即便用同一个 ext_context_id 也落到不同内部会话。
  2. **run_turn(多轮一轮)**:回灌上一轮 transcript 跑一轮(``run_conversation(conversation_history=...)``),
     ``SessionDB`` 落盘;会话标 ``source="a2a"`` 便于从本机会话列表/搜索排除(不污染本机对话)。
     用**降权 agent**(``_build_a2a_agent``,F7):受限工具集 + 不绕审批,挡远程 RCE。

不在本模块碰网络/鉴权;主签票验证(Phase 3)在 org_lan_server 入口做,caller_uid 由那里认证后传入。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import uuid
from typing import Callable, Optional, Tuple

# SessionDB 的 source 标记:A2A 会话据此可被 /api/sessions 的 exclude_sources 排除,
# 不混进用户本机会话列表 / FTS 搜索(见 plan §7.1)。
A2A_SOURCE = "a2a"
# 内部 session_id 命名空间前缀,隔离本机普通会话(F1)。
_PREFIX = "a2a:"

# 受限工具集白名单(F7;最终清单待评审,见 plan 开放问题#2)。
# **只读 / 检索 / 规划类**;**不含** terminal / process / write_file / patch / delegate / execute_code,
# 也**不含 memory**(memory 工具支持 add/replace/remove 会写被叫方记忆——远程调用方不该能改;codex 评审 #2)。
# 故远程调用方没有危险/可写工具可触发;再叠加"不设 YOLO + 无交互审批回调 → 危险操作默认 deny"(approval.py)。
A2A_TOOLSETS = ["web", "search", "todo"]


class UnknownContextError(Exception):
    """客户端带了一个本调用方并不存在的 contextId(#6:server 拥有会话连续性,不凭空建会话)。"""


# --------------------------- contextId ↔ 内部 session 映射(F1 / F2)---------------------------
_DEFAULT_DB_PATH = os.path.expanduser("~/.hermes/kari_a2a_contexts.sqlite")
_db_path = os.environ.get("KARI_A2A_CONTEXT_DB") or _DEFAULT_DB_PATH
_conn: "sqlite3.Connection | None" = None
_conn_lock = threading.RLock()


def set_context_db_path(path: str) -> None:
    """切换映射表存储路径(测试隔离用)。会关掉旧连接。"""
    global _db_path, _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:  # noqa: BLE001
                pass
        _conn = None
        _db_path = path


def _db() -> "sqlite3.Connection":
    global _conn
    with _conn_lock:
        if _conn is None:
            d = os.path.dirname(_db_path)
            if d:
                os.makedirs(d, exist_ok=True)
            _conn = sqlite3.connect(_db_path, check_same_thread=False)
            _conn.execute(
                """CREATE TABLE IF NOT EXISTS a2a_context(
                       caller_uid TEXT NOT NULL,
                       ext_context_id TEXT NOT NULL,
                       internal_session_id TEXT NOT NULL,
                       created_ts REAL NOT NULL,
                       PRIMARY KEY(caller_uid, ext_context_id))"""
            )
            _conn.commit()
        return _conn


def new_context_id() -> str:
    """首轮 server 生成的 contextId(回传给客户端,后续轮带回来)。"""
    return uuid.uuid4().hex


def _gen_internal_id(caller_uid: str, ext_context_id: str) -> str:
    # server 拥有内部 id;掺 uuid → 客户端无法预测/碰撞他人会话。
    seed = f"{caller_uid}|{ext_context_id}|{uuid.uuid4().hex}".encode()
    return _PREFIX + hashlib.sha256(seed).hexdigest()[:24]


def resolve_or_create(caller_uid: str, ext_context_id: str) -> str:
    """把 (caller_uid, ext_context_id) 解析到内部 session_id;首见则新建一个 server-owned 的。"""
    with _conn_lock:
        db = _db()
        row = db.execute(
            "SELECT internal_session_id FROM a2a_context WHERE caller_uid=? AND ext_context_id=?",
            (caller_uid, ext_context_id),
        ).fetchone()
        if row:
            return row[0]
        sid = _gen_internal_id(caller_uid, ext_context_id)
        db.execute(
            "INSERT INTO a2a_context(caller_uid, ext_context_id, internal_session_id, created_ts) VALUES(?,?,?,?)",
            (caller_uid, ext_context_id, sid, time.time()),
        )
        db.commit()
        return sid


def current_internal_id(caller_uid: str, ext_context_id: str) -> Optional[str]:
    """读当前内部 session_id(不存在返回 None)。"""
    with _conn_lock:
        row = _db().execute(
            "SELECT internal_session_id FROM a2a_context WHERE caller_uid=? AND ext_context_id=?",
            (caller_uid, ext_context_id),
        ).fetchone()
        return row[0] if row else None


def update_internal_id(caller_uid: str, ext_context_id: str, new_internal_id: str) -> None:
    """F2:Hermes 上下文压缩会 end 旧 session 建子 session 并改 agent.session_id;每轮回写当前 tip,
    否则下一轮 reopen 到已被 end 的旧 id、丢历史。"""
    if not new_internal_id:
        return
    with _conn_lock:
        db = _db()
        db.execute(
            "UPDATE a2a_context SET internal_session_id=? WHERE caller_uid=? AND ext_context_id=?",
            (new_internal_id, caller_uid, ext_context_id),
        )
        db.commit()


# --------------------------- 同会话串行锁(按 caller|ext 稳定 key)---------------------------
_conv_locks: dict[str, threading.Lock] = {}
_conv_locks_guard = threading.Lock()


def _conversation_lock(caller_uid: str, ext_context_id: str) -> threading.Lock:
    key = f"{caller_uid}|{ext_context_id}"
    with _conv_locks_guard:
        return _conv_locks.setdefault(key, threading.Lock())


# --------------------------- 共享 SessionDB(单实例,单连接单锁)---------------------------
_session_db = None
_session_db_guard = threading.Lock()


def _shared_session_db():
    global _session_db
    with _session_db_guard:
        if _session_db is None:
            from hermes_state import SessionDB  # noqa: PLC0415 — 延迟导入,保持本模块 import 轻

            _session_db = SessionDB()
        return _session_db


# --------------------------- 降权 agent(F7)---------------------------
def _build_a2a_agent(session_id: str, session_db):
    """构造一个**受限**的 AIAgent 给 A2A 子agent作答(绑定到内部 session_id)。

    与 oneshot 的关键区别(F7):**不**设 HERMES_YOLO_MODE / HERMES_ACCEPT_HOOKS、**不**给全量 CLI 工具、
    **不**注册交互审批回调 —— 远程调用方既没有危险工具(白名单)又无法获批危险操作(默认 deny)。
    子用自己配置的默认模型作答(无需 oneshot 那套 --model 覆盖/别名解析)。
    """
    from hermes_cli.config import load_config  # noqa: PLC0415
    from hermes_cli.fallback_config import get_fallback_chain  # noqa: PLC0415
    from hermes_cli.runtime_provider import resolve_runtime_provider  # noqa: PLC0415
    from run_agent import AIAgent  # noqa: PLC0415

    cfg = load_config() or {}
    mcfg = cfg.get("model") or {}
    model = mcfg if isinstance(mcfg, str) else (mcfg.get("default") or mcfg.get("model") or "")
    runtime = resolve_runtime_provider(requested=None, target_model=model or None)
    fb = get_fallback_chain(cfg)

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        enabled_toolsets=list(A2A_TOOLSETS),  # 受限白名单(F7)
        quiet_mode=True,
        # platform="a2a":既是系统提示的平台 hint,又让**压缩生成的子会话** source 继承 "a2a"
        # (conversation_compression 用 source=agent.platform)——否则压缩后子会话 source="cli" 会漏进
        # 本机会话列表(codex 评审 #1)。降权仍靠 enabled_toolsets,不依赖 platform 名。
        platform="a2a",
        session_id=session_id,  # 绑定到我们的内部 id
        session_db=session_db,
        credential_pool=runtime.get("credential_pool"),
        fallback_model=fb or None,
        # 无 user → clarify 不挂起,叫它自行假设继续(不暴露危险默认行为)。
        clarify_callback=lambda q, choices=None: "[a2a: no interactive user; make a reasonable assumption and continue]",
    )
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    return agent


# --------------------------- run_turn:多轮一轮 ---------------------------
def run_turn(
    caller_uid: str,
    ext_context_id: str,
    user_message: str,
    *,
    db=None,
    build_agent: Optional[Callable] = None,
) -> Tuple[str, str]:
    """跑一轮 A2A 多轮对话,返回 ``(answer, ext_context_id)``。

    - ``ext_context_id`` 传空 → server 生成一个新的并回传(首轮)。
    - 多轮记忆:每轮把上一轮 transcript 经 ``conversation_history`` 回灌;持久化由 run_conversation 内部完成。
    - ``db`` / ``build_agent`` 可注入(测试用);默认共享 SessionDB + 降权 agent。
    """
    # 【#6】contextId 客户端可填:没带 → server 新建并回传;带了但本调用方无此会话 → 拒绝
    # (server 拥有会话连续性,不为陌生 contextId 凭空建会话)。
    if ext_context_id:
        if current_internal_id(caller_uid, ext_context_id) is None:
            raise UnknownContextError(ext_context_id)
    else:
        ext_context_id = new_context_id()
        resolve_or_create(caller_uid, ext_context_id)
    with _conversation_lock(caller_uid, ext_context_id):  # 同会话串行(F1 映射 + 历史一致)
        sid = current_internal_id(caller_uid, ext_context_id)
        if sid is None:  # 极罕见竞态(并发删了映射)→ 兜底重建
            sid = resolve_or_create(caller_uid, ext_context_id)
        database = db or _shared_session_db()
        database.create_session(sid, source=A2A_SOURCE)  # INSERT OR IGNORE,把 source 钉成 a2a
        database.reopen_session(sid)  # 续接(被 end 过的也能再收 turn)
        history = database.get_messages_as_conversation(sid)
        agent = (build_agent or _build_a2a_agent)(session_id=sid, session_db=database)
        try:
            result = agent.run_conversation(user_message, conversation_history=history or None)
            new_sid = getattr(agent, "session_id", sid) or sid
            if new_sid != sid:  # F2:压缩换了 session,回写 tip
                update_internal_id(caller_uid, ext_context_id, new_sid)
            return ((result or {}).get("final_response") or ""), ext_context_id
        finally:
            try:
                agent.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------- A2A-shaped 线协议(message/send,自维护无 SDK)---------------------------
def parse_message_send(body: dict) -> dict:
    """从 A2A-shaped ``message/send`` 请求体解析 {rpc_id, text, ext_context_id, caller_uid}。
    解析失败抛 ``ValueError``(HTTP 层转 JSON-RPC -32602)。"""
    if not isinstance(body, dict):
        raise ValueError("请求体必须是 JSON 对象")
    method = body.get("method")
    if method != "message/send":
        raise ValueError(f"不支持的 method:{method!r}(仅 message/send)")
    params = body.get("params") or {}
    msg = params.get("message") or {}
    parts = msg.get("parts") or []
    texts = [
        str(p.get("text"))
        for p in parts
        if isinstance(p, dict) and (p.get("kind") == "text" or p.get("type") == "text") and p.get("text")
    ]
    text = "\n".join(texts).strip()
    if not text:
        raise ValueError("message.parts 里没有文本")
    return {
        "rpc_id": body.get("id"),
        "text": text,
        "ext_context_id": str(msg.get("contextId") or "").strip(),
        "caller_uid": str(params.get("callerUid") or "").strip(),
    }


def make_message_response(rpc_id, answer: str, ext_context_id: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "kind": "message",
            "role": "agent",
            "messageId": new_context_id(),
            "parts": [{"kind": "text", "text": answer}],
            "contextId": ext_context_id,
        },
    }


def make_error_response(rpc_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def handle_a2a_request(body: dict, caller_uid_override: Optional[str] = None) -> dict:
    """``/a2a`` 入口逻辑(已过 token+私网门控后调用)。返回 A2A-shaped JSON-RPC dict。

    caller_uid:v1 取请求里的 ``callerUid``(同团队 token 即可信);Phase 3 验票后由 ``caller_uid_override``
    传入**验过的**身份,届时忽略 body 自报。
    """
    rpc_id = body.get("id") if isinstance(body, dict) else None
    try:
        parsed = parse_message_send(body)
    except ValueError as e:
        return make_error_response(rpc_id, -32602, str(e))
    caller_uid = (caller_uid_override or parsed["caller_uid"] or "").strip()
    if not caller_uid:
        return make_error_response(parsed["rpc_id"], -32602, "缺少 callerUid(调用方身份)")
    try:
        answer, ctx = run_turn(caller_uid, parsed["ext_context_id"], parsed["text"])
    except UnknownContextError as e:
        return make_error_response(parsed["rpc_id"], -32602, f"未知 contextId(无此会话):{e}")
    except Exception as e:  # noqa: BLE001
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).exception("a2a run_turn 失败")
        return make_error_response(parsed["rpc_id"], -32000, f"作答失败:{e}")
    return make_message_response(parsed["rpc_id"], answer, ctx)


# --------------------------- 请求鉴权(Phase 3:主签票 + 上级鉴权 + 防重放,codex #3)---------------------------
_seen_nonces: dict[str, float] = {}
_nonce_lock = threading.Lock()


_NONCE_MAX_TTL = 300.0  # nonce 记录存活上限(秒):封顶,防攻击者用超大 exp 撑爆内存


def _nonce_fresh(caller: str, nonce: str, exp) -> bool:
    """nonce 首次见 → True 并记下;再见/缺失 → False。按 caller 命名空间;存活封顶(不信票里的 exp)。"""
    if not nonce:
        return False
    key = f"{caller}|{nonce}"
    now = time.time()
    try:
        exp_f = float(exp)
    except (TypeError, ValueError):
        exp_f = 0.0
    store_exp = min(exp_f, now + _NONCE_MAX_TTL) if exp_f > now else now + _NONCE_MAX_TTL
    with _nonce_lock:
        for k, v in list(_seen_nonces.items()):
            if v < now:
                _seen_nonces.pop(k, None)
        if key in _seen_nonces:
            return False
        _seen_nonces[key] = store_exp
        return True


def authorize_request(body: dict, *, self_uid=None, fetch_pubkey=None, ancestors=None) -> Tuple[str, str]:
    """验 A2A 请求的**主签票**,返回 (verified_caller_uid, error)。error 非空 = 拒绝。

    团队 LAN 令牌只证「同队」,不证「这个 uid」;callerUid 自报可伪造。故 A2A 强制带 ticket:
      1. 取 ticket.caller 的**云端公钥**(同团队可取)验签(含 target=自身、exp);
      2. nonce 防重放;
      3. **caller 必须是本节点上级**(root_id/parent_id)——「上级可调子 agent」。
    身份用密码学验过的 ticket.caller,**不信 body 的 callerUid**。
    self_uid/fetch_pubkey/ancestors 可注入(测试);默认走 org_client + kari_identity。
    """
    from hermes_cli import kari_identity  # noqa: PLC0415

    params = ((body or {}).get("params") if isinstance(body, dict) else None) or {}
    ticket = params.get("ticket")
    if not isinstance(ticket, dict):
        return "", "缺少授权票(ticket):A2A 需主签票,团队令牌不足以证明调用方身份"
    tbody0 = ticket.get("body")
    if not isinstance(tbody0, dict):  # 畸形 body 早返回,别让 verify 前 .get 崩线程(codex minor)
        return "", "ticket.body 非法"
    if self_uid is None or fetch_pubkey is None or ancestors is None:
        from hermes_cli import org_client  # noqa: PLC0415

        if self_uid is None:
            self_uid = org_client.self_user_id()
        if fetch_pubkey is None:
            fetch_pubkey = org_client.fetch_pubkey
        if ancestors is None:
            ancestors = org_client.self_ancestors()
    if not self_uid:
        return "", "本节点未登录云端,无法验票"
    # 取**实际请求**的内容(核 mh:票必须绑定这条消息,防持票改内容重放)
    msg = params.get("message") or {}
    text = "\n".join(
        str(p.get("text"))
        for p in (msg.get("parts") or [])
        if isinstance(p, dict) and (p.get("kind") == "text" or p.get("type") == "text") and p.get("text")
    ).strip()
    ctx = str(msg.get("contextId") or "")
    claim = str(tbody0.get("caller") or "")

    def _pub(force: bool):
        if not claim:
            return None
        try:
            return fetch_pubkey(claim, force)
        except TypeError:  # 注入的 fetch_pubkey 可能只收 1 个参数
            return fetch_pubkey(claim)

    ok, tbody, reason = kari_identity.verify_ticket(ticket, _pub(False) or "", self_uid, ctx, text)
    if not ok and "签名" in reason:  # 公钥可能轮换/缓存陈旧 → 强制重取再验一次
        ok, tbody, reason = kari_identity.verify_ticket(ticket, _pub(True) or "", self_uid, ctx, text)
    if not ok:
        return "", f"票无效:{reason}"
    caller = str(tbody.get("caller") or "")
    # **先鉴权(必须是上级)再记 nonce** —— 否则非上级也能往 nonce 表灌(DoS,codex #2)。
    if caller not in (ancestors or set()):
        return "", "调用方非本节点上级,无权发起对话"
    if not _nonce_fresh(caller, tbody.get("nonce"), tbody.get("exp")):
        return "", "票已使用或缺 nonce(防重放)"
    return caller, ""


def build_agent_card(base_url: str = "") -> dict:
    """A2A Agent Card(挂 /.well-known/agent-card.json)。能力简介取自本机 gather_capabilities。"""
    try:
        from hermes_cli import org_client  # noqa: PLC0415

        cap = org_client.gather_capabilities() or {}
    except Exception:  # noqa: BLE001
        cap = {}
    summary = str(cap.get("summary") or cap.get("host") or "EasyHermes 子agent")
    return {
        "name": summary,
        "description": summary,
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "url": (base_url.rstrip("/") + "/a2a") if base_url else "/a2a",
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": "chat", "name": "多轮对话", "description": "跟该子agent多轮对话(受限工具集)", "tags": ["chat", "a2a"]}
        ],
        "securitySchemes": {"lanToken": {"type": "apiKey", "in": "header", "name": "X-Kari-LAN-Token"}},
        "security": [{"lanToken": []}],
    }
