"""组织协同(本地侧)—— 子爱马仕上报能力 + 应答主账号查询;主账号扇出查询并收齐。

建在 kari-cloud 的 ``/api/v1/kari/org/*`` 与 ``/api/v1/kari/tasks/*`` 之上。云端地址 + token
复用 ``~/.hermes/workflow-secrets.json`` 里的 ``kari.{token, cloudBaseURL}``(与工作流登录同一份),
也可用 env ``KARI_HUB_URL`` / ``KARI_WORKSPACE_TOKEN`` 覆盖。

子侧(每个本地爱马仕实例):
  - ``report_capabilities()``  把本机能力(模型/工具/MCP/简介)上报管理端。
  - ``poll_and_answer_once()`` 取走分发来的 ``ask`` 任务,用本地爱马仕作答(oneshot agent)再回报。
  - ``run_responder()``        后台循环(定期上报 + 轮询应答),由 hermes dashboard 生命周期拉起。

主侧(``kari_org_mcp`` 暴露给主账号 agent):
  - ``subtree_capabilities()`` 看整棵子树上报的能力(读下级)。
  - ``query_subordinates()``   扇出查询给下级 → 收齐回报(主账号 agent 据此 + 自身知识库汇总)。
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def _read_json(resp) -> object:
    """读 urllib 响应体并解析 JSON。langflow 对大响应(尤其 flows 列表)会 **gzip**(即使没显式要求),
    按 gzip 魔数 1f8b 解压再 parse —— 否则裸 json.loads 撞上 gzip 字节直接抛、整条采集静默变 None。
    空响应体 → None(交调用方按"读失败/跳过"处理,不误降级成 [] 清缓冲)。"""
    raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw or b"null")

_CAP_PATH = "/api/v1/kari/org/capabilities"
_ASK_PATH = "/api/v1/kari/org/ask"
_COLLECT_PATH = "/api/v1/kari/org/collect"
_POLL_PATH = "/api/v1/kari/tasks/poll"
_RESOURCES_PATH = "/api/v1/kari/org/resources"
_RESOURCES_ACK_PATH = "/api/v1/kari/org/resources/ack"


def _cloud() -> tuple[Optional[str], Optional[str]]:
    """(base_url, token):优先 env,回退 workflow-secrets.json 里的 kari.*。"""
    base = (os.environ.get("KARI_HUB_URL") or "").strip()
    token = (os.environ.get("KARI_WORKSPACE_TOKEN") or "").strip()
    if base and token:
        return base.rstrip("/"), token
    try:
        from hermes_cli.workflow_backend import read_secrets

        kari = (read_secrets().get("kari") or {}) if callable(read_secrets) else {}
    except Exception:  # noqa: BLE001
        kari = {}
    base = base or str(kari.get("cloudBaseURL") or kari.get("cloudBaseUrl") or "").strip()
    token = token or str(kari.get("token") or "").strip()
    return (base.rstrip("/") if base else None), (token or None)


def _call(method: str, path: str, token: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    base, tok = (_cloud()[0], token)
    if not base:
        return 0, {"error": "未登录云端(无 kari token / cloudBaseURL)"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"content-type": "application/json", "X-Kari-Workspace-Token": tok},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or "{}")
        except Exception:  # noqa: BLE001
            return e.code, {"error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


# --------------------------- 能力收集 + 上报(子侧)---------------------------
def gather_capabilities() -> dict:
    """采集本机爱马仕能力(尽力而为,任一项失败都不影响整体)。"""
    cap: dict = {"host": socket.gethostname()}
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    try:
        m = cfg.get("model")
        cap["model"] = (m.get("default") or m.get("model")) if isinstance(m, dict) else m
    except Exception:  # noqa: BLE001
        pass
    try:
        from hermes_cli.tools_config import _get_platform_tools

        cap["toolsets"] = sorted(_get_platform_tools(cfg, "cli"))
    except Exception:  # noqa: BLE001
        pass
    try:
        mcp = cfg.get("mcp_servers")
        if isinstance(mcp, dict):
            cap["mcp_servers"] = sorted(str(k) for k in mcp.keys())
    except Exception:  # noqa: BLE001
        pass
    # 「能力承载节点」标志:有 langflow → 能当 MCP 提供者给下级配工作流(MCP 授权设计 slice A)。
    try:
        from hermes_cli.workflow_backend import langflow_capable

        cap["langflow_capable"] = bool(langflow_capable())
    except Exception:  # noqa: BLE001
        cap["langflow_capable"] = False
    # 用户可在 config.yaml 写 kari.capability_summary 或用 env 给一句话简介(主账号据此挑选问谁)。
    summary = (os.environ.get("KARI_CAPABILITY_SUMMARY") or "").strip()
    if not summary:
        try:
            kari_cfg = cfg.get("kari")
            if isinstance(kari_cfg, dict):
                summary = str(kari_cfg.get("capability_summary") or "").strip()
        except Exception:  # noqa: BLE001
            summary = ""
    cap["summary"] = summary or "爱马仕本地实例"
    # A2A 验签公钥(Ed25519,公开信息):上报后同团队节点可据此验本机签的票(Phase 3 身份)。
    try:
        from hermes_cli import kari_identity  # noqa: PLC0415

        cap["pubkey"] = kari_identity.public_key_b64()
    except Exception:  # noqa: BLE001
        pass
    return cap


def report_capabilities(extra: dict | None = None) -> tuple[bool, dict]:
    """把本机能力上报到管理端。extra 可覆盖/补充 gather_capabilities 的字段。"""
    base, token = _cloud()
    if not (base and token):
        return False, {"error": "未登录云端"}
    data = gather_capabilities()
    if extra:
        data.update(extra)
    st, r = _call("POST", _CAP_PATH, token, {"data": data})
    return st == 200 and bool(r.get("ok")), r


# --------------------------- 资源目录上报(子侧,store-and-forward / Phase 1b)---------------------------
def report_resources(kind: str, items: list[dict]) -> tuple[bool, dict]:
    """把某 kind(knowledge/workflow/agent)的资源目录上报到云缓冲(覆盖式)。
    items = [{resource_id, name?, meta?}]。主账号稍后拉走存进**主本地**注册表。
    业务(扫知识库建目录)在 1c;这里只是传输层。"""
    base, token = _cloud()
    if not (base and token):
        return False, {"error": "未登录云端"}
    st, r = _call("POST", _RESOURCES_PATH, token, {"kind": kind, "items": items or []})
    return st == 200 and bool(r.get("ok")), r


# --------------------------- 自身 uid + 自有资源也登记本地注册表(Phase 4)---------------------------
_ACCOUNT_ME_PATH = "/account/me"
_self_uid_cache: dict[str, str] = {}  # token → uid(登录态不变,缓存避免每轮上报都打 /account/me)
_self_root_cache: dict[str, str] = {}  # token → root_id(主账号 uid;3b 直连用)
_PUBKEY_PATH = "/api/v1/kari/org/pubkey/"  # + uid:取同团队某节点的 A2A 验签公钥
_PUBKEY_TTL = 3600.0  # 公钥缓存 TTL(秒):轮换/失效在 TTL 内有窗口,验签失败方会强制重取(force)
_pubkey_cache: dict[tuple, tuple] = {}  # (token, uid) → (pubkey, fetched_ts);按 token 命名空间防串号


def self_user_id() -> Optional[str]:
    """本节点自身账号 uid。用 org 同一份 workspace token 调 /account/me 解析,按 token 缓存。

    用途:把**自有资源**(自己的知识库/工作流,含 copilot 刚创作注册的)也登进本地注册表
    —— 否则注册表只有「拉来的下级资源」,主账号授权不到自己的东西。拿不到 → None(跳过登记)。
    """
    base, token = _cloud()
    if not (base and token):
        return None
    cached = _self_uid_cache.get(token)
    if cached:
        return cached
    st, r = _call("GET", _ACCOUNT_ME_PATH, token)
    uid = str(r.get("user_id") or "").strip() if st == 200 else ""
    if uid:
        _self_uid_cache[token] = uid
    return uid or None


def fetch_pubkey(uid: str, force: bool = False) -> Optional[str]:
    """取某节点(同团队)的 A2A 验签公钥(云端下发,带 TTL 缓存,按 token 命名空间)。

    A2A 子验主签票用。``force=True`` 跳过缓存强制重取(验签失败时用,应对公钥轮换/陈旧缓存)。
    云端临时不可达时回退未过期旧缓存(保可用);否则 None(fail-closed)。
    """
    uid = str(uid or "").strip()
    if not uid:
        return None
    base, token = _cloud()
    if not (base and token):
        return None
    key = (token, uid)
    now = time.time()
    if not force:
        ent = _pubkey_cache.get(key)
        if ent and (now - ent[1]) < _PUBKEY_TTL:
            return ent[0]
    st, r = _call("GET", _PUBKEY_PATH + uid, token)
    pk = str(r.get("pubkey") or "").strip() if st == 200 else ""
    if pk:
        _pubkey_cache[key] = (pk, now)
        return pk
    ent = _pubkey_cache.get(key)  # 取不到:回退未过期旧缓存(云抖动别立刻失效)
    return ent[0] if ent and (now - ent[1]) < _PUBKEY_TTL else None


def self_ancestors() -> set:
    """本节点的上级集合(root_id + parent_id)。A2A「只允许上级发起对话」鉴权用。未登录 → 空集。"""
    out: set = set()
    base, token = _cloud()
    if not (base and token):
        return out
    st, r = _call("GET", _ACCOUNT_ME_PATH, token)
    if st == 200:
        for k in ("root_id", "parent_id"):
            v = str(r.get(k) or "").strip()
            if v:
                out.add(v)
    return out


def _store_own_resources(kind: str, items: "list[dict] | None") -> None:
    """把本节点自己 gather 出来的某 kind 资源整体登进**本地**注册表(node_uid=自身 uid)。

    与「拉下级上报」并存:本地注册表 = 自有(直接登)+ 下级(经云拉)。items 为 None(读失败)
    不动;拿不到自身 uid 也跳过。失败只记日志不影响上报主流程。
    """
    if items is None:
        return
    uid = self_user_id()
    if not uid:
        return
    try:
        from tools import kari_resources  # noqa: PLC0415

        kari_resources.replace_node_resources(uid, kind, items)
    except Exception as e:  # noqa: BLE001
        logger.warning("登记自有 %s 资源到本地注册表失败:%s", kind, e)


# --------------------------- 3b 局域网直连:model-B 弱子把原文交给主入库(子侧)---------------------------
def _main_root_uid() -> Optional[str]:
    """本节点「主账号」uid(账号树根)。override: env ``KARI_MAIN_UID`` / 配置 ``kari.main_uid``;
    否则取 /account/me 的 ``root_id``(缓存)。拿不到 → None。"""
    ov = (os.environ.get("KARI_MAIN_UID") or "").strip()
    if ov:
        return ov
    base, token = _cloud()
    if not (base and token):
        return None
    cached = _self_root_cache.get(token)
    if cached is not None:
        return cached or None
    st, r = _call("GET", _ACCOUNT_ME_PATH, token)
    root = str(r.get("root_id") or "").strip() if st == 200 else ""
    if not root:
        try:
            from hermes_cli.workflow_backend import read_secrets  # noqa: PLC0415

            kari = (read_secrets().get("kari") or {}) if callable(read_secrets) else {}
            root = str(kari.get("main_uid") or "").strip()
        except Exception:  # noqa: BLE001
            root = ""
    _self_root_cache[token] = root
    return root or None


def _peer_addr(uid: str) -> "tuple[str, int] | None":
    """从发现 peer 表查某 uid 当前 ``ip`` + org 服务端口(广播带 org_port,缺省回退默认端口)。"""
    try:
        from hermes_cli import lan_discovery, org_lan_server  # noqa: PLC0415

        for p in lan_discovery.peers():
            if p.get("uid") == uid:
                ip = str(p.get("ip") or "").strip()
                port = int(p.get("org_port") or org_lan_server.ORG_LAN_PORT)
                if ip:
                    return ip, port
    except Exception:  # noqa: BLE001
        pass
    return None


def lan_kb_ingest(kb_name: str, files: list, target_uid: Optional[str] = None, timeout: float = 120.0) -> dict:
    """弱子把知识库**原文**发给主做 embedding+入库(model-B)。子自己不跑 langflow。

    ``files`` = ``[{"rel": 相对路径, "text": 正文}]``。``target_uid`` 默认 = 主账号(树根);
    经发现 peer 表查它当前 ``ip:org_port`` → 带团队 LAN 令牌 POST。返回主的 JSON 或 ``{ok:False,error}``。
    """
    target = target_uid or _main_root_uid()
    if not target:
        return {"ok": False, "error": "解析不到主账号 uid(未登录 / 无 root_id)"}
    addr = _peer_addr(target)
    if not addr:
        return {"ok": False, "error": f"发现表里没有主账号 {str(target)[:8]} 的在线地址(对方没在局域网广播?)"}
    ip, port = addr
    try:
        from hermes_cli import org_lan_server  # noqa: PLC0415

        tok = org_lan_server._lan_token()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        tok = ""
    if not tok:
        return {"ok": False, "error": "本机未配置 kari.lan_token,无法直连主账号"}
    try:
        import httpx  # noqa: PLC0415

        r = httpx.post(
            f"http://{ip}:{port}/kb/ingest",
            json={"owner_uid": self_user_id() or "", "kb_name": kb_name, "files": list(files or [])},
            headers={"X-Kari-LAN-Token": tok, "content-type": "application/json"},
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"连主账号 {ip}:{port} 失败:{e}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"主返回 {r.status_code}:{r.text[:200]}"}
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "主返回非 JSON"}


# --------------------------- LAN 直连多轮对话:主→子(A2A-shaped message/send)---------------------------
def lan_agent_chat(target_uid: str, message: str, context_id: str = "", timeout: float = 180.0) -> dict:
    """主账号 LAN 直连某下级子爱马仕**多轮对话**(对应子侧 org_lan_server 的 ``/a2a``)。

    与云端 ``delegate_to``(单次、经云中继)不同:这里走局域网直连、保持会话——把上次返回的
    ``context_id`` 带回来即可续聊(下级按它续 SessionDB 会话)。返回
    ``{ok, answer, context_id}`` 或 ``{ok:False,error}``。对方不在局域网/未配 LAN 令牌 → ok=False。
    线格式照 A2A ``message/send``;身份(callerUid)v1 = 本机 uid(Phase 3 换主签票)。
    """
    target_uid = str(target_uid or "").strip()
    if not (target_uid and (message or "").strip()):
        return {"ok": False, "error": "需要 target_uid 和非空 message"}
    addr = _peer_addr(target_uid)
    if not addr:
        return {"ok": False, "error": f"发现表里没有 {target_uid[:8]} 的在线地址(对方在同一局域网?)"}
    ip, port = addr
    try:
        from hermes_cli import org_lan_server  # noqa: PLC0415

        tok = org_lan_server._lan_token()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        tok = ""
    if not tok:
        return {"ok": False, "error": "本机未配置 kari.lan_token,无法直连"}
    message_obj: dict = {
        "role": "user",
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": str(message)}],
    }
    if context_id:
        message_obj["contextId"] = context_id
    self_uid = self_user_id() or ""
    params: dict = {"message": message_obj, "callerUid": self_uid}
    # Phase 3 身份:签一张主授权票(子用本机公钥验签),证明 caller 身份不靠可伪造的 callerUid;
    # 票里绑定**本条消息内容**(context_id + message),防持票改内容重放(codex blocker)。
    try:
        from hermes_cli import kari_identity  # noqa: PLC0415

        params["ticket"] = kari_identity.make_ticket(self_uid, target_uid, context_id, str(message))
    except Exception:  # noqa: BLE001
        pass
    body = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": params}
    try:
        import httpx  # noqa: PLC0415

        r = httpx.post(
            f"http://{ip}:{port}/a2a",
            json=body,
            headers={"X-Kari-LAN-Token": tok, "content-type": "application/json"},
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"连下级 {ip}:{port} 失败:{e}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"下级返回 {r.status_code}:{r.text[:200]}"}
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "下级返回非 JSON"}
    if data.get("error"):
        err = data["error"]
        return {"ok": False, "error": str(err.get("message") if isinstance(err, dict) else err)}
    result = data.get("result") or {}
    parts = result.get("parts") or []
    answer = "\n".join(str(p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
    return {"ok": True, "answer": answer, "context_id": str(result.get("contextId") or context_id or "")}


# --------------------------- 知识库目录采集 + 上报(子侧,Phase 1c)---------------------------
_LANGFLOW_KB_PATH = "/api/v1/knowledge_bases"


def _langflow_base() -> Optional[str]:
    """本机 langflow 基址(读自己的知识库目录用)。**不可达返回 None —— 绝不为了上报去强拉起 langflow。**"""
    try:
        from hermes_cli.workflow_backend import is_reachable, langflow_url

        if is_reachable(timeout=2.0):
            return langflow_url().rstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return None


def gather_knowledge_resources() -> "list[dict] | None":
    """读本机 langflow 知识库清单 → 资源目录条目 [{resource_id,name,meta}]。
    langflow 不可达 / 读失败 → 返回 **None**(区分"没起 langflow"和"确实没有知识库 []");
    上报侧据此 None 时跳过、不误清云缓冲(与 kari_resources / store-and-forward 的 None 语义一致)。"""
    base = _langflow_base()
    if not base:
        return None
    # 知识库路由即便开了 SKIP_AUTH_AUTO_LOGIN 也要鉴权(实测无 token → 403),所以尽力带一个
    # auto_login token(与 gather_workflow_resources / expose_flow_tool 一致;取不到就裸调)。
    headers = {"Accept": "application/json"}
    token = _langflow_auto_login_token(base)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + _LANGFLOW_KB_PATH, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as r:
            kbs = _read_json(r)
    except urllib.error.HTTPError as e:
        # 401/403 = langflow 起来了但仍拒绝(token 没取到 / 该版本鉴权更严)。行为仍是跳过
        # (返回 None 不误清缓冲),但记日志,免得"知识库目录一直空"却无从排查(和"langflow 没起"区分)。
        if e.code in (401, 403):
            logger.warning("读知识库目录被 langflow 拒绝(HTTP %s):token 未取到或鉴权更严,本轮跳过上报", e.code)
        return None
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(kbs, list):
        return None
    items: list[dict] = []
    for kb in kbs:
        if not isinstance(kb, dict):
            continue
        # dir_name 是稳定主键(KB 名空格→下划线后的目录名);兜底用 id。
        rid = str(kb.get("dir_name") or kb.get("id") or "").strip()
        if not rid:
            continue
        meta = {
            k: kb.get(k)
            for k in ("chunks", "size", "words", "characters", "status", "embedding_model", "source_types")
            if kb.get(k) is not None
        }
        items.append({"resource_id": rid, "name": str(kb.get("name") or rid).strip(), "meta": meta})
    return items


def report_knowledge_resources() -> tuple[bool, dict]:
    """采集本机知识库目录并上报到云缓冲(经 1b 的 report_resources)。
    langflow 没起 → 跳过(不算失败,也不上报空清单去误清主端缓冲)。"""
    base, token = _cloud()
    if not (base and token):
        return False, {"error": "未登录云端"}
    items = gather_knowledge_resources()
    if items is None:
        return False, {"skipped": "langflow 不可达,跳过知识库上报"}
    _store_own_resources("knowledge", items)  # 自有知识库也登本地注册表(可被自己授权)
    return report_resources("knowledge", items)


# --------------------------- 工作流目录采集 + 上报(子侧,Phase 4)---------------------------
_LANGFLOW_FLOWS_PATH = "/api/v1/flows/?get_all=true&header_flows=false"
_LANGFLOW_AUTO_LOGIN_PATH = "/api/v1/auto_login"


def _workflow_items_from_flows(flows: "list | None") -> "list[dict] | None":
    """把 langflow flows 列表映射成资源目录条目(纯函数,便于测试)。
    只收对话流;resource_id = flow UUID(稳定);meta 带 mcp_enabled / action_name。
    对话流判据与注册口 expose_flow_as_tool 共用 tools.flow_chat(避免漂移);延迟导入守 org_client 顶部只放标准库的约定。"""
    if not isinstance(flows, list):
        return None
    from tools.flow_chat import is_chat_flow  # noqa: PLC0415

    items: list[dict] = []
    for fl in flows:
        if not isinstance(fl, dict) or not is_chat_flow(fl):
            continue
        # 只收**已注册成 MCP 工具**(mcp_enabled)的对话流 —— 没注册的还不是工具、不可授权;
        # 这也天然排除 langflow 自带的一堆只读 starter 示例(它们 mcp_enabled=False),避免授权面板被刷屏。
        if not bool(fl.get("mcp_enabled")):
            continue
        fid = str(fl.get("id") or "").strip()
        if not fid:
            continue
        meta: dict = {"mcp_enabled": True}
        action_name = fl.get("action_name")
        if action_name:
            meta["action_name"] = str(action_name)
        items.append({"resource_id": fid, "name": str(fl.get("name") or fid).strip(), "meta": meta})
    return items


def _langflow_auto_login_token(base: str) -> Optional[str]:
    """尽力拿一个 auto_login token;拿不到返回 None(SKIP_AUTH_AUTO_LOGIN 下本就不需要)。"""
    req = urllib.request.Request(base + _LANGFLOW_AUTO_LOGIN_PATH, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return ((_read_json(r) or {}).get("access_token")) or None
    except Exception:  # noqa: BLE001
        return None


def gather_workflow_resources() -> "list[dict] | None":
    """读本机 langflow flows → 只取「ChatInput + ChatOutput」对话流 → 资源目录条目。
    langflow 不可达 / 读失败 → 返回 **None**(跳过,绝不上报 [] 误清主端缓冲;与知识库同语义)。"""
    base = _langflow_base()
    if not base:
        return None
    headers = {"Accept": "application/json"}
    # flows 端点常需鉴权:能拿到 auto_login token 就带上;SKIP_AUTH_AUTO_LOGIN 下不带也行。
    token = _langflow_auto_login_token(base)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + _LANGFLOW_FLOWS_PATH, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15.0) as r:
            flows = _read_json(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            logger.warning("读 flows 被 langflow 拒绝(HTTP %s):疑似未开 SKIP_AUTH_AUTO_LOGIN,本轮跳过工作流上报", e.code)
        return None
    except Exception:  # noqa: BLE001
        return None
    if isinstance(flows, dict):
        # 分页对象 {flows:[...]} 取数组;其它 dict 形态(如错误体)→ 当读失败跳过,
        # 绝不降级成 [](那会误报空清掉主端工作流缓冲)。
        flows = flows.get("flows")
    return _workflow_items_from_flows(flows)


def report_workflow_resources() -> tuple[bool, dict]:
    """采集本机工作流(仅对话流)并上报到云缓冲(经 1b)。langflow 没起 → 跳过(不误清)。"""
    base, token = _cloud()
    if not (base and token):
        return False, {"error": "未登录云端"}
    items = gather_workflow_resources()
    if items is None:
        return False, {"skipped": "langflow 不可达,跳过工作流上报"}
    _store_own_resources("workflow", items)  # 自有工作流也登本地注册表(可被自己授权)
    return report_resources("workflow", items)


# --------------------------- agent 能力采集 + 上报(子侧,Phase 4 · ②面3 委派任务)---------------------------
def gather_agent_resources() -> "list[dict] | None":
    """本节点的爱马仕 agent 作为一条可授权/可委派资源(②面3)。每个节点恰好一个 agent →
    resource_id 固定 ``"agent"``;meta 带能力简介/模型/工具,便于上级挑谁委派。

    agent 始终存在(不依赖 langflow),故总返回一条(不会 None,也就总会登记 + 上报)。
    与 report_capabilities(能力 blob,给实时 fan-out)是两张表两用途:这里是资源目录(给授权)。
    """
    cap = gather_capabilities()
    meta: dict = {}
    for key in ("summary", "model", "host"):
        val = cap.get(key)
        if val:
            meta[key] = val
    toolsets = cap.get("toolsets")
    if isinstance(toolsets, list) and toolsets:
        meta["toolsets"] = toolsets
    name = str(cap.get("summary") or cap.get("host") or "本地爱马仕").strip()
    return [{"resource_id": "agent", "name": name, "meta": meta}]


def report_agent_resources() -> tuple[bool, dict]:
    """上报本节点 agent 为可授权资源(kind='agent')+ 登本地注册表(可被自己/上级授权委派)。"""
    base, token = _cloud()
    if not (base and token):
        return False, {"error": "未登录云端"}
    items = gather_agent_resources()
    _store_own_resources("agent", items)
    return report_resources("agent", items)


# --------------------------- 应答主账号查询(子侧)---------------------------
def answer_query(query: str) -> str:
    """用本地爱马仕(oneshot agent,带其记忆/知识库/技能)回答一个查询,返回最终文本。"""
    model = (os.environ.get("KARI_ORG_ANSWER_MODEL") or "").strip() or None
    try:
        from hermes_cli.oneshot import _run_agent

        ans = _run_agent(query, model=model)
        return (ans or "").strip() or "(本地爱马仕未产生回答)"
    except Exception as e:  # noqa: BLE001
        return f"(本地爱马仕作答失败:{e})"


def poll_and_answer_once(limit: int = 5) -> int:
    """取走分发来的 ask 任务,逐个作答并回报。返回本次处理的任务数。"""
    base, token = _cloud()
    if not (base and token):
        return 0
    st, r = _call("GET", f"{_POLL_PATH}?flow=ask&limit={int(limit)}", token)
    if st != 200:
        return 0
    tasks = r.get("tasks") or []
    done = 0
    for t in tasks:
        tid = t.get("id")
        query = ((t.get("payload") or {}).get("query") or "").strip()
        if not tid:
            continue
        if not query:
            _call("POST", f"/api/v1/kari/tasks/{tid}/ack", token, {"status": "failed", "result": {"error": "空查询"}})
            continue
        answer = answer_query(query)
        _call("POST", f"/api/v1/kari/tasks/{tid}/ack", token, {"status": "done", "result": {"answer": answer}})
        done += 1
    return done


def run_responder(report_interval: float = 300.0, poll_interval: float = 5.0,
                  stop_event: "threading.Event | None" = None) -> None:
    """组织协同后台循环(每个本地实例都跑;同时承担子向上、主向下两侧):
      - 子侧:启动即上报一次能力 + 本机知识库目录,之后定期重报 + 高频轮询应答上级查询。
      - 主侧:按 report_interval 拉取下级上报的资源目录 → 存主本地注册表 → ack 清云(1d)。
        无下级时拉取返回空、空转无害,所以两侧合用一个线程。

    由 hermes dashboard(web_server)的生命周期在有 kari token 时拉起。无 token 直接返回。
    """
    base, token = _cloud()
    if not (base and token):
        return
    stop = stop_event or threading.Event()
    report_capabilities()
    report_knowledge_resources()  # 1c:启动即上报一次本机知识库目录(langflow 没起则跳过)
    report_workflow_resources()  # Phase 4:上报本机工作流(仅 ChatInput+ChatOutput 对话流)
    report_agent_resources()  # Phase 4 · ②面3:上报本节点 agent 为可委派资源
    pull_and_store_resource_reports()  # 1d:启动即拉一次下级资源上报存本地(无下级=空转)
    last_report = time.time()
    while not stop.is_set():
        try:
            poll_and_answer_once()
            if time.time() - last_report >= report_interval:
                report_capabilities()
                report_knowledge_resources()
                report_workflow_resources()  # Phase 4
                report_agent_resources()  # Phase 4 · ②面3
                pull_and_store_resource_reports()  # 1d:主侧定期拉取下级资源 → 存本地 → ack
                last_report = time.time()
        except Exception:  # noqa: BLE001
            pass
        stop.wait(poll_interval)


def start_responder_thread() -> "threading.Thread | None":
    """有 kari token 才起后台应答线程(daemon)。返回线程或 None。"""
    base, token = _cloud()
    if not (base and token):
        return None
    th = threading.Thread(target=run_responder, name="kari-org-responder", daemon=True)
    th.start()
    return th


# --------------------------- 主账号侧:看能力 + 扇出查询 + 收齐 ---------------------------
def subtree_capabilities() -> list[dict]:
    base, token = _cloud()
    if not (base and token):
        return []
    st, r = _call("GET", _CAP_PATH, token)
    return r.get("capabilities") or [] if st == 200 else []


def pull_resource_reports(since: float | None = None) -> list[dict]:
    """(主)拉取子树下级上报的资源目录。每项 = {user_id,email,name,kind,items,updated_ts}。
    存进主本地注册表(kari_resources)在 1d;这里只是传输层。"""
    base, token = _cloud()
    if not (base and token):
        return []
    path = _RESOURCES_PATH if since is None else f"{_RESOURCES_PATH}?since={float(since)}"
    st, r = _call("GET", path, token)
    return r.get("reports") or [] if st == 200 else []


def ack_resource_reports(user_id: str, kind: str | None = None, up_to_ts: float | None = None) -> int:
    """(主)存进本地后清掉云端这份缓冲。返回清掉的条数(失败 0)。
    传 up_to_ts(= pull 拿到的那条 updated_ts)只清到该水位线,避免清掉 GET 后子又重报的新版本。"""
    base, token = _cloud()
    if not (base and token):
        return 0
    body: dict = {"user_id": str(user_id)}
    if kind:
        body["kind"] = kind
    if up_to_ts is not None:
        body["up_to_ts"] = up_to_ts
    st, r = _call("POST", _RESOURCES_ACK_PATH, token, body)
    return int(r.get("cleared") or 0) if st == 200 else 0


def pull_and_store_resource_reports() -> dict:
    """(主)拉取子树下级上报的资源目录 → 存进**主本地**注册表(kari_resources)→ ack 清云缓冲。Phase 1d。

    在 run_responder 里按 report_interval 调一次。每个节点既是子又可能是主;云端
    subtree_resource_reports 已按 can_read 过滤并排除自己,无下级时返回 [](无害空转)。

    幂等 + 防丢更新:
      - replace_node_resources 是**整体替换**(子重报会删掉已不存在的,保证主端副本跟手)。
      - **先存后 ack**:存进本地成功才清云端;ack 用拉到的 updated_ts 做水位线,
        避免清掉 GET 之后子又重报的新版本(lost-update,见 store.clear_resource_report)。
      - 报文异常(缺 user_id/kind 或 items 不是 list)→ 跳过、不 ack,下轮重试。

    返回 {pulled, stored_nodes, items} 摘要(诊断/日志用)。
    """
    base, token = _cloud()
    if not (base and token):
        return {"pulled": 0, "stored_nodes": 0, "items": 0}
    try:
        from tools import kari_resources  # noqa: PLC0415 — 主本地注册表,与 web_server 的读路径同源
    except Exception as e:  # noqa: BLE001
        logger.warning("加载本地资源注册表失败,跳过拉取存档:%s", e)
        return {"pulled": 0, "stored_nodes": 0, "items": 0, "error": str(e)}
    reports = pull_resource_reports()
    known_kinds = set(getattr(kari_resources, "KINDS", ()))
    stored_nodes = 0
    items_total = 0
    for rep in reports:
        uid = str(rep.get("user_id") or "").strip()
        kind = str(rep.get("kind") or "").strip()
        items = rep.get("items")
        if not (uid and kind) or not isinstance(items, list):
            continue  # 报文异常:不动本地副本、不 ack,下轮重试
        if kind not in known_kinds:
            # 云端允许但本地注册表不认的 kind(理论上 REPORT_KINDS==KINDS,真出现=部署不一致):
            # replace_node_resources 会原样返回 0,若据此 ack 会**没存却清云**=丢数据。
            # 所以显式跳过、不 ack(留在云缓冲可重试),并记日志暴露不一致。
            logger.warning("跳过未知 kind 的资源上报 user=%s kind=%s(本地注册表不支持,疑似云/端版本不一致)", uid, kind)
            continue
        n = kari_resources.replace_node_resources(uid, kind, items)
        if n < 0:
            continue  # None 语义=不动注册表(items 已是 list,防御性分支)
        stored_nodes += 1
        items_total += n
        ts = rep.get("updated_ts")
        ack_resource_reports(uid, kind, up_to_ts=ts if isinstance(ts, (int, float)) else None)
    return {"pulled": len(reports), "stored_nodes": stored_nodes, "items": items_total}


def ask(query: str, targets: list[str] | None = None) -> list[int]:
    base, token = _cloud()
    if not (base and token):
        return []
    st, r = _call("POST", _ASK_PATH, token, {"query": query, "targets": targets})
    return r.get("task_ids") or [] if st == 200 else []


def collect(task_ids: list[int]) -> dict:
    base, token = _cloud()
    if not (base and token):
        return {"answers": [], "all_done": True, "pending": []}
    st, r = _call("POST", _COLLECT_PATH, token, {"task_ids": task_ids})
    return r if st == 200 else {"answers": [], "all_done": True, "pending": []}


def query_subordinates(query: str, wait_seconds: float = 30.0, poll_interval: float = 2.0) -> dict:
    """扇出查询给下级 → 轮询收齐(到 all_done 或超时)。返回 {answers, all_done, asked, capabilities}。

    主账号 agent 据此 + 自身知识库汇总。answers 已并入各下级身份/能力简介,便于归因。
    """
    caps = {c["user_id"]: c for c in subtree_capabilities()}
    task_ids = ask(query, None)
    if not task_ids:
        return {"answers": [], "all_done": True, "asked": 0, "note": "无下级可问(或未登录)"}
    deadline = time.time() + max(1.0, wait_seconds)
    result = collect(task_ids)
    while not result.get("all_done") and time.time() < deadline:
        time.sleep(max(0.5, poll_interval))
        result = collect(task_ids)
    answers = []
    for a in result.get("answers", []):
        cap = caps.get(a.get("target_id")) or {}
        answers.append({
            "subordinate": cap.get("name") or cap.get("email") or a.get("target_id"),
            "status": a.get("status"),
            "answer": (a.get("result") or {}).get("answer") if isinstance(a.get("result"), dict) else a.get("result"),
            "summary": (cap.get("capability") or {}).get("summary") if isinstance(cap.get("capability"), dict) else None,
        })
    return {
        "answers": answers,
        "all_done": result.get("all_done", False),
        "asked": len(task_ids),
        "pending": result.get("pending", []),
    }


def delegate_to(target_user_id: str, task: str, wait_seconds: float = 60.0, poll_interval: float = 2.0) -> dict:
    """把一个任务**委派给某一个具体下级 agent**(②面3 委派任务),等其作答后收回。

    与 query_subordinates(扇出问全部)不同:这里定向给一个目标。复用 ask/dispatch/collect 中继
    (云端 ask 已支持 targets;只能派给子树内可见下级,鉴权在云端 can_read)。返回
    {target, status, answer, all_done}。目标不可达/非下级 → status='no-target'。
    """
    target_user_id = str(target_user_id or "").strip()
    if not (target_user_id and (task or "").strip()):
        return {"target": target_user_id, "status": "bad-request", "answer": None, "note": "需要 target_user_id 和非空 task"}
    caps = {c["user_id"]: c for c in subtree_capabilities()}
    task_ids = ask(task, [target_user_id])
    if not task_ids:
        return {"target": target_user_id, "status": "no-target", "answer": None, "note": "目标不可达或非下级(或未登录)"}
    deadline = time.time() + max(1.0, wait_seconds)
    result = collect(task_ids)
    while not result.get("all_done") and time.time() < deadline:
        time.sleep(max(0.5, poll_interval))
        result = collect(task_ids)
    answers = result.get("answers") or []
    a = answers[0] if answers else {}
    cap = caps.get(a.get("target_id")) or {}
    return {
        "target": cap.get("name") or cap.get("email") or target_user_id,
        "status": a.get("status") or ("pending" if not result.get("all_done") else "no-answer"),
        "answer": (a.get("result") or {}).get("answer") if isinstance(a.get("result"), dict) else a.get("result"),
        "all_done": result.get("all_done", False),
    }
