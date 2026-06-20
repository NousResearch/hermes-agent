"""每节点 Ed25519 身份密钥 —— A2A「主签票 / 子验签」的身份层(Phase 3,codex 评审 #3)。

为什么需要:A2A 的 ``callerUid`` 在共享 LAN 令牌下是**可伪造的自报**,任何基于它的子侧鉴权都能被
同队成员绕过。要可靠区分调用方身份,必须密码学绑定:

  - 每个节点本地持有一对 **Ed25519** 密钥(私钥 0600 存 ``~/.hermes/kari_node_ed25519``,永不外发)。
  - 节点把**公钥**经云端能力通道上报(``gather_capabilities`` 带 ``pubkey``);云端按账号树鉴权后,
    子节点可据 ``root_id`` 取到主的公钥(``org_client.fetch_pubkey``)。
  - 主调用子时签一张短时效**票**(``make_ticket``);子用「从云端取到的主公钥」验签(``verify_ticket``)
    + 校 target/exp/nonce(防重放)→ 据**验过的** ticket.caller 决定授权,**不信 body 自报**。

公钥是公开信息(同团队互见无妨);私钥从不离开本机。纯标准库 + ``cryptography``(已在依赖)。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import uuid

_KEY_PATH = os.environ.get("KARI_NODE_KEY") or os.path.expanduser("~/.hermes/kari_node_ed25519")
_lock = threading.Lock()
_priv = None  # 缓存的私钥对象


def set_key_path(path: str) -> None:
    """切换私钥路径(测试隔离用),清掉缓存。"""
    global _KEY_PATH, _priv
    with _lock:
        _KEY_PATH = path
        _priv = None


def _load_or_create():
    """读本机私钥;没有就生成并以 0600 落盘。返回 Ed25519PrivateKey。"""
    global _priv
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: PLC0415

    with _lock:
        if _priv is not None:
            return _priv
        if os.path.exists(_KEY_PATH):
            with open(_KEY_PATH, "rb") as f:
                _priv = Ed25519PrivateKey.from_private_bytes(f.read())  # 32B raw;损坏/截断会抛 → 调用方 fail-closed
            try:
                os.chmod(_KEY_PATH, 0o600)  # 收紧既有文件权限(可能是早期非 0600 写的)
            except OSError:
                pass
        else:
            _priv = Ed25519PrivateKey.generate()
            d = os.path.dirname(_KEY_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            raw = _priv.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            # 原子写:先写临时文件(0600)再 rename,避免并发首次生成时互相截断。
            tmp = f"{_KEY_PATH}.{uuid.uuid4().hex}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            os.replace(tmp, _KEY_PATH)
        return _priv


def public_key_b64() -> str:
    """本机公钥(base64 raw 32B)。上报云端 / 给对端验签。"""
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    pk = _load_or_create().public_key()
    raw = pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _sign_b64(data: bytes) -> str:
    return base64.b64encode(_load_or_create().sign(data)).decode()


def verify_sig(pubkey_b64: str, data: bytes, sig_b64: str) -> bool:
    """用 pubkey 验 data 的签名。任何异常(格式/无效签名)一律 False。"""
    from cryptography.exceptions import InvalidSignature  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: PLC0415

    try:
        pk = Ed25519PublicKey.from_public_bytes(base64.b64decode(pubkey_b64))
        pk.verify(base64.b64decode(sig_b64), data)
        return True
    except (InvalidSignature, ValueError, TypeError, Exception):  # noqa: BLE001
        return False


def _canon(body: dict) -> bytes:
    """票体规范化序列化(签名/验签必须逐字节一致)。"""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _request_digest(context_id: str, text: str) -> str:
    """把「请求内容(contextId + 文本)」摘成哈希,签进票里 **绑定请求**——
    否则签名只认身份不认内容,持票者可改消息文本/contextId 重放(codex blocker)。"""
    return hashlib.sha256(f"{context_id or ''}\x00{text or ''}".encode("utf-8")).hexdigest()


def make_ticket(caller_uid: str, target_uid: str, context_id: str = "", text: str = "", ttl: float = 120.0) -> dict:
    """主侧签一张授权票:证明「caller_uid(本机,私钥持有者)要在 context 上对 target_uid 发**这条**消息」。

    返回 {"body": {...}, "sig": "..."}。body 里 ``mh`` = 请求内容摘要(绑定 contextId+文本),
    连同 caller/target/exp/nonce 一起签。子侧用 caller 公钥验签 + 重算 mh 比对。短时效 + nonce 防重放。
    """
    now = time.time()
    body = {
        "v": 1,
        "caller": str(caller_uid or ""),
        "target": str(target_uid or ""),
        "ctx": str(context_id or ""),
        "mh": _request_digest(context_id, text),
        "iat": now,
        "exp": now + max(1.0, ttl),
        "nonce": uuid.uuid4().hex,
    }
    return {"body": body, "sig": _sign_b64(_canon(body))}


def verify_ticket(
    ticket: dict,
    caller_pubkey_b64: str,
    expected_target_uid: str,
    context_id: str = "",
    text: str = "",
) -> tuple[bool, dict, str]:
    """子侧验票。返回 (ok, body, reason)。

    校验:结构 → target==自身 → 未过期 → 用 caller 公钥验签 → **请求内容摘要(mh)与实际请求一致**。
    传 context_id/text(实际请求里的)以核 mh —— 防持票改内容重放。**不**在此查 nonce(状态在更上层)。
    """
    if not isinstance(ticket, dict):
        return False, {}, "ticket 非对象"
    body = ticket.get("body")
    sig = ticket.get("sig")
    if not (isinstance(body, dict) and isinstance(sig, str) and sig):
        return False, {}, "ticket 缺 body/sig"
    if str(body.get("target") or "") != str(expected_target_uid or ""):
        return False, body, "target 与本节点不符"
    try:
        exp = float(body.get("exp") or 0)
    except (TypeError, ValueError):
        return False, body, "exp 非法"
    if exp < time.time():
        return False, body, "票已过期"
    if not caller_pubkey_b64:
        return False, body, "拿不到调用方公钥"
    if not verify_sig(caller_pubkey_b64, _canon(body), sig):
        return False, body, "签名验证失败"
    if str(body.get("mh") or "") != _request_digest(context_id, text):
        return False, body, "请求内容与票不符(疑似改内容重放)"
    return True, body, ""
