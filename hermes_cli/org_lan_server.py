"""EasyHermes 组织局域网服务(3b)—— 把「按授权的跨节点能力」暴露到局域网。

**隔离设计(安全要点)**:dashboard 主服务仍只 bind 127.0.0.1(它的会话/OAuth 鉴权一动不动);
跨节点能力另起一个**专用小服务**:纯标准库 ``ThreadingHTTPServer``,bind ``0.0.0.0:ORG_LAN_PORT``,
**只挂 org 端点**。攻击面 = 就这几个端点,不会牵连整个 dashboard。

**门控**(每个请求都查,顺序:令牌→来源 IP):
  - ``X-Kari-LAN-Token`` 必须等于本机配置的 ``kari.lan_token``(团队共享密钥,主/子同值);常数时间比对。
  - 来源地址必须是私网(10/192.168/172.16-31)或回环 —— 公网来源直接拒。

当前能力(slice 1)= **model-B 入库**:弱子(没装工作流引擎 / 内存小)把知识库**原文**发给主,
主做 embedding + 入库(KB 名带子 uid 隔离);子永远只发文本、收结果,自己不跑 langflow。
未配 ``kari.lan_token`` → 本服务**不启动**(opt-in,默认不对外开任何口)。
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

logger = logging.getLogger(__name__)

ORG_LAN_PORT = 48901       # 组织 LAN 服务固定端口(发现广播会带上,子据此连主)
_MAX_BODY = 64 * 1024 * 1024   # 单次上传上限 64MB,挡异常大包


# --------------------------- 配置 / 门控 ---------------------------
def _lan_token() -> str:
    """团队共享 LAN 令牌:env ``KARI_LAN_TOKEN`` 优先,回退 workflow-secrets 的 ``kari.lan_token``。
    空 = 没配 = 本服务不该启动。"""
    tok = (os.environ.get("KARI_LAN_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        from hermes_cli.workflow_backend import read_secrets

        kari = (read_secrets().get("kari") or {}) if callable(read_secrets) else {}
        return str(kari.get("lan_token") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _ip_allowed(ip: str) -> bool:
    """来源 IP 必须私网或回环。"""
    if not ip:
        return False
    if ip.startswith("127.") or ip == "::1":
        return True
    try:
        from hermes_cli.lan_discovery import _is_lan_ip  # noqa: PLC0415

        return _is_lan_ip(ip)
    except Exception:  # noqa: BLE001
        return False


def _token_ok(supplied: str) -> bool:
    expected = _lan_token()
    if not expected or not supplied:
        return False
    return hmac.compare_digest(supplied.encode(), expected.encode())


# --------------------------- model-B:入库 ---------------------------
def ingest_for_sub(owner_uid: str, kb_name: str, files: list) -> dict:
    """把子上传的原文 embedding + 入主的 langflow KB。

    ``files`` = ``[{"rel": 相对路径, "text": 正文}]``。KB 名带子 uid 前缀做隔离。
    需要本机(主)langflow 可用 —— 这正是 model-B 的前提:重活在主、子只发文本。"""
    from tools import knowledge_tool as kt  # noqa: PLC0415
    import tempfile

    base, token = kt._lf()  # noqa: SLF001
    if not base:
        raise RuntimeError("本机工作流引擎(langflow)不可用,无法承接子节点入库")

    # KB 名带子 uid 前缀做隔离(两个子的「客服库」在主上互不覆盖)。中文直接传:langflow 侧
    # 已统一把 Chroma collection 名规整成 ASCII(见 chroma_security.chroma_collection_name),
    # 目录名/显示名仍保留中文。
    short = (str(owner_uid or "sub").strip() or "sub")[:8]
    kb_label = f"子{short}-{str(kb_name or 'kb').strip()}"

    tmpdir = tempfile.mkdtemp(prefix="kari_lan_kb_")
    paths: list[str] = []
    for i, f in enumerate(files or []):
        if not isinstance(f, dict):
            continue
        rel = str(f.get("rel") or f"doc{i}.txt")
        fname = os.path.basename(rel) or f"doc{i}.txt"
        p = os.path.join(tmpdir, f"{i}_{fname}")
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(str(f.get("text") or ""))
            paths.append(p)
        except OSError:
            pass

    import httpx  # noqa: PLC0415

    with httpx.Client(timeout=120.0) as client:
        kb = kt._ensure_kb(client, base, token, kb_label)  # noqa: SLF001
        kt._ingest(client, base, token, kb, kb_label, paths, [])  # noqa: SLF001

    # 入库后让协同注册表/团队面板同步(新 KB 进主的资源,可被授权)
    try:
        kt._refresh_registry()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass

    return {"kb": kb, "kb_display": kb_label, "indexed": len(paths)}


def _self_uid() -> str:
    try:
        from hermes_cli import org_client  # noqa: PLC0415

        return str(org_client.self_user_id() or "")
    except Exception:  # noqa: BLE001
        return ""


# --------------------------- HTTP handler ---------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "EasyHermesOrgLAN/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # 静音默认 stderr 日志
        return

    def _client_ip(self) -> str:
        return (self.client_address[0] if self.client_address else "") or ""

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _gate(self) -> bool:
        """令牌 + 来源 IP 双门控;不过就回 403 并返回 False。"""
        if not _token_ok(self.headers.get("X-Kari-LAN-Token", "")):
            self._send_json(403, {"error": "无效的 LAN 令牌"})
            return False
        if not _ip_allowed(self._client_ip()):
            self._send_json(403, {"error": "来源地址非私网"})
            return False
        return True

    def _read_json(self) -> Optional[dict]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > _MAX_BODY:
            self._send_json(413 if n > _MAX_BODY else 400, {"error": "请求体非法"})
            return None
        try:
            raw = self.rfile.read(n)
            obj = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            self._send_json(400, {"error": "JSON 解析失败"})
            return None
        if not isinstance(obj, dict):
            self._send_json(400, {"error": "请求体必须是对象"})
            return None
        return obj

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/org/ping":
            if not self._gate():
                return
            self._send_json(200, {"ok": True, "uid": _self_uid()})
            return
        self._send_json(404, {"error": "未知端点"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") == "/kb/ingest":
            if not self._gate():
                return
            body = self._read_json()
            if body is None:
                return
            owner_uid = str(body.get("owner_uid") or "")
            kb_name = str(body.get("kb_name") or "kb")
            files = body.get("files") or []
            if not isinstance(files, list) or not files:
                self._send_json(400, {"error": "files 不能为空"})
                return
            try:
                result = ingest_for_sub(owner_uid, kb_name, files)
            except Exception as e:  # noqa: BLE001
                logger.warning("org LAN 入库失败:%s", e)
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, {"ok": True, **result})
            return
        self._send_json(404, {"error": "未知端点"})


# --------------------------- 生命周期 ---------------------------
def run_org_lan_server(stop_event: "threading.Event | None" = None, port: int = ORG_LAN_PORT) -> None:
    """阻塞跑 org LAN 服务(在后台线程里调用)。未配 lan_token 直接返回(不开口)。"""
    if not _lan_token():
        logger.info("未配置 kari.lan_token,组织 LAN 服务不启动(默认不对外开口)")
        return
    stop = stop_event or threading.Event()
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except OSError as e:
        logger.warning("组织 LAN 服务 bind 0.0.0.0:%s 失败:%s", port, e)
        return
    httpd.timeout = 1.0

    def _wait_stop():
        stop.wait()
        try:
            httpd.shutdown()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_wait_stop, name="kari-org-lan-stop", daemon=True).start()
    logger.info("组织 LAN 服务已启动 0.0.0.0:%s(令牌+私网门控)", port)
    try:
        httpd.serve_forever(poll_interval=1.0)
    finally:
        try:
            httpd.server_close()
        except Exception:  # noqa: BLE001
            pass


def start_org_lan_thread(port: int = ORG_LAN_PORT) -> "threading.Thread | None":
    if not _lan_token():
        return None
    th = threading.Thread(target=run_org_lan_server, kwargs={"port": port}, name="kari-org-lan", daemon=True)
    th.start()
    return th
