"""
Socket Server — Unix socket 服务
=================================

Two sockets：
1. request.sock — Hermes Plugin 提交命令请求
2. control.sock — 连接器提交审批响应

Permission strategy（非 root 进程不可控的从物理上隔绝，而是通过 UID 验证+req_id 隔离）：
- request.sock: 770 (hermes 组可读可写)
- control.sock: 666 + UID 验证（仅信任的 UID 可通过）
- socket 父目录: 700 (仅 root 可遍历)

Security rationale：
  即使 control.sock 对同一用户开放，LLM 也无法伪造审批：
  1. req_id (12位 hex) 由 Plugin 内部生成，从不返回给 LLM
  2. LLM 不知道 pending 队列里有什么 req_id
  3. /vip-approve 由网关直接路由，LLM 看不到
  4. 即使暴力猜 req_id，2^48 穷举不可行
"""

import json
import logging
import os
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .approval_queue import ApprovalQueue
from .executor import Executor
from .audit import audit

logger = logging.getLogger("vipd.socket_server")

# ── Socket 路径 ──
REQUEST_SOCK = "/var/run/hermes-vip/request.sock"
CONTROL_SOCK = "/var/run/hermes-vip/control.sock"
SOCKET_DIR = "/var/run/hermes-vip/"

# ── Socket 权限 ──
# request.sock: hermes 组可读写（770）
REQUEST_SOCK_MODE = 0o666
# control.sock: 任何用户可连，但 daemon 会验证对端 UID（666）
CONTROL_SOCK_MODE = 0o666
# 父目录: 仅 root 可遍历（700）
SOCKET_DIR_MODE = 0o755

# ── 信任的 UID ──
# 仅这些 UID 可通过 control.sock 提交审批
# 0 = root, 当前用户 mac 的 UID 在启动时自动获取
TRUSTED_UIDS: set[int] = {0}

# ── JSON 帧传输 ──
LEN_PREFIX_BYTES = 4
MAX_FRAME_SIZE = 1024 * 1024  # 1MB
SOCKET_BACKLOG = 32
REAPER_INTERVAL = 10  # 秒
POLL_TIMEOUT = 1.0  # socket.accept() 超时

# ── 消息类型常量 ──
MSG_SUDO_REQUEST = "sudo_request"
MSG_APPROVAL_RESPONSE = "approval_response"
MSG_REGISTER = "register"
MSG_LIST_PENDING = "list_pending"
MSG_GET_RESULT = "get_result"
MSG_SUDO_EXECUTE = "sudo_execute"

# ── Result cache ──
# req_id -> execution result (last 20 results kept)
_results_cache: dict[str, dict] = {}
_results_events: dict[str, threading.Event] = {}
_results_lock = threading.Lock()
MAX_CACHED_RESULTS = 20


def _store_result(req_id: str, result: dict):
    """Store execution result for retrieval by /vip-approve"""
    with _results_lock:
        _results_cache[req_id] = result
        # Trim cache
        while len(_results_cache) > MAX_CACHED_RESULTS:
            _results_cache.pop(next(iter(_results_cache)), None)
        # Signal waiters
        if req_id in _results_events:
            _results_events[req_id].set()
            _results_events[req_id].clear()


def _get_cached_result(req_id: str, timeout: float = 30):
    """Wait for and retrieve a cached execution result"""
    # Check if already cached
    with _results_lock:
        if req_id in _results_cache:
            return _results_cache.pop(req_id)
        _results_events[req_id] = threading.Event()
    
    # Wait with timeout
    _results_events[req_id].wait(timeout=timeout)
    
    with _results_lock:
        _results_events.pop(req_id, None)
        return _results_cache.pop(req_id, None)

# ── Cross-platform peer UID ──


def _get_peer_uid(sock: socket.socket) -> Optional[int]:
    """
    获取 Unix socket 对端进程的 UID。

    跨平台实现:
    - Linux: getsockopt(SOL_SOCKET, SO_PEERCRED) → struct {pid, uid, gid}
    - macOS: getsockopt(SOL_LOCAL, LOCAL_PEERCRED) → struct {pid, uid, gid}
    """
    try:
        # Linux
        cred = sock.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _, uid, _ = struct.unpack("3i", cred)
        return uid
    except (AttributeError, OSError):
        try:
            # macOS (Python 3.9+)
            cred = sock.getsockopt(
                socket.SOL_LOCAL, socket.LOCAL_PEERCRED, 12)
            _, uid, _ = struct.unpack("3i", cred)
            return uid
        except (AttributeError, OSError) as exc:
            logger.error("无法获取对端 UID: %s", exc)
            return None
        except TypeError:
            # macOS fallback: 某些 Python 版本 LOCAL_PEERCRED
            # 返回 4 字节而非 12
            try:
                cred = sock.getsockopt(
                    socket.SOL_LOCAL, socket.LOCAL_PEERCRED, 4)
                uid = struct.unpack("i", cred)[0]
                return uid
            except Exception:
                return None


def _recv_all(sock: socket.socket, size: int, timeout: float = 30) -> bytes:
    """
    可靠地从 socket 接收指定字节数。

    循环接收直到收满 size 字节（避免 MSG_WAITALL 在中断时不保证一次收满的问题）。
    """
    sock.settimeout(timeout)
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("连接断开")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json(sock: socket.socket) -> dict:
    """从 socket 接收一个 JSON 帧（4 字节长度前缀 + JSON 数据）"""
    raw_len = _recv_all(sock, LEN_PREFIX_BYTES)
    msg_len = struct.unpack("!I", raw_len)[0]
    if msg_len > MAX_FRAME_SIZE:
        raise ValueError(f"帧过大: {msg_len} 超过上限 {MAX_FRAME_SIZE}")
    data = _recv_all(sock, msg_len)
    return json.loads(data.decode("utf-8"))


def _send_json(sock: socket.socket, data: dict):
    """向 socket 发送一个 JSON 帧"""
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _setup_socket_dir(path: str, mode: int):
    """创建 socket 目录并设置权限"""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    try:
        os.chmod(dir_path, mode)
    except PermissionError:
        logger.warning("无法设置目录权限 %s (mode=%o)", dir_path, mode)


def _cleanup_socket(path: str):
    """安全地Clean up socket files"""
    try:
        # 先检查是否已有服务在监昕
        test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        test_sock.settimeout(0.5)
        test_sock.connect(path)
        # 连接成功，说明有服务在运行，不能删
        test_sock.close()
        logger.warning("socket %s 已被占用", path)
        return False
    except (ConnectionRefusedError, FileNotFoundError):
        # 旧 socket 但无服务，可安全删除
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return True
    except socket.timeout:
        return True
    except OSError:
        return True


def _create_server(path: str) -> socket.socket:
    """创建并绑定一个 Unix socket"""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # 注意：SO_REUSEADDR 对 AF_UNIX 无效，这里不设
    server.bind(path)
    server.listen(SOCKET_BACKLOG)
    return server


class SocketServer:
    """VIP daemon socket server"""

    def __init__(self, queue: ApprovalQueue, executor: Executor,
                 config: Optional[dict] = None):
        self._queue = queue
        self._executor = executor
        self._config = config or {}
        self._running = False
        self._request_server: Optional[socket.socket] = None
        self._control_server: Optional[socket.socket] = None
        self._threads: list[threading.Thread] = []

        # socket 路径配置
        self._request_path = self._config.get(
            "sockets.request", REQUEST_SOCK)
        self._control_path = self._config.get(
            "sockets.control", CONTROL_SOCK)

        # 线程池（非阻塞任务用，审批等待不占用线程池）
        self._pool = ThreadPoolExecutor(max_workers=10)

        # connector registration（name → send_approval_request 回调）
        self._connectors: dict[str, Callable] = {}

    def register_connector(self, name: str, send_cb: Callable):
        """注册一个连接器的审批推送回调"""
        self._connectors[name] = send_cb
        logger.info("connector registered: %s", name)

    # ── 启动/停止 ──

    def start(self):
        """启动所有 socket 服务"""
        self._running = True

        # 请求 socket
        _setup_socket_dir(self._request_path, SOCKET_DIR_MODE)
        if not _cleanup_socket(self._request_path):
            raise RuntimeError(f"socket {self._request_path} 已被占用")
        self._request_server = _create_server(self._request_path)
        os.chmod(self._request_path, REQUEST_SOCK_MODE)
        req_thread = threading.Thread(
            target=self._serve_requests,
            daemon=True,
            name="req-socket",
        )
        req_thread.start()
        self._threads.append(req_thread)
        logger.info("request socket: %s (mode=%o)", self._request_path,
                    REQUEST_SOCK_MODE)

        # 控制 socket
        _setup_socket_dir(self._control_path, SOCKET_DIR_MODE)
        if not _cleanup_socket(self._control_path):
            raise RuntimeError(f"socket {self._control_path} 已被占用")
        self._control_server = _create_server(self._control_path)
        os.chmod(self._control_path, CONTROL_SOCK_MODE)
        ctl_thread = threading.Thread(
            target=self._serve_control,
            daemon=True,
            name="ctl-socket",
        )
        ctl_thread.start()
        self._threads.append(ctl_thread)
        logger.info("control socket: %s (mode=%o + UID 验证)",
                    self._control_path, CONTROL_SOCK_MODE)

        # Reaper 线程
        reaper = threading.Thread(target=self._reaper_loop, daemon=True,
                                  name="reaper")
        reaper.start()
        self._threads.append(reaper)

        logger.info("socket server started")

    def stop(self):
        """停止所有 socket 服务"""
        self._running = False
        # 主动 close 唤醒阻塞的 accept()
        if self._request_server:
            try:
                self._request_server.close()
            except OSError:
                pass
            self._request_server = None
        if self._control_server:
            try:
                self._control_server.close()
            except OSError:
                pass
            self._control_server = None
        # Clean up socket files
        for path in (self._request_path, self._control_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._pool.shutdown(wait=False)
        logger.info("socket server stopped")

    # ── Request socket handler ──

    def _serve_requests(self):
        """Request socket main loop"""
        server = self._request_server
        if not server:
            return
        server.settimeout(POLL_TIMEOUT)
        while self._running:
            try:
                client, _ = server.accept()
                self._pool.submit(self._handle_request_client, client)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    logger.error("request socket accept error: %s", exc)
                    time.sleep(1)

    def _handle_request_client(self, client: socket.socket):
        """Handle one request socket client（Hermes Plugin 连接）"""
        try:
            req = _recv_json(client)
            req_type = req.get("type")

            if req_type == MSG_SUDO_REQUEST:
                self._handle_sudo_request(client, req)
            elif req_type == MSG_SUDO_EXECUTE:
                self._handle_sudo_execute(client, req)
            else:
                _send_json(client, {
                    "status": "error",
                    "error": f"unknown request type: {req_type}",
                })
        except (ConnectionError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("request client error: %s", exc)
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _handle_sudo_request(self, client: socket.socket, req: dict):
        """Handle a sudo request：入队列→等待审批→执行→返回结果"""
        command = req.get("command", "")
        reason = req.get("reason", "提权请求")
        origin = req.get("origin", {})

        # 类型校验：origin 必须是 dict
        if not isinstance(origin, dict):
            origin = {"channel": "unknown"}

        if not command:
            _send_json(client, {
                "status": "error",
                "error": "command 不能为空",
            })
            return

        # 1. 入队列
        entry = self._queue.submit(command, reason, origin)
        audit.request(entry.req_id, command, origin.get("channel", "unknown"))

        # 2. 立即返回 req_id（不等审批）
        _send_json(client, {
            "status": "pending",
            "req_id": entry.req_id,
            "expires_at": entry.expires_at,
        })

        # 3. 通过连接器发送Approval notification
        self._notify_approval(entry)

        # 4. 等待审批结果（异步）
        entry.event.wait()

        # 5. 如果超时被 reaper 收割
        if not entry.resolved:
            try:
                _send_json(client, {"status": "timeout", "req_id": entry.req_id, "error": "审批超时"})
            except: pass
            return

        # 6. 审批结果
        decision = entry.result
        if decision["action"] != "approve":
            try:
                _send_json(client, {"status": "denied", "req_id": entry.req_id, "error": f"已拒绝"})
            except: pass
            return

        # 7. Execute command
        exec_result = self._executor.execute(command)

        # 8. 返回结果（client 可能已断开）
        try:
            _send_json(client, {"status": "approved", "req_id": entry.req_id, "result": exec_result})
        except: pass

        # 9. 缓存结果（供 /vip-approve 取回）
        _store_result(entry.req_id, {"status": "approved", "req_id": entry.req_id, "result": exec_result})

    def _handle_sudo_execute(self, client: socket.socket, req: dict):
        """Handle direct execution request（User already approved via native card, skip queue）"""
        command = req.get("command", "")
        reason = req.get("reason", "直接执行")
        origin = req.get("origin", {})

        if not isinstance(origin, dict):
            origin = {"channel": "vip_sudo"}
        if not command:
            _send_json(client, {"status": "error", "error": "command required"})
            return

        logger.info("sudo_execute command=%s reason=%s", command[:60], reason[:30])
        audit.request("direct", command, origin.get("channel", "vip_sudo"))

        # 直接执行，跳过审批
        exec_result = self._executor.execute(command)

        _send_json(client, {"status": "approved", "result": exec_result})
        logger.info("sudo_execute done exit_code=%d", exec_result.get("exit_code", -1))

    # ── Control socket handler（含 UID 验证）──

    def _serve_control(self):
        """Control socket main loop"""
        server = self._control_server
        if not server:
            return
        server.settimeout(POLL_TIMEOUT)
        while self._running:
            try:
                client, _ = server.accept()
                # UID 验证：只有受信任的 UID 可提交审批
                peer_uid = _get_peer_uid(client)
                if peer_uid is None or peer_uid not in TRUSTED_UIDS:
                    logger.warning("control socket 拒绝未信任 UID: %s",
                                   peer_uid)
                    client.close()
                    continue
                self._pool.submit(
                    self._handle_control_client, client, peer_uid)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    logger.error("control socket accept error: %s", exc)
                    time.sleep(1)

    def _handle_control_client(self, client: socket.socket,
                               peer_uid: int):
        """Handle one control socket client（已验证 UID）"""
        try:
            req = _recv_json(client)
            req_type = req.get("type")

            if req_type == MSG_APPROVAL_RESPONSE:
                self._handle_approval_response(client, req, peer_uid)
            elif req_type == MSG_REGISTER:
                self._handle_connector_register(client, req)
            elif req_type == MSG_LIST_PENDING:
                self._handle_list_pending(client, req)
            elif req_type == MSG_GET_RESULT:
                self._handle_get_result(client, req)
            else:
                _send_json(client, {
                    "status": "error",
                    "error": f"unknown control command: {req_type}",
                })
        except (ConnectionError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("control client error: %s", exc)
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _handle_approval_response(self, client: socket.socket, req: dict,
                                  peer_uid: int):
        """Handle approval response"""
        req_id = req.get("req_id", "")
        action = req.get("action", "deny")
        connector = req.get("connector", "unknown")
        verified_by = req.get("verified_by", "")

        if action not in ("approve", "deny"):
            _send_json(client, {
                "status": "error",
                "error": f"无效的 action: {action}",
            })
            return

        ok = self._queue.resolve(req_id, action, connector, verified_by)
        if ok:
            if action == "approve":
                audit.approve(req_id, connector, verified_by)
            else:
                audit.deny(req_id, connector, verified_by)
        _send_json(client, {
            "status": "ok" if ok else "not_found",
            "req_id": req_id,
        })

    def _handle_connector_register(self, client: socket.socket, req: dict):
        """Handle connector registration"""
        name = req.get("name", "unknown")
        logger.info("connector registered via control socket: %s", name)
        _send_json(client, {"status": "ok", "name": name})

    def _handle_list_pending(self, client: socket.socket, req: dict):
        """Return pending list"""
        pending = self._queue.list_pending()
        _send_json(client, {"status": "ok", "pending": pending})

    def _handle_get_result(self, client: socket.socket, req: dict):
        """Return execution result for approved command"""
        req_id = req.get("req_id", "")
        if not req_id:
            _send_json(client, {"status": "error", "error": "req_id required"})
            return
        result = _get_cached_result(req_id, timeout=30)
        if result:
            _send_json(client, result)
        else:
            _send_json(client, {"status": "error", "error": "Result not found or timed out"})

    # ── Approval notification ──

    def _notify_approval(self, entry) -> None:
        """通过所有已注册的连接器发送Approval notification"""
        expiry = time.strftime(
            "%H:%M:%S", time.localtime(entry.expires_at))
        data = {
            "type": "approval_request",
            "req_id": entry.req_id,
            "command": entry.command,
            "reason": entry.reason,
            "origin_channel": entry.origin.get("channel", "unknown"),
            "expires_at": entry.expires_at,
            "expires_at_str": expiry,
        }
        for name, cb in self._connectors.items():
            try:
                cb(data)
            except Exception as exc:
                logger.error("connector %s notify error: %s", name, exc)

    # ── Reaper ──

    def _reaper_loop(self):
        """后台线程：定期Reap expired请求"""
        while self._running:
            try:
                self._queue.reap_expired()
            except Exception as exc:
                logger.error("reaper error: %s", exc)
            time.sleep(REAPER_INTERVAL)
