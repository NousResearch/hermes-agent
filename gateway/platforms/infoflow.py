"""
Infoflow (如流) platform adapter for Baidu internal messaging.

Supports:
- WebSocket long connection for receiving messages
- HTTP API for sending messages
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Check if required dependencies are available
try:
    import aiohttp
    INFOFLOW_AVAILABLE = True
except ImportError:
    INFOFLOW_AVAILABLE = False


def check_infoflow_requirements() -> bool:
    """Check if Infoflow dependencies are available."""
    return INFOFLOW_AVAILABLE


def build_infoflow_jsapi_link(web_url: str) -> str:
    """Build Infoflow JSAPI deep link for sidebar or webview.

    Args:
        web_url: The web URL to open in the sidebar/webview

    Returns:
        Infoflow JSAPI deep link (infoflow://APICenter?data=...)
    """
    import base64
    import json
    from urllib.parse import quote

    jsapi_data = {
        "APIName": "BdHiJs.appnative.webview.loadURL",
        "version": 65,
        "data": {"type": "sidebar", "url": web_url}
    }
    # Must URL-encode the base64 data (OpenClaw's exact implementation)
    base64_data = base64.b64encode(json.dumps(jsapi_data, separators=(',', ':')).encode('utf-8')).decode('utf-8')
    return f"infoflow://APICenter?data={quote(base64_data)}"


def build_dashboard_deep_link(session_id: str, dashboard_base_url: str = "http://localhost:8081") -> str:
    """Build dashboard deep link for a session.

    Uses OpenClaw's Dashboard URL format: /dashboard/task/{date}/{taskId}

    Args:
        session_id: The session/task ID
        dashboard_base_url: Base URL of the dashboard server (should be OpenClaw's port 8081)

    Returns:
        Infoflow JSAPI deep link pointing to the session dashboard
    """
    from datetime import datetime

    # OpenClaw's exact URL format
    date_str = datetime.now().strftime("%Y-%m-%d")
    web_url = f"{dashboard_base_url}/dashboard/task/{date_str}/{session_id}"
    return build_infoflow_jsapi_link(web_url)


def create_clawguard_task(session_id: str, message: str, user_id: str = "") -> tuple:
    """Create a task file in ClawGuard's tasks directory for dashboard tracking.

    Returns (task_uuid, date_str, dashboard_url) or (None, None, None) on failure.
    """
    import uuid
    from datetime import datetime

    # OpenClaw plugin tasks directory (what ClawGuard actually reads)
    OPENCLAW_TASKS_DIR = Path.home() / ".openclaw" / "plugins" / "infoflow-private" / "tasks"
    DISPATCH_BASE = "https://sandbox-clawguard-dispatch.baidu-int.com"

    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        task_uuid = str(uuid.uuid4())
        task_id = f"{date_str}-{task_uuid}"  # OpenClaw format: date-uuid
        task_dir = OPENCLAW_TASKS_DIR / date_str
        task_dir.mkdir(parents=True, exist_ok=True)

        # Use UTC ISO format (OpenClaw uses Z suffix)
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        task_data = {
            "id": task_id,
            "sessionId": user_id or session_id,
            "message": message,
            "status": "running",
            "output": "",
            "error": "",
            "startTime": now_iso,
            "endTime": None,
            "modelName": "hermes",
            "models": ["hermes"],
            "events": [],
            "usage": {},
            "source": "hermes",
            "childTaskIds": [],
            "taskDate": date_str,
        }

        task_file = task_dir / f"{task_id}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)

        # Save session->task mapping for later completion
        runtime_dir = Path.home() / ".hermes" / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = runtime_dir / "clawguard_session_tasks.json"
        try:
            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    mapping = json.load(f)
            else:
                mapping = {}
            mapping[session_id] = {"uuid": task_uuid, "date": date_str, "task_id": task_id}
            with open(mapping_file, "w") as f:
                json.dump(mapping, f)
        except Exception:
            pass

        # Use dispatch/waiting URL (whitelisted in infoflow security gateway)
        dashboard_url = f"{DISPATCH_BASE}/dispatch/waiting/r0rr8sr9/{date_str}/{task_uuid}?channel_type=infoflow-plugin"
        logger.info(f"Infoflow: created ClawGuard task {task_uuid} at {task_file}")
        return task_uuid, date_str, dashboard_url
    except Exception as e:
        logger.warning(f"Infoflow: failed to create ClawGuard task: {e}")
        return None, None, None


def complete_clawguard_task(task_uuid: str, date_str: str, output: str, model_name: str = "") -> bool:
    """Mark a ClawGuard task as completed with its output."""
    from datetime import datetime, timezone

    OPENCLAW_TASKS_DIR = Path.home() / ".openclaw" / "plugins" / "infoflow-private" / "tasks"
    task_id = f"{date_str}-{task_uuid}"
    task_file = OPENCLAW_TASKS_DIR / date_str / f"{task_id}.json"

    try:
        if task_file.exists():
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        else:
            task_data = {}

        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        task_data["status"] = "completed"
        task_data["output"] = output
        task_data["endTime"] = now_iso
        if model_name:
            task_data["modelName"] = model_name

        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Infoflow: completed ClawGuard task {task_id}")
        return True
    except Exception as e:
        logger.warning(f"Infoflow: failed to complete ClawGuard task {task_id}: {e}")
        return False


def append_clawguard_task_event(task_uuid: str, date_str: str, event_type: str, event_data: dict) -> bool:
    """Append a single event to a running ClawGuard task file (for real-time progress)."""
    from datetime import datetime, timezone

    OPENCLAW_TASKS_DIR = Path.home() / ".openclaw" / "plugins" / "infoflow-private" / "tasks"
    task_id = f"{date_str}-{task_uuid}"
    task_file = OPENCLAW_TASKS_DIR / date_str / f"{task_id}.json"

    try:
        if not task_file.exists():
            return False
        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        events = task_data.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append({"type": event_type, "data": event_data, "timestamp": now_iso})
        task_data["events"] = events

        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, separators=(',', ':'))
        return True
    except Exception as e:
        logger.debug(f"Infoflow: failed to append event to ClawGuard task {task_id}: {e}")
        return False


def add_progress_button_if_acknowledgment(message: str, session_id: str = None, dashboard_base_url: str = "http://localhost:9119",
                                          clawguard_task_uuid: str = None, clawguard_date: str = None, clawguard_dashboard_url: str = None) -> str:
    """Add 'View Progress' button if message is an acknowledgment.

    Automatically detects acknowledgment messages and appends a button
    that links to the Kanban dashboard for real-time progress viewing.

    Args:
        message: The original message content
        session_id: Current session ID (optional, for dashboard link)
        dashboard_base_url: Base URL of the dashboard server
        clawguard_task_uuid: ClawGuard task UUID (if pre-created)
        clawguard_date: ClawGuard task date string
        clawguard_dashboard_url: ClawGuard dashboard URL for direct link

    Returns:
        Message with progress button appended if it's an acknowledgment
    """
    import re

    # Check if this is an acknowledgment message
    ack_patterns = [
        r'收到[！!]',
        r'好的[！!]',
        r'明白[！!]',
        r'马上开工',
        r'开始.*[执行处理分析]',
        r'正在.*[执行处理分析]',
    ]

    is_ack = any(re.search(pattern, message) for pattern in ack_patterns)

    # Only add button if:
    # 1. It's an acknowledgment message
    # 2. Doesn't already have buttons
    # 3. Session ID is available
    if is_ack and '[::button' not in message and session_id:
        if clawguard_dashboard_url:
            # Use ClawGuard dashboard link (HTTPS, whitelisted)
            jsapi_link = build_infoflow_jsapi_link(clawguard_dashboard_url)
            progress_button = (
                f"\n\n[::button-group layout=\"flow\"]\n"
                f"  [::button label=\"📊点我看过程\" url=\"{jsapi_link}\" style=\"primary\"]\n"
                f"  [::button label=\"💬查看进度\" query_send=\"show progress {session_id}\" style=\"primary\"]\n"
                f"[::button-group/]"
            )
        else:
            # Fallback to query_send button only
            progress_button = (
                f"\n\n[::button-group layout=\"flow\"]\n"
                f"  [::button label=\"📊点我看过程\" query_send=\"show progress {session_id}\" style=\"primary\"]\n"
                f"[::button-group/]"
            )

        return message + progress_button

    return message


class InfoflowAdapter(BasePlatformAdapter):
    """
    Infoflow (如流) platform adapter using WebSocket.

    Handles:
    - Receiving messages from users and groups
    - Sending responses with Markdown
    - WebSocket long connection
    """

    SUPPORTS_MESSAGE_EDITING = False  # Infoflow 不支持消息编辑

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.INFOFLOW)

        # Clear stale access token from env so dynamic acquisition takes over
        os.environ.pop("INFOFLOW_ACCESS_TOKEN", None)

        # Extract configuration
        extra = getattr(config, "extra", {}) or {}
        self.app_key = extra.get("app_key", os.getenv("INFOFLOW_APP_KEY", ""))
        self.app_secret = extra.get("app_secret", os.getenv("INFOFLOW_APP_SECRET", ""))
        self.robot_name = extra.get("robot_name", os.getenv("INFOFLOW_ROBOT_NAME", "Hermes"))
        self.api_host = extra.get("api_host", os.getenv("INFOFLOW_API_HOST", "https://apiin.im.baidu.com"))
        self.ws_gateway = extra.get("ws_gateway", os.getenv("INFOFLOW_WS_GATEWAY", "infoflow-open-gateway.weiyun.baidu.com"))

        if not self.app_key or not self.app_secret:
            raise ValueError("Infoflow: missing app_key or app_secret")

        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        self.connection_id: Optional[str] = None  # From Phase 1 API response

        # Token caching for send API
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        # Heartbeat mechanism (application layer)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._ping_interval: int = 60  # Default, will be updated by server config
        self._seq_id: int = 0  # Sequence number for frames
        self._last_pong_time: float = 0  # Track last pong response

        # Agent name resolution: maps group_id → {robot_name_lower: agent_id}
        # Built from incoming MESSAGE_RECEIVE body AT elements (like OpenClaw's mentionIdMap)
        self._agent_name_map: Dict[str, Dict[str, int]] = {}

        # Mention ID mapping (模仿 OpenClaw 的 mentionIdMap)
        # Stores mapping of names/IDs to types ("user" or "agent")
        self._mention_id_map: Dict[str, str] = {}  # lowercase name/id -> "user" or "agent"

        # Group member list cache: group_id -> {name_lower: agentId (int)}
        # Populated via /api/v1/robot/group/memberList, TTL 5 min
        self._group_agent_cache: Dict[str, Dict[str, int]] = {}
        self._group_cache_ts: Dict[str, float] = {}

        # Auto-mention: track last sender name per chat_id
        # When replying in a group, auto-@mention the last sender (bot or human)
        self._last_sender_agent: Dict[str, str] = {}
        self._GROUP_CACHE_TTL = 300.0  # seconds

        # Message deduplication: cache recent MsgIds to drop WS re-deliveries
        # Infoflow WebSocket often delivers the same message 2-3 times within seconds
        self._seen_msg_ids: dict = {}  # msg_id -> timestamp
        self._DEDUP_TTL = 30.0  # seconds to remember a seen msg_id

    @property
    def name(self) -> str:
        return f"Infoflow ({self.robot_name})"

    async def connect(self) -> bool:
        """Connect to Infoflow WebSocket."""
        if not INFOFLOW_AVAILABLE:
            logger.error("Infoflow: aiohttp not installed")
            return False

        try:
            self.session = aiohttp.ClientSession()
            ws_url = await self._get_ws_url()

            if not ws_url:
                logger.error("Infoflow: failed to get WebSocket URL")
                return False

            logger.info(f"Infoflow: connecting to {ws_url}")

            # Remove aiohttp heartbeat - we'll use application-layer heartbeat instead
            self.ws = await self.session.ws_connect(ws_url)
            self._mark_connected()
            logger.info(f"✓ Infoflow ({self.robot_name}) connected")

            # Start message listener and heartbeat
            self._ws_task = asyncio.create_task(self._listen_messages())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            return True

        except Exception as e:
            logger.error(f"Infoflow: connection error: {e}")
            self._set_fatal_error("connection_failed", str(e), retryable=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False

        # Stop heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Stop message listener
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        if self.session:
            await self.session.close()
            self.session = None

        self._mark_disconnected()
        logger.info("Infoflow: disconnected")

    async def _get_ws_url(self) -> Optional[str]:
        """Get WebSocket URL from Infoflow API using two-phase connection.

        Phase 1: Call /open/ws/endpoint API to get WebSocket URL
        Phase 2: Use returned URL to establish WebSocket connection

        This matches the NodeJS SDK implementation.
        """
        if not self.session:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

        # Phase 1: Get WebSocket endpoint via API
        # Use ws_gateway for endpoint API (not api_host)
        # Remove port if present, use HTTPS default 443
        gateway_host = self.ws_gateway.split(':')[0]
        endpoint_url = f"https://{gateway_host}/open/ws/endpoint"

        # Request body: app_key and app_secret (no signature needed)
        request_body = {
            "app_key": self.app_key,
            "app_secret": self.app_secret,
        }

        try:
            logger.info(f"Infoflow Phase 1: fetching WebSocket endpoint from {endpoint_url}")
            async with self.session.post(
                endpoint_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Infoflow Phase 1 response: {result}")

                    # Extract WebSocket URL from response
                    ws_url = result.get("url")
                    if ws_url:
                        logger.info(f"Infoflow: got WebSocket URL: {ws_url}")
                        # Store connection_id if present
                        self.connection_id = result.get("connection_id")
                        return ws_url

                    # Try nested data field
                    data = result.get("data", {})
                    ws_url = data.get("url")
                    if ws_url:
                        logger.info(f"Infoflow: got WebSocket URL from data: {ws_url}")
                        self.connection_id = data.get("connection_id")
                        return ws_url

                    logger.error(f"Infoflow: no WebSocket URL in response: {result}")
                else:
                    text = await resp.text()
                    logger.error(f"Infoflow Phase 1 failed: status={resp.status}, body={text}")
        except Exception as e:
            logger.error(f"Infoflow Phase 1 error: {e}")

        # Fallback: construct WebSocket URL directly (old behavior, may not work)
        ws_url = f"wss://{self.ws_gateway}/ws?app_key={self.app_key}"
        logger.warning(f"Infoflow: using fallback WebSocket URL: {ws_url}")
        return ws_url

    def _sign(self, params: Dict[str, Any]) -> str:
        """Generate signature for API request."""
        sorted_params = sorted(params.items())
        sign_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        sign_str += f"&app_secret={self.app_secret}"
        return hashlib.md5(sign_str.encode()).hexdigest()

    async def _get_access_token(self) -> Optional[str]:
        """Get access token for send API, with caching."""
        import time as time_module

        # Use pre-configured access token from environment (if set)
        env_token = os.getenv("INFOFLOW_ACCESS_TOKEN", "").strip()
        if env_token:
            return env_token

        # Check if token is still valid (with 5 min buffer)
        if self._access_token and time_module.time() < self._token_expires_at - 300:
            return self._access_token

        # Create a fresh session for token request (don't reuse self.session)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')

        # Use configured api_host (internal: apiin.im.baidu.com) for token API
        token_url = f"{self.api_host}/api/v1/auth/app_access_token"

        # App secret needs MD5 hash
        app_secret_md5 = hashlib.md5(self.app_secret.encode()).hexdigest()

        payload = {
            "app_key": self.app_key,
            "app_secret": app_secret_md5,
        }

        try:
            logger.info(f"Infoflow: requesting token from {token_url}")
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.post(
                    token_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    logger.info(f"Infoflow: token response status={resp.status}")
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"Infoflow: token response: {result}")
                        if result.get("code") == "ok":
                            # Token is in data.app_access_token
                            data = result.get("data", {})
                            self._access_token = data.get("app_access_token")
                            expires_in = data.get("expire") or data.get("expires_in", 7200)
                            self._token_expires_at = time_module.time() + expires_in
                            logger.info(f"Infoflow: got access token, expires in {expires_in}s")
                            return self._access_token
                        else:
                            logger.error(f"Infoflow: token error: {result}")
                    else:
                        text = await resp.text()
                        logger.error(f"Infoflow: token request failed: {resp.status} {text}")
        except Exception as e:
            logger.error(f"Infoflow: token request error: {e}")

        return None

    async def _fetch_group_member_list(self, group_id: str) -> Dict[str, int]:
        """
        Fetch group member list from /api/v1/robot/group/memberList.
        Returns a dict mapping agent name (lowercase) -> agentId (int).
        Results are cached for self._GROUP_CACHE_TTL seconds.
        """
        import time as time_module

        # Check cache
        ts = self._group_cache_ts.get(group_id, 0)
        if ts and time_module.time() - ts < self._GROUP_CACHE_TTL:
            return self._group_agent_cache.get(group_id, {})

        token = await self._get_access_token()
        if not token:
            logger.warning("Infoflow: _fetch_group_member_list: no token")
            return {}

        url = f"{self.api_host}/api/v1/robot/group/memberList"
        try:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.post(
                    url,
                    json={"groupId": int(group_id), "recallType": 0},
                    headers={
                        "Authorization": f"Bearer-{token}",
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Infoflow: memberList HTTP {resp.status}")
                        return {}
                    result = await resp.json()
                    if result.get("code") != "ok":
                        logger.error(f"Infoflow: memberList error: {result}")
                        return {}
                    outer = result.get("data", {})
                    if outer.get("errcode", 0) != 0:
                        logger.error(f"Infoflow: memberList errcode={outer.get('errcode')}: {outer.get('errmsg')}")
                        return {}
                    inner = outer.get("data", {})
                    agents = inner.get("agentInfoList", [])
                    name_map: Dict[str, int] = {}
                    for a in agents:
                        name = a.get("name", "")
                        agent_id = a.get("agentId")
                        if name and agent_id is not None:
                            name_map[name.lower()] = int(agent_id)
                    self._group_agent_cache[group_id] = name_map
                    self._group_cache_ts[group_id] = time_module.time()
                    logger.info(
                        f"Infoflow: memberList groupId={group_id}: "
                        + ", ".join(f"{n}->{i}" for n, i in name_map.items())
                    )
                    return name_map
        except Exception as e:
            logger.error(f"Infoflow: _fetch_group_member_list exception: {e}")
            return {}

    async def _listen_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        try:
            # Send initial handshake/authentication if needed
            await self._send_handshake()

            async for msg in self.ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_text_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_binary_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self.ws.pong()
                elif msg.type == aiohttp.WSMsgType.PONG:
                    # Handle pong
                    pass
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    logger.error(f"Infoflow: WebSocket closed: {self.ws.exception()}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Infoflow: listener error: {e}")
        finally:
            if self._running:
                # WebSocket disconnected unexpectedly - trigger reconnect
                logger.warning("Infoflow: WebSocket disconnected unexpectedly, triggering reconnect")
                self._mark_disconnected()
                self._set_fatal_error("ws_disconnected", "WebSocket connection closed", retryable=True)
                await self._notify_fatal_error()

    async def _send_handshake(self) -> None:
        """Send initial handshake/authentication message."""
        handshake_msg = {
            "type": "auth",
            "app_key": self.app_key,
            "timestamp": str(int(time.time() * 1000)),
        }
        handshake_msg["sign"] = self._sign(handshake_msg)

        try:
            await self.ws.send_str(json.dumps(handshake_msg))
            logger.debug("Infoflow: sent handshake message")
        except Exception as e:
            logger.error(f"Infoflow: failed to send handshake: {e}")

    def _next_seq_id(self) -> int:
        """Get next sequence ID for frames."""
        self._seq_id += 1
        return self._seq_id

    async def send_heartbeat(self) -> bool:
        """Send application-layer heartbeat (CONTROL frame with type=ping).

        Based on NodeJS SDK implementation:
        - method: FrameType.CONTROL (0)
        - headers: [{ key: 'type', value: 'ping' }]
        - payload: empty
        """
        if not self.ws or self.ws.closed:
            return False

        try:
            from gateway.platforms.infoflow_frame import encode_frame, FrameType, Frame, Header

            # Create CONTROL frame with ping
            frame = Frame(
                seq_id=self._next_seq_id(),
                log_id=str(int(time.time() * 1000)),
                service=0,
                method=FrameType.CONTROL,
                headers=[Header(key="type", value="ping")],
                payload=b"{}",
            )

            # Encode and send
            data = encode_frame(frame)
            await self.ws.send_bytes(data)
            logger.debug(f"Infoflow: sent heartbeat ping, seq_id={frame.seq_id}")
            return True

        except Exception as e:
            logger.error(f"Infoflow: failed to send heartbeat: {e}")
            return False

    async def send_ack(self, seq_id: int) -> bool:
        """Send ACK for a received DATA frame.

        Args:
            seq_id: The sequence ID of the DATA frame to acknowledge
        """
        if not self.ws or self.ws.closed:
            return False

        try:
            from gateway.platforms.infoflow_frame import encode_frame, FrameType, Frame, Header

            # Create CONTROL frame with ack
            frame = Frame(
                seq_id=self._next_seq_id(),
                log_id=str(int(time.time() * 1000)),
                service=0,
                method=FrameType.CONTROL,
                headers=[Header(key="type", value="ack"), Header(key="seq_id", value=str(seq_id))],
                payload=b"{}",
            )

            # Encode and send
            data = encode_frame(frame)
            await self.ws.send_bytes(data)
            logger.debug(f"Infoflow: sent ACK for seq_id={seq_id}")
            return True

        except Exception as e:
            logger.error(f"Infoflow: failed to send ACK: {e}")
            return False

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats.

        This is the application-layer heartbeat required by Infoflow server.
        The server expects ping frames every 'ping_interval' seconds.
        """
        import time as time_module

        # Wait for first config message to set ping_interval
        await asyncio.sleep(2)

        while self._running and self.ws and not self.ws.closed:
            try:
                # Send heartbeat
                success = await self.send_heartbeat()
                if not success:
                    logger.warning("Infoflow: heartbeat send failed, connection may be broken")

                # Wait for next interval (use 80% of ping_interval for safety margin)
                sleep_time = self._ping_interval * 0.8
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Infoflow: heartbeat loop error: {e}")
                await asyncio.sleep(5)

    async def _handle_text_message(self, data: str) -> None:
        """Handle text message from WebSocket."""
        try:
            msg = json.loads(data)
            await self._process_message(msg)
        except json.JSONDecodeError:
            logger.warning(f"Infoflow: invalid JSON: {data[:100]}")

    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle binary message (Protobuf Frame) from WebSocket.

        Infoflow uses Protobuf Frame format:
        - seq_id: sequence number
        - log_id: trace ID
        - service: service identifier
        - method: frame type (0=CONTROL, 1=DATA, 2=REQUEST, 3=RESPONSE)
        - headers: key-value pairs
        - payload: JSON bytes
        """
        try:
            from gateway.platforms.infoflow_frame import decode_frame, FrameType

            frame = decode_frame(data)
            logger.debug(f"Infoflow: received frame seq_id={frame.seq_id}, method={frame.method}, service={frame.service}")

            # Parse payload as JSON
            payload_str = frame.payload.decode('utf-8') if frame.payload else '{}'

            # Handle different frame types
            if frame.method == FrameType.CONTROL:
                # Control frame - heartbeat or config
                logger.debug(f"Infoflow: CONTROL frame payload: {payload_str}")
                try:
                    payload = json.loads(payload_str)

                    # Check for pong response
                    headers_dict = {h.key: h.value for h in frame.headers}
                    if headers_dict.get("type") == "pong":
                        self._last_pong_time = time.time()
                        logger.debug(f"Infoflow: received pong response")

                    # Handle config update (server sends ping_interval config)
                    if 'ping_interval' in payload:
                        self._ping_interval = payload['ping_interval']
                        logger.info(f"Infoflow: updated ping_interval to {self._ping_interval}s from config: {payload}")
                    elif 'reconnect_count' in payload:
                        # Initial config message
                        logger.info(f"Infoflow: received config: {payload}")
                        if 'ping_interval' in payload:
                            self._ping_interval = payload['ping_interval']
                            logger.info(f"Infoflow: set ping_interval to {self._ping_interval}s")

                except json.JSONDecodeError:
                    pass

            elif frame.method == FrameType.DATA:
                # Data frame - actual message
                logger.info(f"Infoflow: DATA frame payload: {payload_str}")

                # Send ACK for the DATA frame (required by server)
                asyncio.create_task(self.send_ack(frame.seq_id))

                try:
                    payload = json.loads(payload_str)
                    # Headers may contain metadata
                    headers_dict = {h.key: h.value for h in frame.headers}
                    # Merge headers into payload for processing
                    if headers_dict:
                        payload['_headers'] = headers_dict
                    await self._process_message(payload)
                except json.JSONDecodeError as e:
                    logger.warning(f"Infoflow: invalid JSON payload: {payload_str[:100]}")

            elif frame.method == FrameType.RESPONSE:
                # Response frame
                logger.debug(f"Infoflow: RESPONSE frame: {payload_str}")

            else:
                logger.warning(f"Infoflow: unknown frame method={frame.method}")

        except Exception as e:
            logger.error(f"Infoflow: binary message error: {e}")
            # Fallback: try old zlib decompression
            try:
                decompressed = zlib.decompress(data)
                text = decompressed.decode("utf-8")
                await self._handle_text_message(text)
            except:
                logger.error(f"Infoflow: fallback also failed")

    async def _process_message(self, msg: Dict[str, Any]) -> None:
        """Process a parsed message."""
        # 如流消息字段可能是大写(MsgType)或小写(msg_type)
        msg_type = msg.get("type") or msg.get("msg_type") or msg.get("MsgType") or msg.get("Msg_type", "")
        msg_type_lower = msg_type.lower() if msg_type else ""

        # DEBUG: dump raw message for file type
        if msg_type_lower == "file":
            import json as _json
            logger.info(f"Infoflow RAW FILE MSG: {_json.dumps(msg, ensure_ascii=False)[:2000]}")

        # Ignore pong and system events
        if msg_type_lower == "pong":
            return

        # 忽略事件类型消息（如用户进入聊天窗口）
        if msg_type_lower == "event":
            event_type = msg.get("Event") or msg.get("event", "")
            logger.debug(f"Infoflow: ignoring event message: {event_type}")
            return

        # Handle new event-type MESSAGE_RECEIVE format (new version bot group/DM messages)
        # Format: {"eventtype":"MESSAGE_RECEIVE", "message":{"header":{...}, "body":[...]}, ...}
        if msg.get("eventtype") == "MESSAGE_RECEIVE":
            msg_data = msg.get("message", {})
            header = msg_data.get("header", {})
            body = msg_data.get("body", [])

            # Extract text from body array (concatenate TEXT type segments)
            text_parts = []
            for part in body:
                if isinstance(part, dict):
                    if part.get("type") in ("TEXT", "MD"):
                        text_parts.append(part.get("content", ""))

            content = "".join(text_parts).strip()
            is_group = header.get("totype") == "GROUP"

            # Build agent name → agent_id mapping from body AT elements
            # This mirrors OpenClaw's mentionIdMap approach: learn robot names from incoming messages
            if is_group:
                group_id = str(msg.get("groupid") or header.get("toid", ""))
                if group_id:
                    if group_id not in self._agent_name_map:
                        self._agent_name_map[group_id] = {}
                    for part in body:
                        if isinstance(part, dict) and part.get("type") == "AT":
                            # Robot mention: has robotid + name
                            robot_id = part.get("robotid")
                            name = part.get("name", "")

                            # Learn mapping (模仿 OpenClaw 的 mentionIdMap)
                            if robot_id:
                                # This is a robot/agent
                                key = str(robot_id).lower()
                                self._mention_id_map[key] = "agent"
                                logger.info(f"Infoflow: learned agent mapping by ID: {robot_id} -> agent")
                                if name:
                                    name_key = name.lower()
                                    self._mention_id_map[name_key] = "agent"
                                    logger.info(f"Infoflow: learned agent mapping by name: '{name}' -> agent (id: {robot_id})")
                            elif part.get("userid"):
                                # This is a user
                                user_id = part.get("userid")
                                key = str(user_id).lower()
                                self._mention_id_map[key] = "user"
                                logger.info(f"Infoflow: learned user mapping by ID: {user_id} -> user")
                                if name:
                                    name_key = name.lower()
                                    self._mention_id_map[name_key] = "user"
                                    logger.info(f"Infoflow: learned user mapping by name: '{name}' -> user (id: {user_id})")
                            if robot_id and name:
                                self._agent_name_map[group_id][name.lower()] = int(robot_id)
                                logger.debug(f"Infoflow: learned agent name '{name}' → agent_id={robot_id} in group {group_id}")
                            # Also track user AT for completeness
                            user_id = part.get("userid")
                            if user_id and name:
                                self._agent_name_map[group_id][name.lower()] = -1  # -1 = regular user

            if content:
                # Normalize to legacy format for _handle_chat_message
                # Note: messageid and groupid are integers in the new format, convert to str
                normalized = {
                    "FromUserId": str(header.get("fromuserid") or msg.get("fromid", "")),
                    "Content": content,
                    "MsgType": "text",
                    "MsgId": str(header.get("messageid", "")),
                    "MsgId2": str(msg.get("msgid2", "")),
                }
                if is_group:
                    normalized["group_id"] = str(msg.get("groupid") or header.get("toid", ""))

                logger.info(f"Infoflow: parsed MESSAGE_RECEIVE -> user={normalized['FromUserId']}, group={is_group}, text='{content[:80]}'")
                await self._handle_chat_message(normalized)
            else:
                logger.debug("Infoflow: MESSAGE_RECEIVE with no text content, skipped")
            return

        # Handle chat message - 支持多种消息类型标识
        if msg_type_lower in ("msg", "message", "chat", "text"):
            await self._handle_chat_message(msg)
        # 如果有 FromUserId/Content 等字段，也当作聊天消息处理
        elif msg.get("FromUserId") or msg.get("from_user_id") or msg.get("Content") or msg.get("content"):
            await self._handle_chat_message(msg)

    async def _handle_chat_message(self, msg: Dict[str, Any]) -> None:
        """Handle chat message."""
        from gateway.session import SessionSource

        # 支持大小写字段名
        content = msg.get("Content") or msg.get("content") or msg.get("text") or msg.get("body", "")
        from_user = msg.get("FromUserId") or msg.get("from_user_id") or msg.get("from") or msg.get("sender", "")
        from_name = msg.get("FromUserName") or msg.get("from_user_name") or msg.get("sender_name", "")
        chat_id = msg.get("chat_id") or msg.get("group_id") or msg.get("conversation_id") or msg.get("GroupId") or msg.get("ConversationId", "")
        msg_id = msg.get("MsgId") or msg.get("msg_id") or msg.get("message_id") or msg.get("MessageId", "")
        msg_id2 = msg.get("MsgId2") or msg.get("msg_id2") or msg.get("msgid2", "")

        # Deduplication: Infoflow WS re-delivers the same message multiple times
        if msg_id:
            import time as _time
            _now = _time.monotonic()
            # Evict stale entries
            self._seen_msg_ids = {k: v for k, v in self._seen_msg_ids.items() if _now - v < self._DEDUP_TTL}
            if msg_id in self._seen_msg_ids:
                logger.debug("Infoflow: dropping duplicate MsgId=%s", msg_id)
                return
            self._seen_msg_ids[msg_id] = _now

        # Determine if group or private chat
        is_group = bool(chat_id and chat_id != from_user)

        # 群聊时记录发送者，用于自动 @回复（包括人和机器人）
        if is_group and chat_id and from_user:
            self._last_sender_agent[str(chat_id)] = from_user
            logger.info(f"Infoflow: auto-mention: stored sender {from_user} for chat {chat_id}")

        # 单聊时发送确认表情（收到）
        if not is_group and from_user and msg_id and msg_id2:
            # 异步发送确认，不阻塞消息处理
            asyncio.create_task(self._send_receipt_confirmation(from_user, msg_id, msg_id2))

        # Create SessionSource
        source = SessionSource(
            platform=Platform.INFOFLOW,
            chat_id=chat_id if is_group else from_user,
            chat_type="group" if is_group else "dm",
            user_id=from_user,
            user_name=from_name,
        )

        # Create MessageEvent
        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=msg,
            message_id=msg_id,
        )

        # 写入消息上下文文件供 MCP 工具使用
        self._write_message_context(
            msg_id=msg_id,
            msg_id2=msg_id2,
            from_user=from_user,
            chat_id=chat_id if is_group else "",
            chat_type=7 if not is_group else 2,  # 7=私聊, 2=群聊
        )

        # Use handle_message() from base class which properly handles
        # response sending, session management, and retries
        logger.info("Infoflow: _handle_chat_message -> handle_message: session=%s, text=%r", source.user_id, (content or "")[:50])
        await self.handle_message(event)


    async def _send_receipt_confirmation(self, user_id: str, msg_id: str, msg_id2: str) -> None:
        """发送收到确认表情（单聊）- 通过 MCP 服务。

        使用 infoflow-emoji MCP 服务发送表情，避免重复实现 token 获取逻辑。

        Args:
            user_id: 用户ID
            msg_id: 消息ID (baseMsgId)
            msg_id2: 消息ID2
        """
        try:
            from tools.mcp_tool import _servers, _lock, _run_on_mcp_loop, _ensure_mcp_loop

            # 确保 MCP 事件循环已启动
            _ensure_mcp_loop()

            # 获取 MCP 服务
            with _lock:
                server = _servers.get("infoflow-emoji")

            if not server or not server.session:
                logger.warning("Infoflow: infoflow-emoji MCP server not connected, skipping emoji confirmation")
                return

            # 调用 MCP 工具
            async def _call_mcp():
                result = await server.session.call_tool("infoflow_emoji_reply", arguments={
                    "emojiId": "d101",
                    "action": "add"
                })
                return result

            result = _run_on_mcp_loop(_call_mcp(), timeout=10)

            # 解析结果
            if hasattr(result, 'isError') and result.isError:
                error_text = ""
                for block in (result.content or []):
                    if hasattr(block, "text"):
                        error_text += block.text
                logger.warning(f"Infoflow: MCP emoji reply error: {error_text}")
            else:
                # 成功
                logger.info(f"Infoflow: ✓ sent emoji confirmation (d101) to {user_id} via MCP")

        except Exception as e:
            logger.error(f"Infoflow: error sending emoji confirmation via MCP: {e}")

    def _write_message_context(
        self,
        msg_id: str,
        msg_id2: str,
        from_user: str,
        chat_id: str,
        chat_type: int,
    ) -> None:
        """写入消息上下文文件供 MCP 工具使用。

        Args:
            msg_id: 消息ID (baseMsgId)
            msg_id2: 消息ID2
            from_user: 发送者用户ID
            chat_id: 聊天ID（群聊时有值）
            chat_type: 聊天类型（2=群聊, 7=私聊）
        """
        try:
            context = {
                "msg_id": msg_id,
                "msg_id2": msg_id2,
                "from_user": from_user,
                "chat_id": chat_id,
                "chat_type": chat_type,
                "timestamp": time.time(),
            }

            # 写入 Hermes runtime 目录
            runtime_dir = Path.home() / ".hermes" / "runtime"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            context_file = runtime_dir / "infoflow_context.json"

            with open(context_file, "w") as f:
                json.dump(context, f, indent=2)

            logger.debug(f"Infoflow: wrote message context to {context_file}")
        except Exception as e:
            logger.warning(f"Infoflow: failed to write message context: {e}")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message via Infoflow API.

        Uses api.im.baidu.com API with Bearer token authentication.
        """
        # Auto-add progress button for acknowledgment messages
        if metadata and metadata.get("session_key"):
            _session_key = metadata.get("session_key")
            _user_id = metadata.get("user_id") or chat_id or ""

            # Create ClawGuard task for dashboard tracking
            _cg_uuid, _cg_date, _cg_dashboard_url = create_clawguard_task(
                session_id=_session_key,
                message=content[:500],  # truncate for storage
                user_id=_user_id,
            )

            # Store task info in metadata for later completion
            if metadata is not None and _cg_uuid:
                metadata["clawguard_task_uuid"] = _cg_uuid
                metadata["clawguard_task_date"] = _cg_date

            content = add_progress_button_if_acknowledgment(
                content,
                session_id=_session_key,
                clawguard_task_uuid=_cg_uuid,
                clawguard_date=_cg_date,
                clawguard_dashboard_url=_cg_dashboard_url,
            )

        if not self.session:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

        # Get access token
        token = await self._get_access_token()
        if not token:
            return SendResult(success=False, error="Failed to get access token")

        # Build payload according to Infoflow API spec
        # touser: user_id for private chat, group_id for group chat
        is_group = metadata and metadata.get("is_group")

        # If metadata is None, infer from chat_id format
        # Group IDs are numeric strings (e.g., "12841364"), DM IDs are usernames (e.g., "anjianjun01")
        if is_group is None and chat_id and chat_id.isdigit():
            is_group = True
            logger.info(f"Infoflow: inferred is_group=True from numeric chat_id: {chat_id}")

        logger.info(f"Infoflow: send called, chat_id={chat_id}, is_group={is_group}, content_length={len(content)}")
        # Log first 100 chars of content for debugging
        if content and len(content) > 0:
            preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(f"Infoflow: sending content preview: {preview}")

        # Check for mentions in metadata
        mentions = metadata.get("mentions") if metadata else None
        if mentions:
            logger.info(f"Infoflow: mentions to include: {mentions}")

        if is_group:
            # Group chat uses dedicated endpoint with different payload format
            # Ref: "以机器人身份发送消息到群聊" doc in 如流开放平台知识库
            url = f"{self.api_host}/api/v1/robot/msg/groupmsgsend"
            client_msg_id = int(time.time() * 1000)

            # Normalize mentions
            mentions_list = mentions if isinstance(mentions, list) else ([mentions] if mentions else [])

            # 优先使用 metadata 中的 mentions
            # 只有在 metadata 中没有 mentions 时才用 auto-sender
            if not mentions_list:
                # Auto-mention: always @mention the last sender (bot or human)
                auto_sender = self._last_sender_agent.get(chat_id)
                if auto_sender is not None:
                    mentions_list.append(auto_sender)
                    logger.info(f"Infoflow: auto-mention: adding '{auto_sender}' for chat {chat_id}")

                # NOTE: 不从回复内容中提取 @xxx —— Agent 回复里出现的 @xxx 通常是引用/解释，
                # 并非真的要 @ 人。从内容提取会导致误触发（如回复里提到 @dodo 就真的 @dodo）。
                # 如需主动 @ 某人，请通过 metadata["mentions"] 传入。

            # Build header
            header = {
                "toid": int(chat_id),
                "totype": "GROUP",
                "clientmsgid": client_msg_id,
                "role": "robot"
            }

            # Build body — AT fragments go in body, NOT in header
            body = []

            if mentions_list:
                # Fetch group member list to resolve agent names -> agentIds
                try:
                    group_agent_map = await self._fetch_group_member_list(chat_id)
                except Exception:
                    group_agent_map = {}

                # Merge with learned _agent_name_map for this group
                local_map = self._agent_name_map.get(chat_id, {})

                # Separate mentions into agents (robots) vs users
                at_agent_ids: List[int] = []   # numeric agentIds for robots
                at_user_ids: List[str] = []     # string userIds for humans

                for mention in mentions_list:
                    mention_lower = mention.lower()

                    # 1. Check group member list cache first (name -> agentId)
                    if mention_lower in group_agent_map:
                        agent_id = group_agent_map[mention_lower]
                        if agent_id not in at_agent_ids:
                            at_agent_ids.append(agent_id)
                        logger.info(f"Infoflow: resolved '{mention}' -> agent agentId={agent_id} (memberList)")
                        continue

                    # 2. Check locally-learned agent name map (from incoming AT elements)
                    if mention_lower in local_map:
                        agent_id = local_map[mention_lower]
                        if agent_id > 0 and agent_id not in at_agent_ids:
                            at_agent_ids.append(agent_id)
                            logger.info(f"Infoflow: resolved '{mention}' -> agent agentId={agent_id} (local map)")
                        elif agent_id == -1:
                            # -1 means regular user
                            if mention not in at_user_ids:
                                at_user_ids.append(mention)
                        continue

                    # 3. Check mention_id_map (type learned from incoming messages)
                    id_type = self._mention_id_map.get(mention_lower)
                    if id_type == "agent":
                        # Already a numeric ID string from _agent_name_map learning
                        if mention.isdigit():
                            agent_id = int(mention)
                            if agent_id not in at_agent_ids:
                                at_agent_ids.append(agent_id)
                        logger.info(f"Infoflow: '{mention}' known as agent (mention_id_map)")
                        continue
                    elif id_type == "user":
                        if mention not in at_user_ids:
                            at_user_ids.append(mention)
                        continue

                    # 4. Fallback: unknown — skip, don't pollute the AT node
                    logger.info(f"Infoflow: unknown mention '{mention}', skipping")

                # Strip resolved mentions from content so they don't appear twice
                # (AT node handles the @ display)
                if at_user_ids or at_agent_ids:
                    resolved = [str(aid) for aid in at_agent_ids] + at_user_ids
                    for name in resolved:
                        content = re.sub(rf'@\s*{re.escape(name)}', '', content)
                    content = re.sub(r' +', ' ', content).strip()
                    logger.info(f"Infoflow: stripped resolved mentions from content")

                # Use TEXT type for @mentions
                header["msgtype"] = "TEXT"
                body.append({"type": "TEXT", "content": content})

                # Filter out own bot agent ID - API rejects self-mention
                own_agent_id = 30534  # Hermes bot ID
                if own_agent_id in at_agent_ids:
                    at_agent_ids.remove(own_agent_id)
                    logger.info(f"Infoflow: filtered out own agent_id={own_agent_id} from mentions")

                # Build AT node - only if we have resolved users or agents
                if at_user_ids or at_agent_ids:
                    at_node: Dict[str, Any] = {"type": "AT", "atall": False}
                    if at_user_ids:
                        at_node["atuserids"] = at_user_ids
                    if at_agent_ids:
                        at_node["atagentids"] = at_agent_ids
                    body.insert(0, at_node)
                    logger.info(f"Infoflow: AT node: users={at_user_ids}, agents={at_agent_ids}")
                else:
                    logger.info(f"Infoflow: no resolved mentions, skipping AT node")
            else:
                header["msgtype"] = "TEXT"
                body.append({"type": "TEXT", "content": content})

            payload = {
                "agentid": 30534,  # Hermes bot ID from incoming messages
                "message": {
                    "header": header,
                    "body": body
                }
            }

            # 记录完整的 payload 用于调试
            import json
            payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
            logger.info(f"Infoflow: group send payload (完整):\n{payload_str[:2000]}")
            logid = str(client_msg_id)
            logger.info(f"Infoflow: group send payload to {url}, toid={int(chat_id)}")
        else:
            # DM uses app/message/send endpoint
            url = f"{self.api_host}/api/v1/app/message/send"
            payload = {
                "touser": chat_id,
                "msgtype": "md",
                "md": {
                    "content": content
                }
            }
            logid = str(int(time.time() * 1000))
            logger.info(f"Infoflow: DM send payload to {url}")

        try:
            async with self.session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer-{token}",
                    "Content-Type": "application/json",
                    "LOGID": logid,
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get("code") == "ok":
                        if is_group:
                            # Group API response format
                            data = result.get("data", {})
                            if isinstance(data, dict) and data.get("errcode") == 0:
                                msg_data = data.get("data", {})
                                msg_id = msg_data.get("messageid")
                                if msg_id:
                                    return SendResult(success=True, message_id=str(msg_id))
                                else:
                                    logger.warning(f"Infoflow: group send success but no messageid: {result}")
                                    return SendResult(success=True, message_id=None)
                            else:
                                err_msg = data.get("errmsg", str(result))
                                logger.error(f"Infoflow: group send API error: {err_msg}")
                                return SendResult(success=False, error=err_msg)
                        else:
                            # DM API response format
                            msg_id = result.get("msgid") or result.get("msg_id")
                            return SendResult(success=True, message_id=msg_id)
                    else:
                        logger.error(f"Infoflow: send API error: {result}")
                        return SendResult(success=False, error=str(result.get("code")))
                else:
                    error = await resp.text()
                    logger.error(f"Infoflow: send failed: {error}")
                    return SendResult(success=False, error=error)

        except Exception as e:
            logger.error(f"Infoflow: send error: {e}")
            return SendResult(success=False, error=str(e))

    async def send_message(
        self,
        target: Any,
        text: str,
        *,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a message via Infoflow API (compatibility method)."""
        chat_id = getattr(target, "chat_id", None) or getattr(target, "user_id", None)
        is_group = bool(getattr(target, "chat_id", None))
        metadata = kwargs.get("metadata", {"is_group": is_group})
        if isinstance(metadata, dict):
            metadata["is_group"] = is_group
        else:
            metadata = {"is_group": is_group}
        # Pass mentions from kwargs if provided
        if "mentions" in kwargs:
            metadata["mentions"] = kwargs["mentions"]
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_typing_indicator(self, target: Any) -> None:
        """Send typing indicator (not supported)."""
        pass

    async def mark_read(self, target: Any) -> None:
        """Mark message as read (not supported)."""
        pass

    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user info from Infoflow."""
        if not self.session:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

        params = {
            "app_key": self.app_key,
            "timestamp": str(int(time.time() * 1000)),
            "user_id": user_id,
        }
        params["sign"] = self._sign(params)

        url = f"{self.api_host}/openapi/user/info"

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}", "user_id": user_id}
        except Exception as e:
            logger.error(f"Infoflow: get user info error: {e}")
            return {"error": str(e), "user_id": user_id}

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get chat/group info from Infoflow."""
        if not self.session:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations('/etc/ssl/certs/ca-certificates.crt')
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

        params = {
            "app_key": self.app_key,
            "timestamp": str(int(time.time() * 1000)),
            "chat_id": chat_id,
        }
        params["sign"] = self._sign(params)

        url = f"{self.api_host}/openapi/chat/info"

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}", "chat_id": chat_id}
        except Exception as e:
            logger.error(f"Infoflow: get chat info error: {e}")
            return {"error": str(e), "chat_id": chat_id}

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an approval request for a dangerous command.

        Uses Infoflow's interactive button syntax to provide a better user experience.
        Buttons send standard /approve commands that work with Hermes's FIFO approval system.
        """
        cmd_preview = command[:200] + "..." if len(command) > 200 else command

        # Use Infoflow's interactive button syntax
        # Buttons send standard /approve commands that work with existing approval logic
        msg = (
            f"⚠️ **危险命令需要审批**\n"
            f"```\n{cmd_preview}\n```\n"
            f"原因: {description}\n\n"
            f"[::button-group layout=\"flow\"]\n"
            f"  [::button label=\"允许一次\" query_send=\"/approve\" style=\"primary\"]\n"
            f"  [::button label=\"本次会话允许\" query_send=\"/approve session\" style=\"primary\"]\n"
            f"  [::button label=\"永久允许\" query_send=\"/approve always\" style=\"primary\"]\n"
            f"  [::button label=\"拒绝\" query_send=\"/deny\" style=\"danger\"]\n"
            f"[::button-group/]"
        )
        return await self.send(chat_id, msg, metadata=metadata)
