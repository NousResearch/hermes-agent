"""
Wechaty personal WeChat platform adapter for Hermes Agent.

Both directions run through a supervised Node sidecar (``sidecar/index.mjs``)
that hosts the Wechaty bot. Inbound messages are streamed as NDJSON on
``GET /inbound``; outbound ``send`` / media helpers POST to loopback control
endpoints authenticated with a shared bearer token.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import quote

if TYPE_CHECKING:
    import httpx

    HTTPX_AVAILABLE = True
else:
    try:
        import httpx

        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
        httpx = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import strip_markdown
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_SIDECAR_PORT = 8790
_DEFAULT_SIDECAR_BIND = "127.0.0.1"
_MAX_MESSAGE_LENGTH = 4000
_DEDUP_MAX_SIZE = 2000
_DEDUP_WINDOW_SECONDS = 5 * 60
_SIDECAR_DIR = Path(__file__).parent / "sidecar"

_DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]


def _coerce_port(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def check_requirements() -> bool:
    if not HTTPX_AVAILABLE:
        return False
    if not shutil.which(os.getenv("WECHATY_NODE_BIN") or "node"):
        return False
    return (_SIDECAR_DIR / "node_modules").exists()


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    token = (
        os.getenv("WECHATY_PUPPET_SERVICE_TOKEN")
        or os.getenv("WECHATY_TOKEN")
        or extra.get("puppet_token")
        or ""
    ).strip()
    puppet = (os.getenv("WECHATY_PUPPET") or extra.get("puppet") or "").strip()
    # Tokenless puppets (e.g. wechaty-puppet-wechat4u) are valid when puppet is set.
    return bool(token or puppet)


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env for status / auto-enable."""
    seed: dict = {}
    puppet = (os.getenv("WECHATY_PUPPET") or "").strip()
    token = (
        os.getenv("WECHATY_PUPPET_SERVICE_TOKEN")
        or os.getenv("WECHATY_TOKEN")
        or ""
    ).strip()
    if puppet:
        seed["puppet"] = puppet
    if token:
        seed["puppet_token"] = token
    home = os.getenv("WECHATY_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("WECHATY_HOME_CHANNEL_NAME", "Home"),
        }
    return seed or None


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: PlatformConfig) -> Optional[dict]:
    """Bridge ``gateway.platforms.wechaty`` YAML keys into env / extra."""
    extra = platform_cfg.extra or {}
    seed: dict = {}
    mapping = {
        "puppet": "WECHATY_PUPPET",
        "puppet_token": "WECHATY_PUPPET_SERVICE_TOKEN",
        "bot_name": "WECHATY_BOT_NAME",
        "sidecar_port": "WECHATY_SIDECAR_PORT",
        "require_mention": "WECHATY_REQUIRE_MENTION",
        "mention_patterns": "WECHATY_MENTION_PATTERNS",
        "allowed_users": "WECHATY_ALLOWED_USERS",
        "allow_all_users": "WECHATY_ALLOW_ALL_USERS",
    }
    for key, env_name in mapping.items():
        if key in extra and not os.getenv(env_name):
            val = extra[key]
            if val is not None and str(val).strip():
                os.environ[env_name] = str(val)
                seed[key] = val
    home = extra.get("home_channel")
    if home and not os.getenv("WECHATY_HOME_CHANNEL"):
        if isinstance(home, dict):
            chat_id = (home.get("chat_id") or "").strip()
        else:
            chat_id = str(home).strip()
        if chat_id:
            os.environ["WECHATY_HOME_CHANNEL"] = chat_id
            seed["home_channel"] = chat_id
    return seed or None


class WechatyAdapter(BasePlatformAdapter):
    """Bidirectional Wechaty bridge via the Node sidecar."""

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("wechaty"))
        extra = config.extra or {}

        self._sidecar_port = _coerce_port(
            extra.get("sidecar_port") or os.getenv("WECHATY_SIDECAR_PORT"),
            _DEFAULT_SIDECAR_PORT,
        )
        self._sidecar_bind = _DEFAULT_SIDECAR_BIND
        self._sidecar_token = (
            os.getenv("WECHATY_SIDECAR_TOKEN") or secrets.token_hex(16)
        )
        self._autostart_sidecar = _truthy(
            extra.get("sidecar_autostart")
            if "sidecar_autostart" in extra
            else os.getenv("WECHATY_SIDECAR_AUTOSTART"),
            default=True,
        )
        self._node_bin = os.getenv("WECHATY_NODE_BIN") or shutil.which("node") or "node"

        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("WECHATY_REQUIRE_MENTION", "true")
        self.require_mention = _truthy(_require_mention, default=True)

        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._sidecar_supervisor_task: Optional[asyncio.Task] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._inbound_running = False
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._seen_messages: Dict[str, float] = {}
        self._logged_in = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not HTTPX_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP", "httpx not installed", retryable=False
            )
            return False
        if not validate_config(self.config):
            self._set_fatal_error(
                "MISSING_CONFIG",
                "Configure WECHATY_PUPPET + WECHATY_PUPPET_SERVICE_TOKEN "
                "(or a tokenless puppet like wechaty-puppet-wechat4u) in "
                "~/.hermes/.env or platforms.wechaty.extra",
                retryable=False,
            )
            return False

        self._http_client = httpx.AsyncClient(timeout=30.0)

        if self._autostart_sidecar:
            try:
                await self._start_sidecar()
            except Exception as exc:
                self._set_fatal_error(
                    "SIDECAR_FAILED",
                    f"failed to start Wechaty sidecar: {exc}",
                    retryable=True,
                )
                await self._http_client.aclose()
                self._http_client = None
                return False
        else:
            logger.warning(
                "[wechaty] sidecar autostart disabled — messaging will fail"
            )

        self._inbound_running = True
        self._inbound_task = asyncio.get_event_loop().create_task(
            self._inbound_loop()
        )
        self._mark_connected()
        logger.info(
            "[wechaty] connected — sidecar on %s:%d",
            self._sidecar_bind,
            self._sidecar_port,
        )
        return True

    async def disconnect(self) -> None:
        self._inbound_running = False
        if self._inbound_task is not None:
            self._inbound_task.cancel()
            try:
                await self._inbound_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._inbound_task = None
        await self._stop_sidecar()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._mark_disconnected()

    async def _inbound_loop(self) -> None:
        client = self._http_client
        if client is None:
            return
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}/inbound"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        backoff = 1.0
        while self._inbound_running:
            try:
                async with client.stream(
                    "GET", url, headers=headers, timeout=None
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"/inbound returned {resp.status_code}")
                    backoff = 1.0
                    async for line in resp.aiter_lines():
                        if not self._inbound_running:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        await self._on_inbound_line(line)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._inbound_running:
                    break
                logger.warning(
                    "[wechaty] inbound stream dropped (%s); reconnecting in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _on_inbound_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        etype = event.get("type")
        if etype == "message":
            msg_id = event.get("messageId")
            if msg_id and self._is_duplicate(msg_id):
                return
            try:
                await self._dispatch_message(event)
            except Exception:
                logger.exception("[wechaty] inbound dispatch failed")
        elif etype == "scan":
            status = event.get("status", "")
            qrcode = event.get("qrcode") or ""
            if qrcode:
                qr_url = (
                    "https://wechaty.js.org/qrcode/"
                    + quote(qrcode, safe="")
                )
                logger.info(
                    "[wechaty] scan QR (status=%s) — open in browser, then scan "
                    "with WeChat 扫一扫: %s",
                    status,
                    qr_url,
                )
            else:
                logger.info(
                    "[wechaty] scan QR (status=%s) — check gateway logs",
                    status,
                )
        elif etype == "login":
            self._logged_in = True
            logger.info(
                "[wechaty] logged in as %s (%s)",
                event.get("userName"),
                event.get("userId"),
            )
        elif etype == "logout":
            self._logged_in = False
            logger.warning("[wechaty] logged out: %s", event.get("userName"))

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        seen = self._seen_messages
        prev = seen.get(msg_id)
        if prev is not None and now - prev < _DEDUP_WINDOW_SECONDS:
            return True
        if msg_id in seen:
            del seen[msg_id]
        seen[msg_id] = now
        if len(seen) > _DEDUP_MAX_SIZE:
            for old in list(seen.keys())[: len(seen) - _DEDUP_MAX_SIZE]:
                del seen[old]
        return False

    async def _dispatch_message(self, event: Dict[str, Any]) -> None:
        chat_id = event.get("chatId") or ""
        if not chat_id:
            logger.warning("[wechaty] inbound missing chatId")
            return

        chat_type = event.get("chatType") or "dm"
        text = (event.get("text") or "").strip()
        mention_self = bool(event.get("mentionSelf"))

        if chat_type == "group" and self.require_mention and not mention_self:
            if not self._message_matches_mention_patterns(text):
                logger.debug(
                    "[wechaty] ignoring group message (require_mention, no @)"
                )
                return
            text = self._clean_mention_text(text)

        media_urls: List[str] = []
        media_types: List[str] = []
        mtype = MessageType.TEXT
        attachment = event.get("attachment") or {}
        if attachment.get("kind") == "image":
            cached = _cache_inbound_image(attachment)
            if cached:
                media_urls.append(cached)
                media_types.append(attachment.get("mimeType") or "image/jpeg")
                mtype = MessageType.PHOTO
            elif attachment.get("tooLarge"):
                text = (text + "\n" if text else "") + "[image too large]"
            elif not text:
                text = "[image]"

        if not text and not media_urls:
            text = "[empty message]"

        ts_str = event.get("timestamp") or ""
        try:
            timestamp = (
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_str
                else datetime.now(tz=timezone.utc)
            )
        except ValueError:
            timestamp = datetime.now(tz=timezone.utc)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=event.get("chatName") or chat_id,
            chat_type=chat_type,
            user_id=event.get("senderId") or chat_id,
            user_name=event.get("senderName"),
        )
        message_event = MessageEvent(
            text=text,
            message_type=mtype,
            source=source,
            message_id=event.get("messageId"),
            raw_message=event,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(message_event)

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> List[re.Pattern]:
        if raw is None:
            patterns = list(_DEFAULT_MENTION_PATTERNS)
        elif isinstance(raw, str):
            text = raw.strip()
            try:
                loaded = json.loads(text) if text else []
            except Exception:
                loaded = None
            patterns = (
                loaded
                if isinstance(loaded, list)
                else [
                    part.strip()
                    for line in text.splitlines()
                    for part in line.split(",")
                ]
            )
        elif isinstance(raw, list):
            patterns = raw
        else:
            patterns = [raw]
        compiled: List[re.Pattern] = []
        for pattern in patterns:
            text = str(pattern).strip()
            if not text:
                continue
            try:
                compiled.append(re.compile(text, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[wechaty] invalid mention pattern %r: %s", text, exc)
        return compiled

    def _message_matches_mention_patterns(self, text: str) -> bool:
        patterns = self._compile_mention_patterns(
            (self.config.extra or {}).get("mention_patterns")
            or os.getenv("WECHATY_MENTION_PATTERNS")
        )
        if not text or not patterns:
            return False
        return any(p.search(text) for p in patterns)

    def _clean_mention_text(self, text: str) -> str:
        patterns = self._compile_mention_patterns(
            (self.config.extra or {}).get("mention_patterns")
            or os.getenv("WECHATY_MENTION_PATTERNS")
        )
        if not text:
            return text
        for pattern in patterns:
            match = pattern.match(text.lstrip())
            if match:
                cleaned = text.lstrip()[match.end() :].lstrip(" ,:-")
                return cleaned or text
        return text

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        plain = strip_markdown(self.format_message(content))
        try:
            data = await self._sidecar_call(
                "/send",
                {"chatId": chat_id, "text": plain[: self.MAX_MESSAGE_LENGTH]},
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        body: Dict[str, Any] = {"chatId": chat_id, "path": image_path}
        if caption:
            body["caption"] = strip_markdown(caption)
        try:
            data = await self._sidecar_call("/send-file", body)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        body: Dict[str, Any] = {
            "chatId": chat_id,
            "path": file_path,
            "name": file_name,
        }
        if caption:
            body["caption"] = strip_markdown(caption)
        try:
            data = await self._sidecar_call("/send-file", body)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def send_typing(self, chat_id: str) -> None:
        try:
            await self._sidecar_call("/typing", {"chatId": chat_id, "state": "start"})
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> dict:
        if chat_id.startswith("room:"):
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    def format_message(self, content: str) -> str:
        return strip_markdown(content)

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}{path}"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Wechaty sidecar {path} returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        data = resp.json() or {}
        if not data.get("ok"):
            raise RuntimeError(
                f"Wechaty sidecar {path} error: {data.get('error')}"
            )
        return data

    # -- Sidecar lifecycle (pattern from photon adapter) -----------------------

    @staticmethod
    def _find_listener_pids(port: int) -> List[int]:
        try:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [int(tok) for tok in out.stdout.split() if tok.strip().isdigit()]

    @staticmethod
    def _pid_is_sidecar(pid: int) -> bool:
        try:
            out = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return "wechaty/sidecar/index.mjs" in out.stdout

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    async def _reap_stale_sidecar(self) -> None:
        if sys.platform == "win32":
            return
        pids = self._find_listener_pids(self._sidecar_port)
        stale = [pid for pid in pids if self._pid_is_sidecar(pid)]
        for pid in stale:
            logger.warning(
                "[wechaty] reaping orphaned sidecar pid %d on port %d",
                pid,
                self._sidecar_port,
            )
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.time() + 3.0
        while time.time() < deadline and any(self._pid_alive(p) for p in stale):
            await asyncio.sleep(0.1)

    async def _start_sidecar(self) -> None:
        if not (_SIDECAR_DIR / "node_modules").exists():
            raise RuntimeError(
                f"Wechaty sidecar deps missing. Run: cd {_SIDECAR_DIR} && npm install"
            )
        await self._reap_stale_sidecar()

        extra = self.config.extra or {}
        env = os.environ.copy()
        for key, env_name in (
            ("puppet", "WECHATY_PUPPET"),
            ("puppet_token", "WECHATY_PUPPET_SERVICE_TOKEN"),
            ("bot_name", "WECHATY_BOT_NAME"),
        ):
            val = extra.get(key)
            if val and not env.get(env_name):
                env[env_name] = str(val)
        token = (
            env.get("WECHATY_PUPPET_SERVICE_TOKEN")
            or env.get("WECHATY_TOKEN")
            or ""
        ).strip()
        if token:
            env["WECHATY_PUPPET_SERVICE_TOKEN"] = token
        env["WECHATY_SIDECAR_PORT"] = str(self._sidecar_port)
        env["WECHATY_SIDECAR_BIND"] = self._sidecar_bind
        env["WECHATY_SIDECAR_TOKEN"] = self._sidecar_token
        env["WECHATY_SIDECAR_WATCH_STDIN"] = "1"

        self._sidecar_proc = subprocess.Popen(
            [self._node_bin, str(_SIDECAR_DIR / "index.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(sys.platform != "win32"),
        )
        loop = asyncio.get_event_loop()
        self._sidecar_supervisor_task = loop.create_task(
            self._supervise_sidecar(self._sidecar_proc)
        )

        deadline = time.time() + 60.0
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.time() < deadline:
                if self._sidecar_proc.poll() is not None:
                    raise RuntimeError(
                        f"Wechaty sidecar exited with code "
                        f"{self._sidecar_proc.returncode}"
                    )
                try:
                    resp = await client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/healthz",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                    )
                    if resp.status_code == 200:
                        return
                except httpx.RequestError as exc:
                    last_err = exc
                await asyncio.sleep(0.3)
        raise RuntimeError(
            f"Wechaty sidecar not ready within 60s (Wechaty login may still be "
            f"pending): {last_err}"
        )

    async def _supervise_sidecar(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        stdout = proc.stdout
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, stdout.readline)
                if not line:
                    break
                logger.info(
                    "[wechaty-sidecar] %s",
                    line.decode("utf-8", "replace").rstrip(),
                )
        except Exception as exc:
            logger.warning("[wechaty-sidecar] supervisor exited: %s", exc)
        if self._inbound_running:
            code = proc.poll()
            self._set_fatal_error(
                "SIDECAR_CRASHED",
                f"Wechaty sidecar exited unexpectedly (code {code})",
                retryable=True,
            )
            try:
                await self._notify_fatal_error()
            except Exception:
                pass

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        self._sidecar_proc = None
        if self._sidecar_supervisor_task is not None:
            task = self._sidecar_supervisor_task
            self._sidecar_supervisor_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if proc is None:
            return
        try:
            await self._sidecar_call("/shutdown", {})
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _cache_inbound_image(attachment: Dict[str, Any]) -> Optional[str]:
    data_b64 = attachment.get("data")
    if not data_b64 or attachment.get("encoding") != "base64":
        return None
    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        return None
    cache_dir = get_hermes_home() / "wechaty" / "media-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg"
    mime = (attachment.get("mimeType") or "").lower()
    if "png" in mime:
        ext = ".png"
    elif "gif" in mime:
        ext = ".gif"
    path = cache_dir / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(raw)
    return str(path)


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not installed"}
    token = os.getenv("WECHATY_SIDECAR_TOKEN")
    if not token:
        return {
            "error": (
                "Wechaty standalone send needs a running gateway sidecar with "
                "WECHATY_SIDECAR_TOKEN in the environment."
            )
        }
    port = _coerce_port(
        (pconfig.extra or {}).get("sidecar_port")
        or os.getenv("WECHATY_SIDECAR_PORT"),
        _DEFAULT_SIDECAR_PORT,
    )
    base = f"http://{_DEFAULT_SIDECAR_BIND}:{port}"
    headers = {"X-Hermes-Sidecar-Token": token}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if message:
                resp = await client.post(
                    f"{base}/send",
                    json={"chatId": chat_id, "text": message[:_MAX_MESSAGE_LENGTH]},
                    headers=headers,
                )
                if resp.status_code != 200:
                    return {"error": f"send failed: HTTP {resp.status_code}"}
                body = resp.json() or {}
                if not body.get("ok"):
                    return {"error": body.get("error") or "send failed"}
            for item in media_files or []:
                path = item if isinstance(item, str) else item.get("path")
                if not path:
                    continue
                resp = await client.post(
                    f"{base}/send-file",
                    json={"chatId": chat_id, "path": path},
                    headers=headers,
                )
                if resp.status_code != 200:
                    return {"error": f"send-file failed: HTTP {resp.status_code}"}
        return {"success": True}
    except Exception as exc:
        return {"error": f"Wechaty standalone send failed: {exc}"}


def interactive_setup() -> None:
    """Minimal setup wizard for Wechaty (env vars + sidecar deps)."""
    from hermes_cli.config import save_env_value
    from hermes_cli.gateway import Colors, color, print_info, print_success

    print()
    print(color("  ─── 💬 WeChat (Wechaty) Setup ───", Colors.CYAN))
    print_info(
        "  Sidecar path (run npm install here if needed):\n"
        f"    {_SIDECAR_DIR}"
    )
    if not (_SIDECAR_DIR / "node_modules").exists():
        print_info("  Installing sidecar npm dependencies…")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=_SIDECAR_DIR,
                check=True,
                timeout=300,
            )
            print_success("  Sidecar dependencies installed.")
        except Exception as exc:
            print_info(f"  npm install failed: {exc}")
            print_info(f"  Run manually: cd {_SIDECAR_DIR} && npm install")
    else:
        print_success("  Sidecar dependencies already installed.")

    puppet = (
        os.getenv("WECHATY_PUPPET")
        or input("  Wechaty puppet [wechaty-puppet-wechat4u]: ").strip()
        or "wechaty-puppet-wechat4u"
    )
    save_env_value("WECHATY_PUPPET", puppet)

    if puppet == "wechaty-puppet-service":
        token = os.getenv("WECHATY_PUPPET_SERVICE_TOKEN") or input(
            "  Puppet service token: "
        ).strip()
        if token:
            save_env_value("WECHATY_PUPPET_SERVICE_TOKEN", token)

    print_info(
        "  Enable in config.yaml:\n"
        "    gateway:\n"
        "      platforms:\n"
        "        wechaty:\n"
        "          enabled: true"
    )
    print_info("  Then: hermes gateway run  (scan QR in logs)")


def register(ctx) -> None:
    ctx.register_platform(
        name="wechaty",
        label="WeChat (Wechaty)",
        adapter_factory=lambda cfg: WechatyAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        install_hint=(
            f"Install sidecar deps: cd {_SIDECAR_DIR} && npm install. "
            "Set WECHATY_PUPPET in ~/.hermes/.env "
            "(or use wechaty-puppet-wechat4u for a free trial puppet)."
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="WECHATY_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="WECHATY_ALLOWED_USERS",
        allow_all_env="WECHATY_ALLOW_ALL_USERS",
        max_message_length=_MAX_MESSAGE_LENGTH,
        emoji="💬",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via personal WeChat (Wechaty). Keep replies "
            "concise and conversational. In groups you were invoked via @-mention. "
            "WeChat does not render rich markdown — prefer plain text and short lists."
        ),
    )
