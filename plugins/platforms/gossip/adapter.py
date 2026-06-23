"""Gossip platform adapter.

The Gossip SDK is TypeScript/Node-first, so Hermes talks to it through a small
loopback sidecar. This adapter keeps the Hermes-facing platform contract the
same as Telegram/Discord while the sidecar owns SDK lifecycle, polling, and
encrypted text message send/receive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx is a Hermes dependency
    HTTPX_AVAILABLE = False
    httpx = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_SIDECAR_PORT = 8797
_DEFAULT_SIDECAR_BIND = "127.0.0.1"
_MAX_MESSAGE_LENGTH = 8000
_SIDECAR_DIR = Path(__file__).parent / "sidecar"
_GOSSIP_SDK_DIR = _SIDECAR_DIR / "node_modules" / "@massalabs" / "gossip-sdk"
_GOSSIP_SDK_DIST = _GOSSIP_SDK_DIR / "dist" / "index.js"


def _terminate_process_tree(proc: subprocess.Popen, *, force: bool = False) -> None:
    """Terminate the sidecar and any child processes it created."""
    if sys.platform == "win32":
        cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            cmd.append("/F")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            if force:
                proc.kill()
            else:
                proc.terminate()
            return

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {proc.pid}")
        return

    import psutil

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        if force:
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        else:
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
    except psutil.NoSuchProcess:
        return


def _coerce_port(value: Any, default: int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return default
    return port if 0 < port < 65536 else default


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_storage_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "gossip"


def _gossip_sdk_problem() -> Optional[str]:
    if not _GOSSIP_SDK_DIR.exists():
        return (
            "Gossip sidecar dependencies are not installed. Run:\n"
            f"  cd {_SIDECAR_DIR} && npm install\n"
            "Then run setup again."
        )
    if not _GOSSIP_SDK_DIST.exists():
        return (
            "@massalabs/gossip-sdk is installed, but the npm package is missing "
            "dist/index.js. The published package points its exports at dist/index.js, "
            "so Node cannot import it. Install a fixed npm release of "
            "@massalabs/gossip-sdk that includes its built dist files, then run setup "
            "again."
        )
    return None


def check_requirements() -> bool:
    if not HTTPX_AVAILABLE:
        return False
    if not shutil.which(os.getenv("GOSSIP_NODE_BIN") or "node"):
        return False
    if _gossip_sdk_problem():
        return False
    return True


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    mnemonic = os.getenv("GOSSIP_MNEMONIC") or extra.get("mnemonic")
    admin = os.getenv("GOSSIP_ADMIN_USER_ID") or extra.get("admin_user_id")
    return bool(str(mnemonic or "").strip() and str(admin or "").strip())


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    mnemonic = os.getenv("GOSSIP_MNEMONIC", "").strip()
    admin = os.getenv("GOSSIP_ADMIN_USER_ID", "").strip()
    if not (mnemonic and admin):
        return None
    seed: dict[str, Any] = {
        "mnemonic": mnemonic,
        "admin_user_id": admin,
    }
    bot_user_id = os.getenv("GOSSIP_BOT_USER_ID", "").strip()
    if bot_user_id:
        seed["bot_user_id"] = bot_user_id
    api_url = os.getenv("GOSSIP_API_URL", "").strip()
    if api_url:
        seed["api_url"] = api_url
    seed["home_channel"] = {
        "chat_id": os.getenv("GOSSIP_HOME_CHANNEL", "").strip() or admin,
        "name": os.getenv("GOSSIP_HOME_CHANNEL_NAME", "Gossip Admin"),
    }
    return seed


def _apply_yaml_config(_yaml_cfg: dict, gossip_cfg: dict) -> dict | None:
    extras: dict[str, Any] = {}
    key_map = {
        "mnemonic": "GOSSIP_MNEMONIC",
        "admin_user_id": "GOSSIP_ADMIN_USER_ID",
        "bot_user_id": "GOSSIP_BOT_USER_ID",
        "api_url": "GOSSIP_API_URL",
        "home_channel": "GOSSIP_HOME_CHANNEL",
        "storage_dir": "GOSSIP_STORAGE_DIR",
        "sidecar_port": "GOSSIP_SIDECAR_PORT",
        "sidecar_autostart": "GOSSIP_SIDECAR_AUTOSTART",
        "poll_interval_ms": "GOSSIP_POLL_INTERVAL_MS",
    }
    for key, env_name in key_map.items():
        if key in gossip_cfg:
            extras[key] = gossip_cfg[key]
            if not os.getenv(env_name):
                os.environ[env_name] = str(gossip_cfg[key])
    return extras or None


def _create_identity_with_sidecar(storage_dir: Path, api_url: str = "") -> dict[str, str]:
    env = os.environ.copy()
    env["GOSSIP_STORAGE_DIR"] = str(storage_dir)
    if api_url:
        env["GOSSIP_API_URL"] = api_url
    node_bin = os.getenv("GOSSIP_NODE_BIN") or shutil.which("node") or "node"
    proc = subprocess.run(  # noqa: S603
        [node_bin, str(_SIDECAR_DIR / "index.mjs"), "--create-identity"],
        cwd=str(_SIDECAR_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    output = (proc.stdout or "").strip().splitlines()
    payload_raw = output[-1] if output else "{}"
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"identity generator returned invalid JSON: {detail}") from exc
    if proc.returncode != 0 or not payload.get("ok"):
        raise RuntimeError(payload.get("error") or (proc.stderr or "identity generation failed"))
    mnemonic = str(payload.get("mnemonic") or "").strip()
    user_id = str(payload.get("userId") or "").strip()
    if not mnemonic or not user_id:
        raise RuntimeError("identity generator did not return mnemonic and userId")
    return {"mnemonic": mnemonic, "user_id": user_id}


def interactive_setup() -> None:
    """Create or configure the Hermes Gossip identity."""
    from hermes_cli.config import get_env_value, save_env_value

    storage_dir = Path(get_env_value("GOSSIP_STORAGE_DIR") or _default_storage_dir())
    api_url = get_env_value("GOSSIP_API_URL") or ""

    mnemonic = get_env_value("GOSSIP_MNEMONIC")
    bot_user_id = get_env_value("GOSSIP_BOT_USER_ID")
    if not mnemonic:
        sdk_problem = _gossip_sdk_problem()
        if sdk_problem:
            print(sdk_problem)
            return
        storage_dir.mkdir(parents=True, exist_ok=True)
        identity = _create_identity_with_sidecar(storage_dir, api_url)
        save_env_value("GOSSIP_MNEMONIC", identity["mnemonic"])
        save_env_value("GOSSIP_BOT_USER_ID", identity["user_id"])
        bot_user_id = identity["user_id"]
        print("Created a new Hermes Gossip identity.")
    else:
        print("Using existing GOSSIP_MNEMONIC from ~/.hermes/.env or environment.")

    if bot_user_id:
        print(f"Hermes Gossip user id: {bot_user_id}")
        print("Add this user id as a contact in the admin Gossip account.")

    admin = get_env_value("GOSSIP_ADMIN_USER_ID")
    if not admin:
        admin = input("Admin Gossip user id allowed to talk to Hermes: ").strip()
        if admin:
            save_env_value("GOSSIP_ADMIN_USER_ID", admin)
            if not get_env_value("GOSSIP_HOME_CHANNEL"):
                save_env_value("GOSSIP_HOME_CHANNEL", admin)
    if not get_env_value("GOSSIP_HOME_CHANNEL_NAME"):
        save_env_value("GOSSIP_HOME_CHANNEL_NAME", "Gossip Admin")
    print("Gossip integration configured.")


class GossipAdapter(BasePlatformAdapter):
    """Hermes adapter for Gossip text DMs."""

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH
    supports_code_blocks = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("gossip"))
        extra = config.extra or {}
        self._mnemonic = os.getenv("GOSSIP_MNEMONIC") or extra.get("mnemonic") or ""
        self._admin_user_id = (
            os.getenv("GOSSIP_ADMIN_USER_ID") or extra.get("admin_user_id") or ""
        ).strip()
        self._api_url = os.getenv("GOSSIP_API_URL") or extra.get("api_url") or ""
        self._storage_dir = Path(
            os.getenv("GOSSIP_STORAGE_DIR")
            or extra.get("storage_dir")
            or _default_storage_dir()
        )
        self._sidecar_port = _coerce_port(
            extra.get("sidecar_port") or os.getenv("GOSSIP_SIDECAR_PORT"),
            _DEFAULT_SIDECAR_PORT,
        )
        self._sidecar_bind = os.getenv("GOSSIP_SIDECAR_BIND", _DEFAULT_SIDECAR_BIND)
        self._sidecar_token = os.getenv("GOSSIP_SIDECAR_TOKEN") or secrets.token_urlsafe(24)
        self._node_bin = os.getenv("GOSSIP_NODE_BIN") or shutil.which("node") or "node"
        self._autostart_sidecar = _coerce_bool(
            extra.get("sidecar_autostart") or os.getenv("GOSSIP_SIDECAR_AUTOSTART"),
            True,
        )
        self._poll_interval_ms = _coerce_int(
            extra.get("poll_interval_ms") or os.getenv("GOSSIP_POLL_INTERVAL_MS"),
            5000,
        )
        self._http_client: Optional[httpx.AsyncClient] = None
        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._sidecar_supervisor_task: Optional[asyncio.Task] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._inbound_running = False

    @property
    def name(self) -> str:
        return "Gossip"

    def format_message(self, content: str) -> str:
        return content

    async def connect(self) -> bool:
        if not HTTPX_AVAILABLE:
            self._set_fatal_error("MISSING_DEP", "httpx not installed", retryable=False)
            return False
        if not self._mnemonic or not self._admin_user_id:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "GOSSIP_MNEMONIC and GOSSIP_ADMIN_USER_ID are required",
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
                    f"failed to start Gossip sidecar: {exc}",
                    retryable=True,
                )
                await self._http_client.aclose()
                self._http_client = None
                return False

        self._inbound_running = True
        self._inbound_task = asyncio.get_event_loop().create_task(self._inbound_loop())
        self._mark_connected()
        logger.info("[gossip] connected as text-only DM adapter")
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
                async with client.stream("GET", url, headers=headers, timeout=None) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"/inbound returned {resp.status_code}")
                    backoff = 1.0
                    async for line in resp.aiter_lines():
                        if not self._inbound_running:
                            break
                        line = line.strip()
                        if line:
                            await self._on_inbound_line(line)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._inbound_running:
                    logger.warning("[gossip] inbound stream error: %s", exc)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    async def _on_inbound_line(self, line: str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[gossip] ignoring invalid sidecar line")
            return
        kind = str(data.get("kind") or "message")
        contact_user_id = str(data.get("contactUserId") or "").strip()
        if contact_user_id != self._admin_user_id:
            logger.info("[gossip] dropped non-admin inbound event from %s", contact_user_id)
            return

        text = str(data.get("text") or "").strip()
        if not text:
            return
        reply_to_text = str(data.get("replyToText") or "").strip()
        if reply_to_text:
            text = f"[Replying to: {reply_to_text}]\n{text}"
        if kind == "message_deleted":
            text = f"[The admin deleted a Gossip message. Deleted content was: {text}]"
        elif kind == "message_updated":
            text = f"[The admin edited a Gossip message. Current content: {text}]"

        message_id = str(data.get("messageId") or data.get("dbId") or "")
        source = self.build_source(
            chat_id=contact_user_id,
            chat_name="Gossip Admin",
            chat_type="dm",
            user_id=contact_user_id,
            user_name="Gossip Admin",
            message_id=message_id or None,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
            message_id=message_id or None,
        )
        await self.handle_message(event)

    async def _start_sidecar(self) -> None:
        await self._reap_stale_sidecar()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "GOSSIP_MNEMONIC": self._mnemonic,
                "GOSSIP_ADMIN_USER_ID": self._admin_user_id,
                "GOSSIP_STORAGE_DIR": str(self._storage_dir),
                "GOSSIP_SIDECAR_PORT": str(self._sidecar_port),
                "GOSSIP_SIDECAR_BIND": self._sidecar_bind,
                "GOSSIP_SIDECAR_TOKEN": self._sidecar_token,
                "GOSSIP_POLL_INTERVAL_MS": str(self._poll_interval_ms),
                "GOSSIP_SIDECAR_WATCH_STDIN": "1",
            }
        )
        if self._api_url:
            env["GOSSIP_API_URL"] = self._api_url

        self._sidecar_proc = subprocess.Popen(  # noqa: S603
            [self._node_bin, str(_SIDECAR_DIR / "index.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(sys.platform != "win32"),
        )
        self._sidecar_supervisor_task = asyncio.get_event_loop().create_task(
            self._supervise_sidecar(self._sidecar_proc)
        )

        deadline = time.time() + 20.0
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if self._sidecar_proc.poll() is not None:
                    raise RuntimeError(
                        f"Gossip sidecar exited with code {self._sidecar_proc.returncode}"
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
                await asyncio.sleep(0.2)
        raise RuntimeError(f"Gossip sidecar did not become ready: {last_err}")

    async def _supervise_sidecar(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    break
                logger.info("[gossip-sidecar] %s", line.decode("utf-8", "replace").rstrip())
        except Exception as exc:
            logger.warning("[gossip-sidecar] supervisor exited: %s", exc)
        if self._inbound_running:
            self._set_fatal_error(
                "SIDECAR_CRASHED",
                f"Gossip sidecar exited unexpectedly (code {proc.poll()})",
                retryable=True,
            )
            try:
                await self._notify_fatal_error()
            except Exception:
                logger.debug("[gossip] fatal-error notification failed", exc_info=True)

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if self._http_client is not None:
                try:
                    await self._http_client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/shutdown",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                        timeout=2.0,
                    )
                except Exception:
                    pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(proc, force=True)
        finally:
            self._sidecar_proc = None
            if self._sidecar_supervisor_task is not None:
                self._sidecar_supervisor_task.cancel()
                self._sidecar_supervisor_task = None

    async def _reap_stale_sidecar(self) -> None:
        return None

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._http_client is None:
            raise RuntimeError("Gossip sidecar HTTP client is not connected")
        resp = await self._http_client.post(
            f"http://{self._sidecar_bind}:{self._sidecar_port}{path}",
            json=body,
            headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"sidecar returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json() or {}
        if not data.get("ok"):
            raise RuntimeError(str(data.get("error") or "sidecar reported failure"))
        return data

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if chat_id != self._admin_user_id:
            return SendResult(False, error="Gossip adapter only sends to configured admin")
        try:
            data = await self._sidecar_call(
                "/send",
                {
                    "contactUserId": chat_id,
                    "text": content[: self.MAX_MESSAGE_LENGTH],
                    "replyTo": reply_to,
                },
            )
            return SendResult(True, message_id=str(data.get("messageId") or ""))
        except Exception as exc:
            return SendResult(False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"chat_id": chat_id, "name": "Gossip Admin", "type": "dm"}


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
    port = _coerce_port(
        (pconfig.extra or {}).get("sidecar_port") or os.getenv("GOSSIP_SIDECAR_PORT"),
        _DEFAULT_SIDECAR_PORT,
    )
    token = os.getenv("GOSSIP_SIDECAR_TOKEN")
    if not token:
        return {"error": "Gossip standalone send requires a running gateway sidecar"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://{_DEFAULT_SIDECAR_BIND}:{port}/send",
                json={"contactUserId": chat_id, "text": message[:_MAX_MESSAGE_LENGTH]},
                headers={"X-Hermes-Sidecar-Token": token},
            )
            if resp.status_code != 200:
                return {"error": f"sidecar returned {resp.status_code}: {resp.text[:200]}"}
            data = resp.json() or {}
            if not data.get("ok"):
                return {"error": data.get("error") or "sidecar reported failure"}
            return {"success": True, "message_id": data.get("messageId")}
    except Exception as exc:
        return {"error": f"Gossip standalone send failed: {exc}"}


def register(ctx) -> None:
    ctx.register_platform(
        name="gossip",
        label="Gossip",
        adapter_factory=lambda cfg: GossipAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["GOSSIP_MNEMONIC", "GOSSIP_ADMIN_USER_ID"],
        install_hint=(
            "Run npm install in hermes-agent/plugins/platforms/gossip/sidecar "
            "to install @massalabs/gossip-sdk from npm."
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="GOSSIP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="GOSSIP_ADMIN_USER_ID",
        allow_all_env="",
        max_message_length=_MAX_MESSAGE_LENGTH,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are communicating over Gossip in a text-only direct message "
            "with the configured admin. Gossip does not support file sharing, "
            "topics, or group chats in this integration yet. Treat edit/delete "
            "notices as context updates from the admin."
        ),
    )
