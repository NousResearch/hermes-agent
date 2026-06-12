"""Avocado fleet controller — self-serve tenant provisioning for Hermes.

AVOCADO FORK SERVICE (not upstream). Runs inside the Hermes Railway
container as a supervised s6 longrun service (docker/s6-rc.d/fleet-controller)
and automates what `provision-in-container.sh` did by hand: the Avocado app
calls this HTTP API the moment a customer pastes their Telegram bot token,
and a working, paired, isolated agent comes up without any operator action.

Architecture decision — pairing WITHOUT forking the Telegram adapter:
the Hermes profile (and its gateway) is only created at PAIRING SUCCESS.
While a tenant is unpaired, this controller runs a lightweight Telegram
getUpdates long-poller on the bot token (a few raw Bot-API calls, no
Hermes) that ONLY understands ``/start <pairing-code>``. On a valid code
(verified by calling back to the Avocado app) the poller is stopped, the
profile is written to the Volume with the allowlist locked to the paired
Telegram user, and the real Hermes gateway takes over polling. This keeps
the gateway codebase untouched and guarantees an unpaired bot can never
reach the agent, the model, or any MCP.

Inbound API (auth: ``Authorization: Bearer $FLEET_PROVISION_SECRET``):

    POST /provision    {tenantId, botToken, botUsername?, avocadoMcpKey,
                        pairingCode}            -> 200 {"ok": true}
    POST /deprovision  {tenantId}               -> 200 {"ok": true}
    GET  /health       (no auth)                -> 200 {"ok": true}

Outbound callback (pairing, auth: same bearer secret):

    POST $AVOCADO_APP_URL/api/super-agent/channels/telegram/pair
         {tenantId, pairingCode, telegramUserId}
    200 -> lock + welcome; 403 -> wrong code; other -> transient, retryable.

Environment:
    FLEET_PROVISION_SECRET  shared bearer secret (service refuses to start
                            without it — the s6 run script guards this too)
    AVOCADO_APP_URL         e.g. https://avocadoai.co (no trailing slash)
    FLEET_CONTROLLER_PORT   default 8800
    HERMES_HOME             default /opt/data (the Railway Volume)

State: $HERMES_HOME/fleet/registry.json (chmod 600, atomic writes). Bot
tokens necessarily live there for unpaired tenants (the poller needs them
across controller restarts); paired tenants' tokens live in the profile
.env exactly like concierge-provisioned ones. Secrets are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s fleet: %(message)s",
)
log = logging.getLogger("fleet-controller")

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/opt/data"))
HERMES_BIN = "/opt/hermes/.venv/bin/hermes"
FLEET_DIR = HERMES_HOME / "fleet"
REGISTRY_PATH = FLEET_DIR / "registry.json"
TELEGRAM_API = "https://api.telegram.org"

SECRET = os.environ.get("FLEET_PROVISION_SECRET", "")
APP_URL = os.environ.get("AVOCADO_APP_URL", "").rstrip("/")
PORT = int(os.environ.get("FLEET_CONTROLLER_PORT", "8800"))

DEFAULT_MODEL = os.environ.get("FLEET_DEFAULT_MODEL", "xiaomi/mimo-v2.5-pro")
MAX_ITER = int(os.environ.get("FLEET_MAX_ITER", "40"))

PAIR_ENDPOINT = "/api/super-agent/channels/telegram/pair"
START_RE = re.compile(r"^/start(?:@\w+)?\s+(\S+)\s*$")

MSG_PRIVATE = "This agent is private. Connect it from your Avocado account."
MSG_WRONG_CODE = (
    "That code doesn't match. Open your Avocado account, copy the pairing "
    "code shown there, and send: /start <code>"
)
MSG_TRANSIENT = "Something went wrong on our side — please try that again in a minute."
MSG_WELCOME = "You're connected — ask me to create something!"

# Mirrors hermes_cli.service_manager.validate_profile_name.
_VALID_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def tenant_slug(tenant_id: str) -> str:
    """Deterministic, collision-safe Hermes profile name for a tenant.

    Clerk userIds are case-sensitive ("user_2Abc…"), profile names must be
    lowercase — so a lossy lowercase sanitize alone could collide. Append a
    short hash of the exact tenantId to keep the mapping injective.
    """
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:8]
    base = re.sub(r"[^a-z0-9_-]+", "-", tenant_id.lower()).strip("-_") or "tenant"
    slug = f"t-{base[:24]}-{digest}"
    if not _VALID_PROFILE_RE.match(slug):  # pragma: no cover — belt & braces
        slug = f"t-{digest}"
    return slug


# ---------------------------------------------------------------------------
# Registry (volume-persisted tenant state)
# ---------------------------------------------------------------------------

class Registry:
    """Tiny JSON registry with atomic writes. Single-process access only."""

    def __init__(self, path: Path):
        self._path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._data = {}
        except Exception:
            log.exception("registry unreadable — starting empty (file kept)")
            self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self._path)

    def get(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(tenant_id)

    def put(self, tenant_id: str, record: Dict[str, Any]) -> None:
        record["updatedAt"] = int(time.time())
        self._data[tenant_id] = record
        self.save()

    def remove(self, tenant_id: str) -> None:
        if self._data.pop(tenant_id, None) is not None:
            self.save()

    def items(self):
        return list(self._data.items())


# ---------------------------------------------------------------------------
# Hermes profile plumbing (mirrors provision-in-container.sh)
# ---------------------------------------------------------------------------

async def _run_hermes(*args: str, timeout: float = 120) -> tuple[int, str]:
    """Run the hermes CLI inside the container; return (rc, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        HERMES_BIN, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "HOME": str(HERMES_HOME)},
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, f"hermes {' '.join(args[:3])}… timed out"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


def _profile_dir(slug: str) -> Path:
    return HERMES_HOME / "profiles" / slug


def _write_profile_files(
    slug: str, *, bot_token: str, telegram_user_id: str,
    avocado_mcp_key: str,
) -> None:
    """Write config.yaml + .env for a paired tenant profile.

    Mirrors provision-in-container.sh: Avocado MCP scoped to the tenant's
    key, creative-safe toolset only (no terminal/file/code), manual
    approvals, cron denied. No OPENROUTER_API_KEY line — the profile
    inherits the shared fleet key from the Railway service variable.
    """
    pdir = _profile_dir(slug)
    pdir.mkdir(parents=True, exist_ok=True)

    config = f"""model:
  default: {DEFAULT_MODEL}
  provider: openrouter
agent:
  max_turns: {MAX_ITER}
  gateway_timeout: 1800
delegation:
  max_iterations: 30
approvals:
  mode: manual
  timeout: 120
  cron_mode: deny
mcp_servers:
  avocado:
    url: https://www.avocadoai.co/api/mcp
    headers:
      Authorization: "Bearer {avocado_mcp_key}"
    connect_timeout: 60
    timeout: 180
platform_toolsets:
  telegram:
    - image_gen
    - vision
    - tts
    - web
    - memory
    - session_search
    - messaging
    - clarify
    - todo
telegram:
  enabled: true
  reactions: false
cron:
  wrap_response: true
"""
    env = (
        f"TELEGRAM_BOT_TOKEN={bot_token}\n"
        f"TELEGRAM_ALLOWED_USERS={telegram_user_id}\n"
        f"HERMES_MAX_ITERATIONS={MAX_ITER}\n"
        "AUTO_UPDATE=false\n"
    )
    (pdir / "config.yaml").write_text(config, encoding="utf-8")
    env_path = pdir / ".env"
    env_path.write_text(env, encoding="utf-8")
    os.chmod(env_path, 0o600)


# ---------------------------------------------------------------------------
# Telegram Bot API helpers (raw, for the unpaired poller only)
# ---------------------------------------------------------------------------

async def _tg(session: aiohttp.ClientSession, token: str, method: str,
              **params: Any) -> Dict[str, Any]:
    async with session.post(
        f"{TELEGRAM_API}/bot{token}/{method}", json=params,
        timeout=aiohttp.ClientTimeout(total=70),
    ) as resp:
        return await resp.json(content_type=None)


async def _tg_say(session: aiohttp.ClientSession, token: str,
                  chat_id: Any, text: str) -> None:
    try:
        await _tg(session, token, "sendMessage", chat_id=chat_id, text=text)
    except Exception:
        log.warning("sendMessage failed (chat %s)", chat_id)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class FleetController:
    def __init__(self) -> None:
        self.registry = Registry(REGISTRY_PATH)
        self._pollers: Dict[str, asyncio.Task] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._http: Optional[aiohttp.ClientSession] = None

    def _lock(self, tenant_id: str) -> asyncio.Lock:
        return self._locks.setdefault(tenant_id, asyncio.Lock())

    async def start(self) -> None:
        self._http = aiohttp.ClientSession()
        FLEET_DIR.mkdir(parents=True, exist_ok=True)
        resumed = 0
        for tenant_id, rec in self.registry.items():
            if rec.get("status") == "unpaired":
                self._start_poller(tenant_id)
                resumed += 1
        log.info("fleet controller up on :%s (%d unpaired poller(s) resumed)",
                 PORT, resumed)

    # -- pairing poller ----------------------------------------------------

    def _start_poller(self, tenant_id: str) -> None:
        self._stop_poller(tenant_id)
        task = asyncio.create_task(
            self._poll_unpaired(tenant_id), name=f"poller-{tenant_id}",
        )
        self._pollers[tenant_id] = task

        def _log_poller_exit(t: asyncio.Task, _tid: str = tenant_id) -> None:
            # Surface unexpected poller deaths LOUDLY. Without this, an
            # exception inside the poller task is swallowed silently (the
            # task is never awaited) — exactly how the 2026-06-12 pairing
            # incident hid: the chain died and nothing was logged.
            if t.cancelled():
                log.info("poller for %s cancelled", _tid)
                return
            exc = t.exception()
            if exc is not None:
                log.error("poller for %s DIED: %r", _tid, exc, exc_info=exc)
            else:
                log.info("poller for %s exited cleanly", _tid)

        task.add_done_callback(_log_poller_exit)

    def _stop_poller(self, tenant_id: str) -> None:
        task = self._pollers.pop(tenant_id, None)
        # NEVER cancel the task we're running inside of. _finalize_pairing
        # runs IN the poller task and calls this to retire itself — the
        # original code cancelled the current task here, which raised
        # CancelledError at the next await and silently killed the whole
        # post-pairing chain (profile create / gateway start / welcome).
        # The poller returns naturally after pairing; popping the registry
        # entry is all that's needed in the self-call case.
        if (
            task is not None
            and not task.done()
            and task is not asyncio.current_task()
        ):
            task.cancel()

    async def _poll_unpaired(self, tenant_id: str) -> None:
        """Long-poll getUpdates for an unpaired tenant's bot.

        Understands exactly one command: ``/start <code>``. Everything else
        gets the privacy notice (rate-limited per chat).
        """
        rec = self.registry.get(tenant_id)
        if rec is None:
            return
        token = rec["botToken"]
        slug = rec["slug"]
        offset = 0
        last_notice: Dict[Any, float] = {}
        assert self._http is not None
        try:
            await _tg(self._http, token, "deleteWebhook")
        except Exception:
            pass
        log.info("pairing poller up for %s", slug)
        while True:
            try:
                resp = await _tg(
                    self._http, token, "getUpdates",
                    timeout=50, offset=offset, allowed_updates=["message"],
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(5)
                continue
            if not resp.get("ok"):
                # 409 = another consumer holds getUpdates (stale gateway?).
                await asyncio.sleep(10)
                continue
            for update in resp.get("result", []):
                offset = max(offset, update.get("update_id", 0) + 1)
                msg = update.get("message") or {}
                chat_id = (msg.get("chat") or {}).get("id")
                from_id = (msg.get("from") or {}).get("id")
                text = (msg.get("text") or "").strip()
                if chat_id is None or from_id is None:
                    continue
                m = START_RE.match(text)
                if not m:
                    now = time.monotonic()
                    if now - last_notice.get(chat_id, 0) > 5:
                        last_notice[chat_id] = now
                        await _tg_say(self._http, token, chat_id, MSG_PRIVATE)
                    continue
                code = m.group(1)
                log.info("/start received for %s — verifying code with app", slug)
                paired = await self._attempt_pairing(
                    tenant_id, code, str(from_id), chat_id, offset,
                )
                if paired:
                    log.info("pairing complete for %s — poller retiring", slug)
                    return  # poller's job is done; gateway owns the bot now

    async def _attempt_pairing(self, tenant_id: str, code: str,
                               telegram_user_id: str, chat_id: Any,
                               last_update_offset: int = 0) -> bool:
        """Verify the code with the Avocado app; finalize on 200."""
        rec = self.registry.get(tenant_id)
        if rec is None:
            return False
        token = rec["botToken"]
        assert self._http is not None
        try:
            async with self._http.post(
                f"{APP_URL}{PAIR_ENDPOINT}",
                json={
                    "tenantId": tenant_id,
                    "pairingCode": code,
                    "telegramUserId": telegram_user_id,
                },
                headers={"Authorization": f"Bearer {SECRET}"},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                status = resp.status
        except Exception:
            log.warning("pair callback unreachable for %s", rec["slug"])
            await _tg_say(self._http, token, chat_id, MSG_TRANSIENT)
            return False

        if status == 403:
            log.info("pair callback rejected (403 wrong code) for %s", rec["slug"])
            await _tg_say(self._http, token, chat_id, MSG_WRONG_CODE)
            return False
        if status != 200:
            log.warning("pair callback HTTP %s for %s", status, rec["slug"])
            await _tg_say(self._http, token, chat_id, MSG_TRANSIENT)
            return False

        log.info("pair callback OK (200) for %s — finalizing", rec["slug"])
        try:
            async with self._lock(tenant_id):
                ok = await self._finalize_pairing(
                    tenant_id, telegram_user_id, last_update_offset,
                )
        except Exception:
            log.exception("finalize_pairing CRASHED for %s", rec["slug"])
            ok = False
        if ok:
            await _tg_say(self._http, token, chat_id, MSG_WELCOME)
        else:
            await _tg_say(self._http, token, chat_id, MSG_TRANSIENT)
        return ok

    async def _finalize_pairing(self, tenant_id: str,
                                telegram_user_id: str,
                                last_update_offset: int = 0) -> bool:
        """Create + start the real Hermes profile, locked to the paired user.

        Every step logs before it runs — this chain died silently once
        (self-cancellation via _stop_poller) and we never want to grep in
        the dark for it again.
        """
        rec = self.registry.get(tenant_id)
        if rec is None:
            log.error("finalize: tenant vanished from registry")
            return False
        slug = rec["slug"]
        # Retire our poller entry BEFORE the gateway starts: two getUpdates
        # consumers on one token fight each other with 409s. (_stop_poller
        # is self-call-safe: it never cancels the current task.)
        log.info("finalize[%s] 1/6: retiring pairing poller", slug)
        self._stop_poller(tenant_id)

        # Ack the consumed /start update with Telegram so the real gateway
        # doesn't re-receive it as the first message after takeover.
        if last_update_offset and self._http is not None:
            log.info("finalize[%s] 2/6: acking telegram updates", slug)
            try:
                await _tg(self._http, rec["botToken"], "getUpdates",
                          offset=last_update_offset, timeout=0)
            except Exception:
                log.warning("finalize[%s]: update ack failed (gateway may "
                            "see the /start message once — harmless)", slug)

        log.info("finalize[%s] 3/6: creating hermes profile", slug)
        rc, out = await _run_hermes("profile", "create", slug)
        if rc != 0 and "exist" not in out.lower():
            log.error("finalize[%s] FAILED at profile create (rc=%s): %s",
                      slug, rc, out[-300:])
            return False

        log.info("finalize[%s] 4/6: writing config + env (allowlist locked)", slug)
        _write_profile_files(
            slug,
            bot_token=rec["botToken"],
            telegram_user_id=telegram_user_id,
            avocado_mcp_key=rec["avocadoMcpKey"],
        )

        log.info("finalize[%s] 5/6: starting gateway", slug)
        await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
        rc, out = await _run_hermes("-p", slug, "gateway", "start", timeout=120)
        if rc != 0:
            log.error("finalize[%s] FAILED at gateway start (rc=%s): %s",
                      slug, rc, out[-300:])
            return False

        log.info("finalize[%s] 6/6: flipping registry to paired", slug)
        rec.update(status="paired", telegramUserId=telegram_user_id)
        rec.pop("pairingCode", None)  # single-use
        self.registry.put(tenant_id, rec)
        log.info("tenant paired: %s (telegram user locked)", slug)
        return True

    # -- HTTP API ----------------------------------------------------------

    def _check_auth(self, request: web.Request) -> Optional[web.Response]:
        header = request.headers.get("Authorization", "")
        if not SECRET or header != f"Bearer {SECRET}":
            return web.json_response({"ok": False, "error": "unauthorized"},
                                     status=401)
        return None

    async def handle_provision(self, request: web.Request) -> web.Response:
        if (err := self._check_auth(request)) is not None:
            return err
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"},
                                     status=400)
        tenant_id = str(body.get("tenantId") or "").strip()
        bot_token = str(body.get("botToken") or "").strip()
        avk = str(body.get("avocadoMcpKey") or "").strip()
        code = str(body.get("pairingCode") or "").strip()
        bot_username = (body.get("botUsername") or None)
        if not tenant_id or not bot_token or not avk or not code:
            return web.json_response(
                {"ok": False,
                 "error": "tenantId, botToken, avocadoMcpKey, pairingCode required"},
                status=400)
        if re.search(r"[\r\n\x00]", tenant_id + bot_token + avk + code):
            return web.json_response({"ok": False, "error": "invalid characters"},
                                     status=400)

        slug = tenant_slug(tenant_id)
        async with self._lock(tenant_id):
            # Idempotent refresh: silence any existing gateway/poller first
            # so the new token's poller is the only getUpdates consumer.
            self._stop_poller(tenant_id)
            await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
            self.registry.put(tenant_id, {
                "slug": slug,
                "status": "unpaired",
                "botToken": bot_token,
                "botUsername": bot_username,
                "avocadoMcpKey": avk,
                "pairingCode": code,
            })
            self._start_poller(tenant_id)
        log.info("tenant provisioned (unpaired): %s", slug)
        return web.json_response({"ok": True})

    async def handle_deprovision(self, request: web.Request) -> web.Response:
        if (err := self._check_auth(request)) is not None:
            return err
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"},
                                     status=400)
        tenant_id = str(body.get("tenantId") or "").strip()
        if not tenant_id:
            return web.json_response({"ok": False, "error": "tenantId required"},
                                     status=400)
        slug = tenant_slug(tenant_id)
        async with self._lock(tenant_id):
            self._stop_poller(tenant_id)
            await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
            await _run_hermes("profile", "delete", slug, "--yes", timeout=60)
            # Belt & braces: the CLI delete unregisters the s6 slot; make
            # sure no profile remnants survive on the volume either way.
            shutil.rmtree(_profile_dir(slug), ignore_errors=True)
            self.registry.remove(tenant_id)
        log.info("tenant deprovisioned: %s", slug)
        return web.json_response({"ok": True})

    async def handle_health(self, request: web.Request) -> web.Response:
        counts = {"unpaired": 0, "paired": 0}
        for _, rec in self.registry.items():
            status = rec.get("status", "")
            if status in counts:
                counts[status] += 1
        return web.json_response({"ok": True, **counts})


def main() -> None:
    if not SECRET:
        log.error("FLEET_PROVISION_SECRET is not set — refusing to start")
        sys.exit(1)
    if not APP_URL:
        log.error("AVOCADO_APP_URL is not set — refusing to start")
        sys.exit(1)

    controller = FleetController()
    app = web.Application()
    app.router.add_post("/provision", controller.handle_provision)
    app.router.add_post("/deprovision", controller.handle_deprovision)
    app.router.add_get("/health", controller.handle_health)

    async def _on_startup(_app: web.Application) -> None:
        await controller.start()

    app.on_startup.append(_on_startup)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
