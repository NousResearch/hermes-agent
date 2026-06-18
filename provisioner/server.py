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

    POST /provision        {tenantId, botToken, botUsername?, avocadoMcpKey,
                            pairingCode}            -> 200 {"ok": true}
    POST /provision-slack  {tenantId, avocadoMcpKey, soul?,
                            slack:{botToken, appToken, workspaceName?}}
                                                    -> 200 {"ok": true}
    POST /deprovision      {tenantId, channel?}     -> 200 {"ok": true}
                           (channel="telegram"|"slack" drops one channel and
                            keeps the rest; omit channel for full teardown)
    GET  /health           (no auth)                -> 200 {"ok": true}

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


SOUL_TEMPLATE = """\
# Super Agent

You are **Super Agent** — the creative AI agent for this user's Avocado AI
account, made by Avocado AI (avocadoai.co).

## Identity rules
- Your name is Super Agent. You are a product of Avocado AI.
- Never mention Hermes, Nous Research, or any underlying framework, model
  vendor, or infrastructure. If asked what you are or who made you:
  "I'm Super Agent, your creative AI agent from Avocado AI."

## Personality & style
- Helpful creative partner: warm, direct, encouraging.
- Concise and mobile-chat friendly: short messages. No markdown walls,
  no giant bullet lists unless the user asks for detail.

## How you create
- Create directly in this conversation: generate images, video, music,
  and speech with your avocado tools and send the finished asset back
  here in the chat.
- Do NOT use storyboard or flow tools unless the user explicitly asks
  for a storyboard or flow.

## Photos the user sends
- When the user attaches a photo and wants it edited or animated:
  first call avocado_upload_image with the photo's local path (from the
  "[User sent an image: …]" note), then pass the returned file_id to
  the avocado edit_image or generate_video tool.
- Never pass local file paths or base64 to edit_image / generate_video
  — they only accept a file_id from avocado_upload_image or a public
  https URL the user pasted.

{user_section}
"""

SOUL_KNOWN_USER = """\
## What you know about this user
{soul}
"""

SOUL_FIRST_MEETING = """\
## First meeting
You haven't met this user yet. Over the first few messages, naturally
learn about their business or project, their audience, and the tone and
style they like — one question at a time, woven into the conversation,
never as a form or checklist. Remember what you learn.
"""


# Creative-safe toolset for every channel: no terminal/file/code/computer_use,
# so a connected channel can never reach a shell. Applied per-platform.
SAFE_TOOLSET = [
    "image_gen", "vision", "tts", "web", "memory",
    "session_search", "messaging", "clarify", "todo",
]


def _render_config_yaml(*, avocado_mcp_key: str,
                        tg_ready: bool, sl_ready: bool) -> str:
    """Build config.yaml from whichever channels are READY for this tenant.

    Avocado MCP scoped to the tenant's key, creative-safe toolset per
    platform, manual approvals, cron denied. No OPENROUTER_API_KEY — the
    profile inherits the shared fleet key from the Railway service variable.
    Telegram and Slack each get their own ``platform_toolsets`` entry +
    enable block ONLY when ready, so rendering one channel never disables
    the other.
    """
    toolsets = ""
    if tg_ready:
        toolsets += "  telegram:\n" + "".join(f"    - {t}\n" for t in SAFE_TOOLSET)
    if sl_ready:
        toolsets += "  slack:\n" + "".join(f"    - {t}\n" for t in SAFE_TOOLSET)

    blocks = ""
    if tg_ready:
        blocks += "telegram:\n  enabled: true\n  reactions: false\n"
    if sl_ready:
        # Slack also auto-enables from SLACK_BOT_TOKEN in .env (gateway
        # config.py), but an explicit block keeps intent obvious.
        blocks += "slack:\n  enabled: true\n"

    return f"""model:
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
{toolsets}{blocks}cron:
  wrap_response: true
"""


def _render_env(*, rec: Dict[str, Any], tg_ready: bool, sl_ready: bool) -> str:
    """Build .env from the ready channels. Telegram locks the allowlist to
    the paired user; Slack is whole-workspace (no allowlist). Branding is
    white-labelled to "Super Agent" via HERMES_BRAND_NAME.
    """
    lines: list[str] = []
    if tg_ready:
        tuid = rec.get("telegramUserId", "")
        lines += [
            f"TELEGRAM_BOT_TOKEN={rec['botToken']}",
            f"TELEGRAM_ALLOWED_USERS={tuid}",
            f"TELEGRAM_HOME_CHANNEL={tuid}",
        ]
    if sl_ready:
        sl = rec["slack"]
        lines += [
            f"SLACK_BOT_TOKEN={sl['botToken']}",
            f"SLACK_APP_TOKEN={sl['appToken']}",
        ]
    lines += [
        "HERMES_BRAND_NAME=Super Agent",
        f"HERMES_MAX_ITERATIONS={MAX_ITER}",
        "AUTO_UPDATE=false",
    ]
    return "\n".join(lines) + "\n"


def _write_profile_files(slug: str, rec: Dict[str, Any], *,
                         tg_ready: bool, sl_ready: bool) -> None:
    """Write config.yaml + .env + SOUL.md for a tenant from its full channel
    set. SOUL.md (identity slot #1) presents the agent as "Super Agent" by
    Avocado AI, never Hermes/Nous. ``rec['soul']`` is the user's saved
    profile; when absent the agent interviews the user over first messages.
    """
    pdir = _profile_dir(slug)
    pdir.mkdir(parents=True, exist_ok=True)

    soul_clean = (rec.get("soul") or "").strip()[:8000]
    user_section = (
        SOUL_KNOWN_USER.format(soul=soul_clean)
        if soul_clean else SOUL_FIRST_MEETING
    )
    (pdir / "SOUL.md").write_text(
        SOUL_TEMPLATE.format(user_section=user_section), encoding="utf-8",
    )
    (pdir / "config.yaml").write_text(
        _render_config_yaml(
            avocado_mcp_key=rec["avocadoMcpKey"],
            tg_ready=tg_ready, sl_ready=sl_ready,
        ),
        encoding="utf-8",
    )
    env_path = pdir / ".env"
    env_path.write_text(
        _render_env(rec=rec, tg_ready=tg_ready, sl_ready=sl_ready),
        encoding="utf-8",
    )
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

    async def _render_profile(self, tenant_id: str) -> bool:
        """Rebuild a tenant's profile (config + env + gateway) from the FULL
        set of ready channels in the registry. The single source of truth for
        what a tenant's gateway runs — telegram pairing, slack connect, and
        per-channel disconnect all funnel through here, so adding/removing one
        channel never clobbers another.

        Readiness: telegram = paired (status=='paired' + token); slack =
        both tokens present. If NO channel is ready, the profile is torn down
        entirely (the caller decides whether to drop the registry entry).
        """
        rec = self.registry.get(tenant_id)
        if rec is None:
            return False
        slug = rec["slug"]
        tg_ready = rec.get("status") == "paired" and bool(rec.get("botToken"))
        sl = rec.get("slack")
        sl_ready = bool(sl and sl.get("botToken") and sl.get("appToken"))

        if not tg_ready and not sl_ready:
            log.info("render[%s]: no active channels — tearing profile down", slug)
            await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
            await _run_hermes("profile", "delete", slug, "--yes", timeout=60)
            shutil.rmtree(_profile_dir(slug), ignore_errors=True)
            return True

        log.info("render[%s] 1/3: ensuring profile (telegram=%s slack=%s)",
                 slug, tg_ready, sl_ready)
        rc, out = await _run_hermes("profile", "create", slug)
        if rc != 0 and "exist" not in out.lower():
            log.error("render[%s] FAILED at profile create (rc=%s): %s",
                      slug, rc, out[-300:])
            return False

        log.info("render[%s] 2/3: writing config + env + soul", slug)
        _write_profile_files(slug, rec, tg_ready=tg_ready, sl_ready=sl_ready)

        log.info("render[%s] 3/3: restarting gateway", slug)
        await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
        rc, out = await _run_hermes("-p", slug, "gateway", "start", timeout=120)
        if rc != 0:
            log.error("render[%s] FAILED at gateway start (rc=%s): %s",
                      slug, rc, out[-300:])
            return False
        log.info("render[%s]: gateway up", slug)
        return True

    async def _finalize_pairing(self, tenant_id: str,
                                telegram_user_id: str,
                                last_update_offset: int = 0) -> bool:
        """Lock the paired Telegram user into the registry, then (re)render the
        profile. Rendering goes through _render_profile, so any Slack channel
        already configured for this tenant comes up alongside Telegram.

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
        log.info("finalize[%s] 1/4: retiring pairing poller", slug)
        self._stop_poller(tenant_id)

        # Ack the consumed /start update with Telegram so the real gateway
        # doesn't re-receive it as the first message after takeover.
        if last_update_offset and self._http is not None:
            log.info("finalize[%s] 2/4: acking telegram updates", slug)
            try:
                await _tg(self._http, rec["botToken"], "getUpdates",
                          offset=last_update_offset, timeout=0)
            except Exception:
                log.warning("finalize[%s]: update ack failed (gateway may "
                            "see the /start message once — harmless)", slug)

        log.info("finalize[%s] 3/4: marking paired (allowlist locked)", slug)
        rec.update(status="paired", telegramUserId=telegram_user_id)
        rec.pop("pairingCode", None)  # single-use
        self.registry.put(tenant_id, rec)

        log.info("finalize[%s] 4/4: rendering profile", slug)
        if not await self._render_profile(tenant_id):
            log.error("finalize[%s] FAILED at render", slug)
            return False
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
        # Known fields only — anything else in the payload is ignored for
        # forward-compat (the app may add fields before the fleet does).
        tenant_id = str(body.get("tenantId") or "").strip()
        bot_token = str(body.get("botToken") or "").strip()
        avk = str(body.get("avocadoMcpKey") or "").strip()
        code = str(body.get("pairingCode") or "").strip()
        bot_username = (body.get("botUsername") or None)
        soul_raw = body.get("soul")
        soul = str(soul_raw).strip() if isinstance(soul_raw, str) else None
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
            # Idempotent refresh: silence any existing telegram poller first
            # so the new token's poller is the only getUpdates consumer.
            self._stop_poller(tenant_id)
            # Merge onto any existing record so a Slack channel already
            # configured for this tenant survives a telegram (re)connect.
            rec = self.registry.get(tenant_id) or {}
            rec.update({
                "slug": slug,
                "status": "unpaired",
                "botToken": bot_token,
                "botUsername": bot_username,
                "avocadoMcpKey": avk,
                "pairingCode": code,
                "soul": soul,
            })
            rec.pop("telegramUserId", None)  # re-pair from scratch
            self.registry.put(tenant_id, rec)
            # Telegram isn't ready until paired; render brings up Slack alone
            # if present, or tears the profile down while we await pairing.
            await self._render_profile(tenant_id)
            self._start_poller(tenant_id)
        log.info("tenant provisioned (telegram unpaired): %s", slug)
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
        channel = str(body.get("channel") or "").strip().lower()
        if not tenant_id:
            return web.json_response({"ok": False, "error": "tenantId required"},
                                     status=400)
        slug = tenant_slug(tenant_id)
        async with self._lock(tenant_id):
            rec = self.registry.get(tenant_id)
            if channel in ("telegram", "slack") and rec is not None:
                # Per-channel disconnect: drop just one channel, keep the rest.
                if channel == "telegram":
                    self._stop_poller(tenant_id)
                    for k in ("botToken", "botUsername", "pairingCode",
                              "telegramUserId", "status"):
                        rec.pop(k, None)
                else:
                    rec.pop("slack", None)
                self.registry.put(tenant_id, rec)
                # Re-render with whatever remains; _render_profile tears the
                # profile down if nothing is left.
                await self._render_profile(tenant_id)
                tg_left = rec.get("status") == "paired" and bool(rec.get("botToken"))
                if not tg_left and not rec.get("slack"):
                    self.registry.remove(tenant_id)
                log.info("channel %s disconnected for %s", channel, slug)
            else:
                # Full teardown (no channel given) — account deletion path.
                self._stop_poller(tenant_id)
                await _run_hermes("-p", slug, "gateway", "stop", timeout=60)
                await _run_hermes("profile", "delete", slug, "--yes", timeout=60)
                # Belt & braces: the CLI delete unregisters the s6 slot; make
                # sure no profile remnants survive on the volume either way.
                shutil.rmtree(_profile_dir(slug), ignore_errors=True)
                self.registry.remove(tenant_id)
                log.info("tenant deprovisioned (full): %s", slug)
        return web.json_response({"ok": True})

    async def handle_provision_slack(self, request: web.Request) -> web.Response:
        """Connect (or refresh) a tenant's Slack channel. No pairing — the
        pasted xoxb-/xapp- token pair proves possession of the workspace app,
        so the channel lands ready immediately and the profile is rendered
        (alongside Telegram if already paired). Whole-workspace access: no
        per-user allowlist.
        """
        if (err := self._check_auth(request)) is not None:
            return err
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"},
                                     status=400)
        tenant_id = str(body.get("tenantId") or "").strip()
        avk = str(body.get("avocadoMcpKey") or "").strip()
        slack = body.get("slack") or {}
        bot_token = str(slack.get("botToken") or "").strip()
        app_token = str(slack.get("appToken") or "").strip()
        workspace = slack.get("workspaceName") or None
        soul_raw = body.get("soul")
        soul = str(soul_raw).strip() if isinstance(soul_raw, str) else None
        if not tenant_id or not avk or not bot_token or not app_token:
            return web.json_response(
                {"ok": False,
                 "error": "tenantId, avocadoMcpKey, slack.botToken, slack.appToken required"},
                status=400)
        if not bot_token.startswith("xoxb-") or not app_token.startswith("xapp-"):
            return web.json_response(
                {"ok": False, "error": "slack tokens must be xoxb-/xapp-"},
                status=400)
        if re.search(r"[\r\n\x00]", tenant_id + avk + bot_token + app_token):
            return web.json_response({"ok": False, "error": "invalid characters"},
                                     status=400)

        slug = tenant_slug(tenant_id)
        async with self._lock(tenant_id):
            rec = self.registry.get(tenant_id) or {}
            rec["slug"] = slug
            rec["avocadoMcpKey"] = avk
            if soul is not None:
                rec["soul"] = soul
            rec.setdefault("soul", None)
            rec["slack"] = {
                "botToken": bot_token,
                "appToken": app_token,
                "workspaceName": workspace,
            }
            self.registry.put(tenant_id, rec)
            ok = await self._render_profile(tenant_id)
        if not ok:
            return web.json_response({"ok": False, "error": "render failed"},
                                     status=500)
        log.info("slack connected for %s", slug)
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
    app.router.add_post("/provision-slack", controller.handle_provision_slack)
    app.router.add_post("/deprovision", controller.handle_deprovision)
    app.router.add_get("/health", controller.handle_health)

    async def _on_startup(_app: web.Application) -> None:
        await controller.start()

    app.on_startup.append(_on_startup)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
