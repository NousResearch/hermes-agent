"""Native chat surfaces for Collective Wisdom: Telegram and Slack cards, buttons and notices.

One card model (text + labelled actions) rendered as a Telegram inline keyboard or a Slack
Block Kit actions block. Button payloads are opaque ``wisdom:<token>:<idx>`` references to a
process-local card registry, so nothing about the action (skill ids, versions, hashes) rides on
the wire, a stale button after a restart simply reports itself expired, and Telegram's 64-byte
``callback_data`` cap never matters. Every tap is authorized against the adapter's own user
check before anything runs, and every mutation still passes through ``Wisdom``'s ``confirm``:
the confirm posts an Approve / Deny card and blocks the worker thread until an authorized human
taps, so a conversational "yes" (or a model) can never install or publish.

A supervised poller per connected platform delivers what the team published, what the update
policy applied or needs decided, and which local skills qualify for sharing — once per item,
to the platform's home channel (or the last chat that used ``/wisdom``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

POLL_INTERVAL = 600.0
FIRST_POLL_DELAY = 20.0
CONFIRM_TIMEOUT = 600.0
CARD_TTL = 24 * 3600.0
_CB_RE = re.compile(r"^wisdom:([0-9a-f]{12}):(\d{1,2})$")

Action = tuple[str, ...]


@dataclass
class Live:
    """One connected adapter: the platform SDK handle plus the adapter (authorization, config)."""
    platform: str
    native: Any
    adapter: Any
    home: str


@dataclass
class Card:
    token: str
    live: Live
    chat_id: str
    thread_id: Optional[str]
    actions: list[Action]
    message_id: Optional[str] = None
    created: float = field(default_factory=time.time)

    @property
    def platform(self) -> str:
        return self.live.platform


_ctx: Any = None                                 # PluginContext (supervised tasks)
_live: dict[tuple[str, str], Live] = {}         # (platform, profile home key) -> Live
_cards: dict[str, Card] = {}
_pending: dict[str, dict] = {}                   # confirm token -> {"event", "approved", "actor"}


def _state():
    from plugins.wisdom import state
    return state()


def _home_key() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


def live_for(platform: str) -> Optional[Live]:
    """The connected adapter of ``platform`` serving the current profile (multiplex-safe)."""
    return _live.get((platform, _home_key()))


# --- registration -----------------------------------------------------------------------------
def register(ctx) -> None:
    global _ctx
    _ctx = ctx
    ctx.register_telegram_handler(lambda native, adapter: _connect(ctx, "telegram", native, adapter))
    ctx.register_platform_handler("slack", lambda native, adapter: _connect(ctx, "slack", native, adapter))
    ctx.register_slack_action_handler(re.compile(r"^wisdom:"), _slack_action)


def _connect(ctx, platform: str, native: Any, adapter: Any) -> None:
    """Adapter ``connect()`` hook (runs under the adapter's profile scope): remember the live SDK
    handle, bind callbacks, start this profile's poller."""
    if native is None:
        return
    live = Live(platform, native, adapter, _home_key())
    _live[(platform, live.home)] = live
    if platform == "telegram":
        from telegram.ext import CallbackQueryHandler
        native.add_handler(CallbackQueryHandler(
            lambda update, context: _telegram_callback(update, context, live), pattern=_CB_RE))
    ctx.spawn_task(_poll(live), name=f"plugin:wisdom:notices:{platform}")


# --- cards ------------------------------------------------------------------------------------
def _new_card(live: Live, chat_id: str, thread_id: Optional[str], actions: list[Action]) -> Card:
    now = time.time()
    for tok in [t for t, c in _cards.items() if now - c.created > CARD_TTL]:
        _cards.pop(tok, None)
    card = Card(secrets.token_hex(6), live, str(chat_id), thread_id or None, actions)
    _cards[card.token] = card
    return card


def _tg_markup(token: str, labels: list[str]):
    """Inline keyboard, two buttons per row; PTB is imported here because it is an optional dependency
    that is only present when a Telegram adapter is connected."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keys = [InlineKeyboardButton(lbl[:64], callback_data=f"wisdom:{token}:{i}") for i, lbl in enumerate(labels)]
    return InlineKeyboardMarkup([keys[i:i + 2] for i in range(0, len(keys), 2)]) if keys else None


async def send_card(live: Live, chat_id: str, thread_id: Optional[str], text: str,
                    buttons: Optional[list[tuple[str, Action]]] = None) -> Card:
    native = live.native
    buttons = buttons or []
    card = _new_card(live, chat_id, thread_id, [a for _, a in buttons])
    labels = [label for label, _ in buttons]
    if live.platform == "telegram":
        kwargs: dict[str, Any] = {"chat_id": int(chat_id) if str(chat_id).lstrip("-").isdigit() else chat_id,
                                  "text": text[:4000], "reply_markup": _tg_markup(card.token, labels)}
        if thread_id and str(thread_id).isdigit():
            kwargs["message_thread_id"] = int(thread_id)
        msg = await native.bot.send_message(**kwargs)
        card.message_id = str(msg.message_id)
    else:
        blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}]
        if labels:
            blocks.append({"type": "actions", "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": lbl[:75]},
                 "action_id": f"wisdom:{card.token}:{i}", "value": card.token} for i, lbl in enumerate(labels)]})
        res = await native.client.chat_postMessage(channel=chat_id, text=text[:300], blocks=blocks,
                                                   thread_ts=thread_id or None)
        card.message_id = str(res.get("ts") or "")
    return card


async def edit_card(card: Card, text: str) -> None:
    """Replace a card's text and drop its buttons (outcome shown in place; taps become no-ops)."""
    _cards.pop(card.token, None)
    native = card.live.native
    try:
        if card.platform == "telegram":
            await native.bot.edit_message_text(chat_id=int(card.chat_id) if card.chat_id.lstrip("-").isdigit() else card.chat_id,
                                               message_id=int(card.message_id or 0), text=text[:4000], reply_markup=None)
        else:
            await native.client.chat_update(channel=card.chat_id, ts=card.message_id, text=text[:300],
                                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}])
    except Exception as exc:  # the outcome is also logged; a failed edit must not lose it
        logger.debug("wisdom card edit failed: %s", exc)


# --- callbacks (inbound wire data: authorize first) ----------------------------------------------
async def _telegram_callback(update, _context, live: Live) -> None:
    q = update.callback_query
    m = _CB_RE.match(q.data or "")
    if not m:
        return
    adapter = live.adapter
    chat = q.message.chat if q.message else None
    thread = getattr(q.message, "message_thread_id", None) if q.message else None
    ok = adapter._is_callback_user_authorized(
        str(q.from_user.id), chat_id=str(chat.id) if chat else None, chat_type=getattr(chat, "type", None),
        thread_id=str(thread) if thread is not None else None, user_name=getattr(q.from_user, "username", None))
    if not ok:
        await q.answer("Not authorized for Collective Wisdom actions.", show_alert=True)
        return
    await q.answer()
    reply = await _dispatch(m.group(1), int(m.group(2)), actor=q.from_user.username or str(q.from_user.id))
    if reply:
        await q.answer(reply[:190], show_alert=True)


async def _slack_action(ack, body, action) -> None:
    await ack()
    m = _CB_RE.match(str(action.get("action_id") or ""))
    if not m:
        return
    card = _cards.get(m.group(1))
    lives = [card.live] if card else [x for x in _live.values() if x.platform == "slack"]
    if not lives:
        return
    adapter = lives[0].adapter
    user = body.get("user") or {}
    channel = (body.get("channel") or {}).get("id", "")
    auth = getattr(adapter, "_is_interactive_user_authorized", None)
    ok = auth(str(user.get("id", "")), channel_id=channel, user_name=user.get("username") or user.get("name"),
              team_id=(body.get("team") or {}).get("id")) if auth else False
    if not ok:
        logger.warning("[wisdom] unauthorized Slack tap by %s ignored", user.get("id"))
        return
    reply = await _dispatch(m.group(1), int(m.group(2)), actor=user.get("username") or user.get("name") or str(user.get("id")))
    if reply:
        await lives[0].native.client.chat_postEphemeral(channel=channel, user=user.get("id"), text=reply[:3000])


async def _dispatch(token: str, idx: int, *, actor: str) -> Optional[str]:
    card = _cards.get(token)
    if card is None or idx >= len(card.actions):
        return "This Wisdom card has expired; run /wisdom again."
    verb, *args = card.actions[idx]
    handler = _ACTIONS.get(verb)
    if handler is None:
        return f"unknown action {verb}"
    try:
        return await handler(card, actor, *args)
    except Exception as exc:
        logger.warning("[wisdom] chat action %s failed: %s", verb, exc, exc_info=True)
        return f"wisdom: {exc}"


# --- blocking human confirmation ---------------------------------------------------------------
class ChatConfirm:
    """``Wisdom`` ``confirm`` for chat surfaces: post an Approve / Deny card from the worker thread
    and block until an authorized tap (or ``CONFIRM_TIMEOUT``). Approvals bind to the card, so
    the title + detail the human read is exactly what the service then applies."""

    def __init__(self, live: Live, chat_id: str, thread_id: Optional[str], loop: asyncio.AbstractEventLoop):
        self.live, self.chat_id, self.thread_id, self.loop = live, chat_id, thread_id, loop
        self.decisions: list[tuple[str, bool, str]] = []

    def __call__(self, title: str, detail: str) -> bool:
        fut = asyncio.run_coroutine_threadsafe(
            send_card(self.live, self.chat_id, self.thread_id, f"{title}\n\n{detail}\n\nApprove to proceed.",
                      [("✅ Approve", ("approve",)), ("✖ Deny", ("deny",))]), self.loop)
        card = fut.result(timeout=30)
        pending = {"event": threading.Event(), "approved": False, "actor": ""}
        _pending[card.token] = pending
        pending["event"].wait(timeout=CONFIRM_TIMEOUT)
        _pending.pop(card.token, None)
        if not pending["event"].is_set():
            asyncio.run_coroutine_threadsafe(edit_card(card, f"{title}\n\n⌛ No decision within the approval window; nothing was applied."), self.loop)
        self.decisions.append((title, bool(pending["approved"]), pending["actor"]))
        return bool(pending["approved"])


async def _act_decide(card: Card, actor: str, *, approved: bool) -> Optional[str]:
    pending = _pending.get(card.token)
    if pending is None:
        return "That approval already closed."
    pending["approved"], pending["actor"] = approved, actor
    pending["event"].set()
    await edit_card(card, ("✅ Approved by " if approved else "✖ Denied by ") + actor)
    return None


async def run_verb(live: Live, chat_id: str, thread_id: Optional[str], label: str,
                   fn: Callable[[Callable[[str, str], bool]], Any]) -> None:
    """Run a mutating ``Wisdom`` verb off the loop with a chat confirm; post the outcome as a card."""
    from plugins.wisdom.client import WisdomAuthError, WisdomError
    from plugins.wisdom.package import PackageError
    from plugins.wisdom.service import NotConfirmed
    confirm = ChatConfirm(live, chat_id, thread_id, asyncio.get_running_loop())
    try:
        result = await asyncio.to_thread(fn, confirm)
        text = f"{label}: done." if not result else f"{label}:\n{_fmt(result)}"
    except NotConfirmed as exc:
        text = f"{label}: {exc}."
    except WisdomAuthError as exc:
        text = f"{label}: {exc}. Run `hermes login` with your team account."
    except (WisdomError, PackageError, ValueError) as exc:
        text = f"{label} failed: {exc}"
    await send_card(live, chat_id, thread_id, text)


def _fmt(value: Any) -> str:
    import json
    if isinstance(value, list):
        return "\n".join(_fmt(v) for v in value) or "nothing to do"
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items() if v not in (None, "", [], {}) and k != "path")
    return json.dumps(value, default=str)


def _service():
    from plugins.wisdom import _service
    return _service()


# --- actions -----------------------------------------------------------------------------------
async def _act_install(card: Card, actor: str, skill_id: str, version: str = "") -> Optional[str]:
    await edit_card(card, f"Preparing install of {skill_id}{' v' + version if version else ''} (requested by {actor})…")
    _background(run_verb(card.live, card.chat_id, card.thread_id, f"Install {skill_id}",
                         lambda confirm: _service().install(skill_id, version=int(version) if version else None, confirm=confirm)), "install")
    return None


async def _act_update(card: Card, actor: str, skill_id: str = "") -> Optional[str]:
    await edit_card(card, f"Reviewing update{' of ' + skill_id if skill_id else 's'} (requested by {actor})…")
    _background(run_verb(card.live, card.chat_id, card.thread_id, f"Update {skill_id or 'all'}",
                         lambda confirm: _service().update(skill_id or None, confirm=confirm)), "update")
    return None


async def _act_keep(card: Card, actor: str, skill_id: str, version: str) -> Optional[str]:
    from plugins.wisdom import updates
    updates.defer(_state(), skill_id, int(version))
    await edit_card(card, f"Keeping your edited copy of {skill_id}; v{version} will not be applied ({actor}).")
    return None


async def _act_uninstall(card: Card, actor: str, skill_id: str) -> Optional[str]:
    await edit_card(card, f"Uninstall of {skill_id} requested by {actor}…")
    _background(run_verb(card.live, card.chat_id, card.thread_id, f"Uninstall {skill_id}",
                         lambda confirm: _service().uninstall(skill_id, confirm=confirm)), "uninstall")
    return None


async def _act_share(card: Card, actor: str, skill_name: str) -> Optional[str]:
    from plugins.wisdom import candidates
    await edit_card(card, f"Preparing {skill_name} for sharing (requested by {actor})…")
    _background(run_verb(card.live, card.chat_id, card.thread_id, f"Share {skill_name}",
                         lambda confirm: _service().share(skill_name, description=candidates.default_description(skill_name), confirm=confirm)), "share")
    return None


async def _act_not_now(card: Card, actor: str, skill_name: str) -> Optional[str]:
    from plugins.wisdom import candidates
    until = candidates.defer(_state(), skill_name)
    await edit_card(card, f"{skill_name} will not be suggested again before {until} ({actor}).")
    return None


async def _act_mute(card: Card, actor: str) -> Optional[str]:
    from plugins.wisdom import notices
    notices.mute(_state(), 24)
    await edit_card(card, f"Collective Wisdom notices muted for 24 hours ({actor}). `/wisdom mute 0` to unmute.")
    return None


async def _act_show(card: Card, actor: str, skill_id: str) -> Optional[str]:
    detail = await asyncio.to_thread(_service().show, skill_id)
    latest = detail.get("latest") or {}
    sec = latest.get("security_check") or {}
    text = (f"{(detail.get('skill') or {}).get('slug') or skill_id} — latest v{latest.get('version')}\n"
            f"security: {sec.get('status')} — {sec.get('summary', '')}\n"
            f"content_hash: {latest.get('content_hash')}\n"
            f"{latest.get('author_description') or ''}").strip()
    installed = detail.get("installed_version")
    buttons = [] if installed == latest.get("version") else [("⬇ Install", ("install", skill_id, str(latest.get("version") or "")))]
    await send_card(card.live, card.chat_id, card.thread_id, text, buttons)
    return None


_ACTIONS: dict[str, Callable[..., Any]] = {
    "approve": lambda card, actor: _act_decide(card, actor, approved=True),
    "deny": lambda card, actor: _act_decide(card, actor, approved=False),
    "install": _act_install, "update": _act_update, "keep": _act_keep, "uninstall": _act_uninstall,
    "share": _act_share, "not-now": _act_not_now, "mute": _act_mute, "show": _act_show,
}


# --- /wisdom on a chat platform ------------------------------------------------------------------
def slash_target() -> Optional[tuple[Live, str, Optional[str]]]:
    """``(live, chat_id, thread_id)`` when the current turn is a connected chat platform."""
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    live = live_for(platform) if platform else None
    if live is None or not chat_id:
        return None
    homes = dict(_state().get("chat_home") or {})
    homes[platform] = {"chat_id": chat_id, "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", "") or None}
    _state().set("chat_home", homes)
    return live, chat_id, homes[platform]["thread_id"]


def _background(coro, name: str) -> None:
    """Mutations run supervised off the slash reply: the confirm may wait minutes for a tap, and the
    chat must stay usable meanwhile."""
    if _ctx is not None:
        _ctx.spawn_task(coro, name=f"plugin:wisdom:{name}")
    else:
        asyncio.get_running_loop().create_task(coro)


async def slash(ns, target: tuple[Live, str, Optional[str]]) -> Optional[str]:
    """Card-rendering ``/wisdom`` for chat platforms; ``None`` when a card was (or will be) sent, else text."""
    from plugins.wisdom import candidates, updates
    live, chat_id, thread_id = target
    cmd = ns.wisdom_command or "status"
    svc = _service()
    if cmd == "list":
        rows = await asyncio.to_thread(svc.browse)
        if not rows:
            return "No shared skills in your team yet."
        installed = _state().get("installed") or {}
        text = "\n".join(f"• {r['slug'] or r['id']} v{r['version']} · {r['installs']} installs · security {r['security']}"
                         + (f"\n  {r['description']}" if r.get("description") else "") for r in rows[:12])
        buttons = [(f"⬇ {r['slug'] or r['id']}"[:40], ("install", r["id"], "")) for r in rows if r["id"] not in installed][:6]
        await send_card(live, chat_id, thread_id, text, buttons)
        return None
    if cmd in ("status", "updates"):
        rows = await asyncio.to_thread(updates.pending, svc)
        installed = _state().get("installed") or {}
        lines = [f"{len(installed)} Wisdom skill(s) installed."] + [
            f"• {u['slug']} v{u['installed']} → v{u['latest']} ({u['mode']}, {u['action']}"
            + (", you edited your copy" if u["modified"] else "") + ")" for u in rows]
        buttons: list[tuple[str, Action]] = []
        for u in rows:
            if u["action"] == "conflict":
                buttons += [(f"↻ Replace {u['slug']}"[:40], ("update", u["skill_id"])),
                            (f"✋ Keep mine {u['slug']}"[:40], ("keep", u["skill_id"], str(u["latest"])))]
            elif u["action"] in ("manual", "deferred"):
                buttons.append((f"⬆ Update {u['slug']}"[:40], ("update", u["skill_id"])))
        buttons.append(("🔕 Mute 24h", ("mute",)))
        await send_card(live, chat_id, thread_id, "\n".join(lines), buttons[:8])
        return None
    if cmd == "candidates":
        rows = candidates.qualify(_state())
        if not rows:
            return "No local skill qualifies as a share candidate right now."
        buttons = []
        for c in rows:
            candidates.mark_presented(_state(), c["skill"])
            buttons += [(f"📤 Share {c['skill']}"[:40], ("share", c["skill"])), (f"⏸ Not now {c['skill']}"[:40], ("not-now", c["skill"]))]
        await send_card(live, chat_id, thread_id, "Share candidates:\n" + "\n".join(f"• {candidates.describe(c)}" for c in rows), buttons[:8])
        return None
    if cmd == "show":
        card = _new_card(live, chat_id, thread_id, [])
        await _act_show(card, "", ns.skill_id)
        return None
    if cmd == "install":
        _background(run_verb(live, chat_id, thread_id, f"Install {ns.skill_id}",
                             lambda confirm: svc.install(ns.skill_id, version=ns.version, confirm=confirm)), "install")
        return None
    if cmd == "update" and not getattr(ns, "keep", False):
        _background(run_verb(live, chat_id, thread_id, f"Update {ns.skill_id or 'all'}",
                             lambda confirm: svc.update(ns.skill_id, confirm=confirm)), "update")
        return None
    if cmd == "uninstall":
        _background(run_verb(live, chat_id, thread_id, f"Uninstall {ns.skill_id}",
                             lambda confirm: svc.uninstall(ns.skill_id, confirm=confirm)), "uninstall")
        return None
    if cmd == "share":
        _background(run_verb(live, chat_id, thread_id, f"Share {ns.skill_name}",
                             lambda confirm: svc.share(ns.skill_name, description=ns.description, confirm=confirm)), "share")
        return None
    return await asyncio.to_thread(_text_dispatch, ns)


def _text_dispatch(ns) -> str:
    from plugins.wisdom import _dispatch as dispatch
    return dispatch(ns)[0]


# --- proactive delivery ----------------------------------------------------------------------------
def _target(live: Live, st) -> Optional[tuple[str, Optional[str]]]:
    home = getattr(getattr(live.adapter, "config", None), "home_channel", None)
    if home is not None and getattr(home, "chat_id", None):
        return str(home.chat_id), getattr(home, "thread_id", None)
    last = (st.get("chat_home") or {}).get(live.platform)
    return (last["chat_id"], last.get("thread_id")) if last else None


def pending_items(st, *, now: float | None = None) -> list[tuple[str, str, list[tuple[str, Action]]]]:
    """``(dedupe key, text, buttons)`` for everything a human should hear about right now."""
    from plugins.wisdom import candidates, notices, updates
    items: list[tuple[str, str, list[tuple[str, Action]]]] = []
    for n in notices.refresh(st, now=now):
        have = f" (you have v{n['installed']})" if n.get("installed") else ""
        items.append((f"notice:{n['skill_id']}:{n['version']}", f"Team skill {n['kind']}: {n['skill_id']} v{n['version']}{have}",
                      [("🔍 Show", ("show", n["skill_id"])), ("⬇ Install", ("install", n["skill_id"], str(n["version"] or "")))]))
    if st.get("installed"):
        report = updates.sweep(st, now=now)
        for r in report.get("applied") or []:
            kept = "; your edited copy was kept aside" if r.get("preserved_local_edits") else ""
            items.append((f"applied:{r['skill_id']}:{r['latest']}",
                          f"Updated {r['slug']} v{r['installed']} → v{r['latest']} under the team's {r['mode']} policy{kept}.", []))
        for r in report.get("conflicts") or []:
            items.append((f"conflict:{r['skill_id']}:{r['latest']}",
                          f"{r['slug']} v{r['latest']} is available ({r['mode']}) but you edited your copy.",
                          [("↻ Replace (keep a copy of my edits)", ("update", r["skill_id"])), ("✋ Keep mine", ("keep", r["skill_id"], str(r["latest"])))]))
        for r in report.get("manual") or []:
            items.append((f"update:{r['skill_id']}:{r['latest']}",
                          f"Update available: {r['slug']} v{r['installed']} → v{r['latest']}" + (" (REQUIRED by your org)" if r.get("required") else ""),
                          [("⬆ Update", ("update", r["skill_id"])), ("🔍 Show", ("show", r["skill_id"]))]))
    for c in candidates.qualify(st, now=now):
        candidates.mark_presented(st, c["skill"], now=now)
        items.append((f"candidate:{c['skill']}:{time.strftime('%G-W%V', time.localtime(now or time.time()))}",
                      f"Worth sharing with your team? {candidates.describe(c)}.",
                      [("📤 Share", ("share", c["skill"])), ("⏸ Not now", ("not-now", c["skill"]))]))
    return items


async def deliver(live: Live, *, now: float | None = None) -> int:
    """One delivery pass for a connected adapter; returns how many cards were sent."""
    from plugins.wisdom.client import entitled
    st = _state()
    now = now or time.time()
    if not entitled() or st.get("muted_until", 0) > now:
        return 0
    target = _target(live, st)
    if target is None:
        return 0
    delivered = dict(st.get("chat_delivered") or {})
    mine = {k: v for k, v in (delivered.get(live.platform) or {}).items() if now - v < 30 * 86400}
    sent = 0
    for key, text, buttons in pending_items(st, now=now):
        if key in mine:
            continue
        await send_card(live, target[0], target[1], text, buttons)
        mine[key] = now
        sent += 1
    delivered[live.platform] = mine
    st.set("chat_delivered", delivered)
    return sent


async def _poll(live: Live) -> None:
    await asyncio.sleep(FIRST_POLL_DELAY)
    while True:
        try:
            await deliver(live)
        except Exception as exc:  # the poller must outlive a bad Gateway answer
            logger.debug("[wisdom] %s notice delivery skipped: %s", live.platform, exc)
        await asyncio.sleep(POLL_INTERVAL)
