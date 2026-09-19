"""Collective Wisdom parity surfaces: update policy + conflicts, share-candidate qualification,
Telegram/Slack cards with authorized, hash-bound native consent, proactive delivery.

Fake natives stand in for the PTB ``Application`` / slack_bolt ``AsyncApp``; the Gateway is the
same in-memory fake as ``test_wisdom_plugin.py``; installs hit a real temp ``HERMES_HOME``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path

import pytest

VECTORS = json.loads((Path(__file__).parent / "wisdom_hash_vectors.json").read_text(encoding="utf-8"))
DAY = 86400.0


class _State(dict):
    def __init__(self, tmp: Path, **kw):
        super().__init__(**kw)
        self.data_dir = tmp / "wisdom-state"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get(self, k, default=None):
        return super().get(k, default)

    def set(self, k, v):
        self[k] = v


class _Gateway:
    """Two published skills; ``modes`` is the org policy per installation row."""
    org_id = "org-test"
    owner = "owner"

    def __init__(self, latest=2, mode="MANUAL", security="pass"):
        self.files = [(f["path"], f["mode"], base64.b64decode(f["content_base64"])) for f in VECTORS["files"]]
        self.latest, self.mode, self.security = latest, mode, security
        self.recorded, self.identities, self.feed_events = [], [], []

    def register_identity(self, ident):
        self.identities.append(ident)

    def skill(self, skill_id):
        return {"skill": {"id": skill_id, "slug": "canonical", "state": "active", "takedown_generation": 0,
                          "update_mode": self.mode},
                "versions": [{"version": v} for v in range(1, self.latest + 1)]}

    def version(self, skill_id, version):
        return {"version": {"version": version, "content_hash": VECTORS["content_hash"],
                            "security_check": {"status": self.security, "summary": "ok"}}}

    def content(self, skill_id, version, *, installation_id, takedown_generation):
        return VECTORS["content_hash"], self.files

    def record_install(self, **kw):
        self.recorded.append(kw)
        return {"installed_version": kw["version"], "effective_update_mode": self.mode}

    def installations(self, installation_id):
        return [{"skill_id": "sk1", "latest_version": self.latest, "update_mode": self.mode}]

    def list_skills(self, *, cursor=None):
        return {"skills": [{"id": "sk1", "slug": "canonical", "latest_version": self.latest, "install_count": 3,
                            "state": "active", "author_description": "Team canonical skill",
                            "security_check": {"status": self.security}}], "next_cursor": None}

    def feed(self, cursor=None):
        return {"events": self.feed_events, "next_cursor": None}

    def deactivate(self, installation_id, skill_id):
        pass


def _installed(tmp_path, monkeypatch, gw, *, version=1):
    """A real v1 install into the temp home (ledger + files), returning (service, state)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.wisdom.service import Wisdom
    state = _State(tmp_path, installation_id="inst-1")
    svc = Wisdom(state, client=gw)
    gw.latest = version
    svc.install("sk1", version=version, confirm=lambda *_: True)
    return svc, state


# --- update policy -------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,edited,expect", [
    ("MANUAL", False, "manual"), ("MANUAL", True, "manual"),
    ("AUTO_WITH_NOTICE", False, "auto"), ("AUTO_WITH_NOTICE", True, "conflict"),
    ("REQUIRED", False, "auto"), ("REQUIRED", True, "auto"),
])
def test_update_policy_never_overwrites_local_edits_silently(tmp_path, monkeypatch, mode, edited, expect):
    from plugins.wisdom import updates
    gw = _Gateway(mode=mode)
    svc, state = _installed(tmp_path, monkeypatch, gw)
    path = Path(state["installed"]["sk1"]["path"])
    if edited:
        (path / "SKILL.md").write_text("# my local tweak\n", encoding="utf-8")
    gw.latest = 2
    rows = updates.pending(svc)
    assert [r["action"] for r in rows] == [expect] and rows[0]["modified"] is edited

    report = updates.apply_automatic(svc, rows)
    if expect == "auto":
        assert [r["latest"] for r in report["applied"]] == [2] and state["installed"]["sk1"]["version"] == 2
        # REQUIRED with edits lands but keeps the edited copy aside, outside the skills tree.
        kept = report["applied"][0]["preserved_local_edits"]
        assert bool(kept) is edited
        if kept:
            assert (Path(kept) / "SKILL.md").read_text(encoding="utf-8") == "# my local tweak\n"
            assert not Path(kept).is_relative_to(tmp_path / "skills")
    else:
        assert state["installed"]["sk1"]["version"] == 1 and len(gw.recorded) == 1
        assert [r["skill_id"] for r in report[expect + "s" if expect == "conflict" else "manual"]] == ["sk1"]
        # "Keep mine" pins exactly this version; a newer one asks again.
        updates.defer(state, "sk1", 2)
        assert updates.pending(svc)[0]["action"] == "deferred"
        gw.latest = 3
        assert updates.pending(svc)[0]["action"] == expect


def test_automatic_update_requires_a_passing_security_verdict(tmp_path, monkeypatch):
    from plugins.wisdom import updates
    gw = _Gateway(mode="AUTO_WITH_NOTICE")
    svc, state = _installed(tmp_path, monkeypatch, gw)
    gw.latest, gw.security = 2, "unknown"
    report = updates.apply_automatic(svc)
    assert report["applied"] == [] and [r["skill_id"] for r in report["manual"]] == ["sk1"]
    assert state["installed"]["sk1"]["version"] == 1


# --- share candidates ------------------------------------------------------------------------------
def test_candidate_qualification_is_deterministic_and_quota_bound(tmp_path, monkeypatch):
    from plugins.wisdom import candidates
    monkeypatch.setattr(candidates, "_eligible", lambda name, prov: not name.startswith("_wisdom"))
    state = _State(tmp_path)
    monday = time.mktime((2026, 9, 7, 12, 0, 0, 0, 0, -1))  # Mon 2026-09-07
    # 7 consecutive business days: Mon..Fri + next Mon, Tue (a weekend in between does not break the run).
    for i in (0, 1, 2, 3, 4, 7, 8):
        candidates.observe(state, action="loaded", skill_name="daily-standup", now=monday + i * DAY)
        candidates.observe(state, action="loaded", skill_name="_wisdom/team", now=monday + i * DAY)
    # 6 business days with a gap is not enough.
    for i in (0, 1, 2, 3, 7, 8):
        candidates.observe(state, action="loaded", skill_name="gappy", now=monday + i * DAY)
    # Refined 3x, then stable for a week and still used.
    for i in (0, 1, 2):
        candidates.observe(state, action="patched", skill_name="polished", now=monday + i * DAY)
    candidates.observe(state, action="loaded", skill_name="polished", now=monday + 12 * DAY)
    now = monday + 12 * DAY
    got = {c["skill"]: c["reason"] for c in candidates.qualify(state, now=now)}
    assert got == {"daily-standup": "high_usage", "polished": "refinement"}
    assert candidates.consecutive_business_days(state["candidate_facts"]["gappy"]["days"]) == 4

    # Weekly quota: presenting counts; a presented skill is not re-raised the same week.
    candidates.mark_presented(state, "daily-standup", now=now)
    assert [c["skill"] for c in candidates.qualify(state, now=now)] == ["polished"]
    # Not now silences until the returned date; a shared skill is never a candidate again.
    until = candidates.defer(state, "polished", days=2, now=now)
    assert candidates.qualify(state, now=now) == [] and until == time.strftime("%Y-%m-%d", time.localtime(now + 2 * DAY))
    # A new ISO week re-raises last week's presented skill too (weekly cadence), never a shared one.
    assert {c["skill"] for c in candidates.qualify(state, now=now + 3 * DAY)} == {"daily-standup", "polished"}
    state.set("shared", {"polished": {"slug": "polished"}})
    assert [c["skill"] for c in candidates.qualify(state, now=now + 3 * DAY)] == ["daily-standup"]


# --- chat surfaces -----------------------------------------------------------------------------------
class _TgMsg:
    def __init__(self, mid, chat_id, text, markup):
        self.message_id, self.text, self.reply_markup = mid, text, markup
        self.chat = type("Chat", (), {"id": int(chat_id), "type": "private"})()
        self.message_thread_id = None


class _TgBot:
    def __init__(self):
        self.sent, self.edits, self.n = [], [], 0

    async def send_message(self, **kw):
        self.n += 1
        msg = _TgMsg(self.n, kw["chat_id"], kw["text"], kw.get("reply_markup"))
        self.sent.append(msg)
        return msg

    async def edit_message_text(self, **kw):
        self.edits.append(kw)


class _TgApp:
    def __init__(self):
        self.bot, self.handlers = _TgBot(), []

    def add_handler(self, h):
        self.handlers.append(h)


class _TgAdapter:
    allowed = {"1001"}

    def __init__(self):
        self.config = type("Cfg", (), {"home_channel": type("Home", (), {"chat_id": "555", "thread_id": None})()})()

    def _is_callback_user_authorized(self, user_id, **_):
        return user_id in self.allowed


class _Ctx:
    def __init__(self):
        self.tasks = []

    def spawn_task(self, coro, *, name=None):
        t = asyncio.get_running_loop().create_task(coro, name=name)
        self.tasks.append(t)
        return t


def _buttons(msg):
    return [(b.text, b.callback_data) for row in msg.reply_markup.inline_keyboard for b in row]


class _Btn:
    def __init__(self, text, data):
        self.text, self.callback_data = text, data


def _fake_markup(token, labels):
    """PTB is optional in CI; mirror InlineKeyboardMarkup's shape (rows of buttons with text + callback_data)."""
    keys = [_Btn(lbl, f"wisdom:{token}:{i}") for i, lbl in enumerate(labels)]
    return type("Markup", (), {"inline_keyboard": [keys[i:i + 2] for i in range(0, len(keys), 2)]})() if keys else None


async def _tap(chat, msg, data, user_id, username="tek"):
    """Simulate a Telegram callback query through the handler the plugin registered on the PTB app."""
    answers = []
    handler = chat._TG_HANDLERS[-1]

    class Q:
        def __init__(self):
            self.data, self.message = data, msg
            self.from_user = type("U", (), {"id": int(user_id), "username": username})()

        async def answer(self, text=None, show_alert=False):
            answers.append(text)

    update = type("Upd", (), {"callback_query": Q()})()
    await handler(update, None)
    return answers


def test_telegram_card_flow_is_authorized_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.wisdom as wisdom
    import plugins.wisdom.service as service_mod
    from plugins.wisdom import chat
    from plugins.wisdom.service import Wisdom
    gw = _Gateway(latest=1)
    state = _State(tmp_path)
    monkeypatch.setattr(wisdom, "state", lambda: state)
    monkeypatch.setattr(service_mod, "WisdomClient", lambda: gw)
    monkeypatch.setattr(chat, "CONFIRM_TIMEOUT", 5.0)
    monkeypatch.setattr(chat, "_tg_markup", _fake_markup)
    handlers = []
    import sys, types
    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.CallbackQueryHandler = lambda cb, pattern=None: handlers.append(cb) or ("cb", cb, pattern)
    monkeypatch.setitem(sys.modules, "telegram", sys.modules.get("telegram") or types.ModuleType("telegram"))
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setattr(chat, "_TG_HANDLERS", handlers, raising=False)
    chat._live.clear(); chat._cards.clear(); chat._pending.clear()
    app, adapter, ctx = _TgApp(), _TgAdapter(), _Ctx()

    async def scenario():
        chat._connect(ctx, "telegram", app, adapter)
        for t in ctx.tasks:  # the poller is started; not exercised here
            t.cancel()
        live = chat.live_for("telegram")
        assert live is not None and app.handlers  # the callback handler was bound to THIS bot
        # /wisdom list typed in a Telegram chat: the gateway's plugin-command dispatch gets a coroutine
        # (awaited by the gateway) that renders a card whose install button carries no wire payload.
        from gateway.session_context import clear_session_vars, set_session_vars
        tokens = set_session_vars(platform="telegram", chat_id="555", user_id="1001", session_key="telegram:555")
        try:
            pending = wisdom._slash("list")
            assert asyncio.iscoroutine(pending)
            assert await pending is None
        finally:
            clear_session_vars(tokens)
        assert state["chat_home"]["telegram"]["chat_id"] == "555"
        listing = app.bot.sent[-1]
        (label, data), = _buttons(listing)
        assert label.startswith("⬇") and re.fullmatch(r"wisdom:[0-9a-f]{12}:0", data) and "sk1" not in data

        # An unauthorized user's tap is refused before anything runs.
        assert "Not authorized" in (await _tap(chat, listing, data, user_id="9999"))[0]
        assert gw.recorded == []

        # Authorized tap -> the service posts its consent card (title + hash + verdict) and waits.
        await _tap(chat, listing, data, user_id="1001")
        for _ in range(50):
            await asyncio.sleep(0.05)
            if len(app.bot.sent) >= 2:
                break
        consent = app.bot.sent[-1]
        assert "Install Wisdom skill canonical v1" in consent.text and VECTORS["content_hash"] in consent.text
        assert "security: pass" in consent.text
        assert gw.recorded == []  # nothing applied before the human decides
        approve, deny = _buttons(consent)
        assert deny[0].startswith("✖")
        # Deny from a stranger is ignored; deny from the owner withdraws.
        await _tap(chat, consent, deny[1], user_id="9999")
        assert gw.recorded == [] and chat._pending
        await _tap(chat, consent, approve[1], user_id="1001")
        for _ in range(100):
            await asyncio.sleep(0.05)
            if gw.recorded:
                break
        assert gw.recorded and gw.recorded[0]["version"] == 1
        assert (tmp_path / "skills" / "_wisdom" / "org-test" / "canonical" / "SKILL.md").exists()
        assert any("Approved by tek" in e["text"] for e in app.bot.edits)
        # The stale card token is gone: a re-tap reports expiry instead of re-running the action.
        assert "expired" in (await _tap(chat, consent, approve[1], user_id="1001"))[-1]
        for t in ctx.tasks:
            t.cancel()

    asyncio.run(scenario())
    assert isinstance(Wisdom(state, client=gw).status()["installed"]["sk1"]["version"], int)


def test_proactive_delivery_sends_each_team_event_once_per_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.wisdom as wisdom
    import plugins.wisdom.service as service_mod
    from plugins.wisdom import chat
    gw = _Gateway(latest=2, mode="MANUAL")
    gw.feed_events = [{"kind": "new", "skill_id": "sk2", "version": 1}]
    svc, state = _installed(tmp_path, monkeypatch, gw, version=1)
    gw.latest = 2
    monkeypatch.setattr(wisdom, "state", lambda: state)
    monkeypatch.setattr(service_mod, "WisdomClient", lambda: gw)
    monkeypatch.setattr("plugins.wisdom.client.WisdomClient", lambda **_: gw)
    monkeypatch.setattr("plugins.wisdom.client.entitled", lambda scope="wisdom:read": True)
    monkeypatch.setattr(chat, "_tg_markup", _fake_markup)
    chat._live.clear(); chat._cards.clear()
    app = _TgApp()
    live = chat.Live("telegram", app, _TgAdapter(), "home")

    async def scenario():
        sent = await chat.deliver(live, now=10_000.0)
        texts = [m.text for m in app.bot.sent]
        assert sent == 2 and any("sk2 v1" in t for t in texts) and any("canonical v1 → v2" in t for t in texts)
        # Same items, next poll: nothing is repeated, even past the poll interval.
        assert await chat.deliver(live, now=10_000.0 + 2 * chat.POLL_INTERVAL) == 0
        # Muted: silence; a home channel is the target so a 555 chat received everything.
        state.set("muted_until", 99_999.0)
        gw.feed_events.append({"kind": "new", "skill_id": "sk3", "version": 1})
        assert await chat.deliver(live, now=20_000.0) == 0
        assert {str(m.chat.id) for m in app.bot.sent} == {"555"}

    asyncio.run(scenario())


class _SlackClient:
    def __init__(self):
        self.posted, self.updated, self.ephemeral = [], [], []

    async def chat_postMessage(self, **kw):
        self.posted.append(kw)
        return {"ts": f"{len(self.posted)}.000"}

    async def chat_update(self, **kw):
        self.updated.append(kw)

    async def chat_postEphemeral(self, **kw):
        self.ephemeral.append(kw)


class _SlackAdapter:
    def _is_interactive_user_authorized(self, user_id, **_):
        return user_id == "U1"


def test_slack_blocks_carry_opaque_actions_and_refuse_strangers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.wisdom as wisdom
    from plugins.wisdom import chat
    state = _State(tmp_path)
    monkeypatch.setattr(wisdom, "state", lambda: state)
    chat._live.clear(); chat._cards.clear(); chat._pending.clear()
    native = type("App", (), {"client": _SlackClient()})()
    live = chat.Live("slack", native, _SlackAdapter(), "home")
    chat._live[("slack", "home")] = live

    async def scenario():
        card = await chat.send_card(live, "C1", "171.5", "Mute?", [("🔕 Mute 24h", ("mute",))])
        post = native.client.posted[-1]
        assert post["thread_ts"] == "171.5" and post["blocks"][1]["elements"][0]["action_id"] == f"wisdom:{card.token}:0"
        assert "mute" not in json.dumps(post["blocks"][1])  # the verb never rides on the wire
        acked = []

        async def ack():
            acked.append(True)
        body = {"user": {"id": "U9", "username": "eve"}, "channel": {"id": "C1"}, "team": {"id": "T1"}}
        await chat._slack_action(ack, body, {"action_id": f"wisdom:{card.token}:0"})
        assert acked == [True] and "muted_until" not in state and native.client.updated == []
        body["user"] = {"id": "U1", "username": "tek"}
        await chat._slack_action(ack, body, {"action_id": f"wisdom:{card.token}:0"})
        assert state["muted_until"] > time.time() and "tek" in native.client.updated[-1]["text"]
        # Stale token after the card closed: an ephemeral hint, no second mutation.
        state.set("muted_until", 0)
        await chat._slack_action(ack, body, {"action_id": f"wisdom:{card.token}:0"})
        assert state["muted_until"] == 0 and "expired" in native.client.ephemeral[-1]["text"]

    asyncio.run(scenario())


def test_desktop_router_keeps_conflicts_and_gates_share_on_gateway_verdicts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import plugins.wisdom.service as service_mod
    from plugins.wisdom.dashboard import plugin_api
    gw = _Gateway(mode="AUTO_WITH_NOTICE")
    svc, state = _installed(tmp_path, monkeypatch, gw)
    (Path(state["installed"]["sk1"]["path"]) / "SKILL.md").write_text("edited\n", encoding="utf-8")
    gw.latest = 2
    monkeypatch.setattr(service_mod, "WisdomClient", lambda: gw)
    monkeypatch.setattr(plugin_api, "entitlement", lambda: {"org_id": "org-test", "scopes": ("wisdom:read",)})
    monkeypatch.setattr(plugin_api, "state", lambda: state)
    http = TestClient(_mk_app(plugin_api.router))

    over = http.get("/overview").json()
    (upd,) = over["status"]["updates"]
    assert upd["action"] == "conflict" and upd["modified"] is True and over["candidates"] == []
    assert http.post("/update/keep", json={"skill_id": "sk1", "version": 2}).status_code == 200
    assert http.get("/overview").json()["status"]["updates"][0]["action"] == "deferred"
    assert state["installed"]["sk1"]["version"] == 1  # the overview sweep applied nothing: conflict, then deferred

    # Share: confirm 2 passes only on pass+pass; anything else withdraws the draft and surfaces the verdict.
    seen = []

    def share(self, name, *, description, confirm):
        assert confirm("Share x with your team as x", "content_hash: sha256:" + "a" * 64) is True
        assert confirm("Share x with your team as x", "content_hash: sha256:" + "b" * 64) is False
        seen.append(confirm("Publish x to your team", "security: pass — ok\nprofessionalism: pass — ok"))
        seen.append(confirm("Publish x to your team", "security: pass — ok\nprofessionalism: needs_review — tone"))
        return {"ok": True}
    monkeypatch.setattr(service_mod.Wisdom, "share", share)
    r = http.post("/share", json={"skill_name": "x", "description": "d", "content_hash": "sha256:" + "a" * 64})
    assert r.status_code == 200 and seen == [True, False]


def _mk_app(router):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return app
