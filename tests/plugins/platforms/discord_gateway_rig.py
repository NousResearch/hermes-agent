"""A real end-to-end Discord -> multiplexed gateway rig: send one Discord message, see what ran where.

Use it for anything that re-homes, routes or rewrites a Discord message (``$<profile>`` mention
dispatch, ``profile_routes``, auto-threads): it runs the REAL pipeline an inbound message takes::

    DiscordAdapter._handle_message        (gates, attachments, backfill, $<profile> dispatch)
      -> the gateway's real message handler (``_make_default_profile_message_handler`` or a
         secondary bot's ``_make_profile_message_handler``: native callback + runtime scope)
      -> GatewayRunner._handle_message     (admission, command handling)
      -> the session authority's durable FIFO (``admit_native`` -> ``capture_provenance`` ->
         ``session_ingress_context._binding``, then ``restore_provenance`` when the turn drains)
      -> GatewayRunner._handle_message_with_agent (session setup, reply shaping, label, delivery)

Only three things are stand-ins: the model call (``GatewayRunner._run_agent`` answers ``ok``), the
Discord wire (``send`` records the reply instead of posting it) and the profile directory lookup
(``hermes_cli.profiles``, pointed at a temp tree). Everything a message passes through on its way
to a turn, including the authority binding that refuses a turn it cannot prove (the
``profile_mismatch`` that tests calling ``resolve_identity`` directly never reach), is real.

The multiplexing set-up mirrors a real ``gateway.multiplex_profiles`` server (each piece is needed;
leaving one out yields a ``profile_mismatch`` that looks like a product bug): the process latch
(``set_multiplex_active``), the reserved runtime homes ``_binding`` trusts
(``config._runtime_profile_homes``), one session authority per served home in a multiplexed
``SessionAuthorities``, and each served profile's own ``config.yaml`` loaded through the real
loader into ``runner._profile_configs``.

Typical use (see ``test_discord_profile_mentions_e2e.py``)::

    @pytest.fixture
    async def rig(tmp_path, monkeypatch):
        async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE) as rig:
            yield rig

    async def test_x(rig):
        outcome = await rig.send("$agent-one status?")
        assert outcome.turns_by_profile() == {"agent-one": ["status?"]}
        assert outcome.refusals == []

``Outcome`` records what really happened: the events the bot handed to the gateway (``handed``),
each model turn with the profile, runtime home and session it ran in (``turns``), every reply
posted to Discord (``replies``), every ``AuthorityRefusal`` the gateway raised, including ones it
turned into a notice (``refusals``), and any exception that escaped its message handler (``errors``).
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import hermes_yaml as yaml

#: The user every rig message comes from, and the guild/channel it is posted in.
USER_ID, GUILD_ID, CHANNEL_ID = 42, 777, 555
#: Replies the gateway sends on its own that are not an answer to the message (first-contact tips).
_TIP_MARKERS = ("home channel", "/sethome")
_MESSAGE_IDS = itertools.count(10_000)
_WRITES = itertools.count()


@dataclasses.dataclass(frozen=True)
class Handed:
    """An event as the bot handed it to the gateway (a snapshot: the gateway edits events in place)."""

    bot: str
    message_type: Any
    text: str
    profile: Optional[str]
    profile_mention: bool


@dataclasses.dataclass(frozen=True)
class Turn:
    """One model turn: the profile and home it ran under, the session it ran in, and its prompt."""

    profile: Optional[str]
    home: Path
    session_key: Optional[str]
    session_id: Optional[str]
    message: Optional[str]
    context_prompt: Optional[str]


@dataclasses.dataclass(frozen=True)
class Refusal:
    """An ``AuthorityRefusal`` the gateway raised (caught later or not): its reason, the function that
    raised it (``_binding`` for the session binding) and the gateway functions above it."""

    reason: str
    where: str
    chain: tuple[str, ...]


@dataclasses.dataclass
class Outcome:
    """Everything one ``DiscordGatewayRig.send`` caused, end to end."""

    handed: list[Handed] = dataclasses.field(default_factory=list)
    turns: list[Turn] = dataclasses.field(default_factory=list)
    replies: list[tuple[str, str]] = dataclasses.field(default_factory=list)  # (bot, text)
    refusals: list[Refusal] = dataclasses.field(default_factory=list)
    errors: list[BaseException] = dataclasses.field(default_factory=list)
    commands: list[tuple[str, Optional[str], Path]] = dataclasses.field(default_factory=list)  # (command, profile, home)

    def turns_by_profile(self) -> dict[Optional[str], list[str]]:
        """``{profile: [prompt, ...]}`` for every turn that ran (``None``: the channel's own session)."""
        out: dict[Optional[str], list[str]] = {}
        for turn in self.turns:
            out.setdefault(turn.profile, []).append(turn.message)
        return out

    def reply_texts(self) -> list[str]:
        return [text for _bot, text in self.replies]

    def summary(self) -> str:
        """A compact, printable view: pass it as an assertion message (a ``str`` is never truncated)."""
        return repr({
            "refusals": [(r.reason, r.where) for r in self.refusals], "errors": [repr(e) for e in self.errors],
            "turns": [(t.profile, t.message, t.home.name) for t in self.turns],
            "handed": [(h.bot, getattr(h.message_type, "name", h.message_type), h.text, h.profile) for h in self.handed],
            "replies": self.replies, "commands": [(c, p) for c, p, _ in self.commands]})


def ensure_discord_mock() -> None:
    """Install a mock ``discord`` module when discord.py isn't available (call before importing the adapter)."""
    import sys
    from unittest.mock import MagicMock

    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


def write_config(home: Path, doc: dict) -> None:
    """Write *home*'s config.yaml with a distinct, settled mtime (two rewrites never share a stat)."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    stamp = time.time_ns() - 10_000_000_000 + next(_WRITES) * 1_000_000
    os.utime(path, ns=(stamp, stamp))


class _TextChannel:
    def __init__(self, history=()):
        self.id, self.name, self.topic = CHANNEL_ID, "general", None
        self.guild = SimpleNamespace(name="Hermes Server", id=GUILD_ID)
        self._history = list(history)

    def history(self, *, limit, before, after=None, oldest_first=None):
        items = list(self._history)

        async def _iter():
            for text in items:
                yield SimpleNamespace(id=100, content=text, clean_content=text, attachments=[], type=_discord().MessageType.default,
                                      author=SimpleNamespace(id=43, display_name="Ann", name="Ann", bot=False))
        return _iter()


def _discord():
    import plugins.platforms.discord.adapter as discord_platform
    return discord_platform.discord


class DiscordGatewayRig:
    """A multiplexed gateway serving ``default`` plus *profiles*, with one Discord bot per entry in
    *bots* (``default`` always owns one). Build it with :func:`discord_gateway_rig`."""

    def __init__(self, root: Path, monkeypatch):
        self.root, self.monkeypatch = root, monkeypatch
        self.runner: Any = None
        self.bots: dict[str, Any] = {}
        self.homes: dict[str, Path] = {}
        self._outcome = Outcome()
        self._handles: list[Any] = []
        self._hold: Optional[asyncio.Event] = None

    # --- what tests call -------------------------------------------------------------------------

    def home(self, profile: Optional[str]) -> Path:
        """The runtime home a turn for *profile* must run in (``None``: the channel's own, ``default``)."""
        return self.homes[profile or "default"]

    def message(self, content: str, *, attachments=(), channel_history=(), message_id: Optional[int] = None):
        """A Discord message as discord.py hands it to ``on_message`` (``mentions`` for ``<@999>`` is set)."""
        bot_ids = [bot._client.user.id for bot in self.bots.values()]
        return SimpleNamespace(
            id=message_id if message_id is not None else next(_MESSAGE_IDS), content=content,
            mentions=[SimpleNamespace(id=i) for i in bot_ids if f"<@{i}>" in content],
            attachments=list(attachments), reference=None, created_at=datetime.now(UTC),
            channel=_TextChannel(channel_history),
            author=SimpleNamespace(id=USER_ID, display_name="Jezza", name="Jezza", bot=False),
            guild=SimpleNamespace(id=GUILD_ID, name="Hermes Server"), type=_discord().MessageType.default)

    async def send(self, content: str, *, to=None, attachments=(), channel_history=(), message=None,
                   wait: bool = True, into: Optional[Outcome] = None, timeout: float = 15) -> Outcome:
        """Deliver one Discord message to the bots in *to* (default: every bot, in order, each with its
        own copy as Discord does) and wait until the gateway has settled. Returns what it caused.

        ``wait=False`` returns at once (the outcome keeps filling in; call ``settle`` later), and
        ``into`` adds to an earlier outcome instead of starting a new one: together they send a
        follow-up while a held turn (``hold_turns``) is still running."""
        self._outcome = outcome = into if into is not None else Outcome()
        message = message or self.message(content, attachments=attachments, channel_history=channel_history)
        for name in (to or list(self.bots)):
            await self.bots[name]._handle_message(_copy(message))
        if wait:
            await self.settle(timeout=timeout)
        return outcome

    async def recover(self, content: str, *, bot: str = "default", timeout: float = 15) -> Outcome:
        """Run *content* through *bot*'s missed-message recovery (``_dispatch_recovered_message``, the
        path a message takes when it arrived while the bot was offline) and wait until it settled."""
        self._outcome = outcome = Outcome()
        await self.bots[bot]._dispatch_recovered_message(self.message(content))
        await self.settle(timeout=timeout)
        return outcome

    def hold_turns(self) -> asyncio.Event:
        """Make every model turn wait (after it is recorded in ``turns``) until the returned event is set."""
        self._hold = asyncio.Event()
        return self._hold

    async def hand_over(self, event, *, bot: str = "default", timeout: float = 15) -> Outcome:
        """Hand an already-built ``MessageEvent`` to the gateway as *bot* would (skips the adapter's own
        gates; for proving the rig can reach a state at all). Returns what it caused."""
        self._outcome = outcome = Outcome()
        await self.bots[bot].handle_message(event)
        await self.settle(timeout=timeout)
        return outcome

    async def settle(self, *, timeout: float = 15, quiet: float = 0.3) -> None:
        """Wait until every bot's background work is done and nothing new happened for *quiet* seconds."""
        deadline, last, still_since = time.monotonic() + timeout, None, time.monotonic()
        while time.monotonic() < deadline:
            await asyncio.sleep(0.02)  # health: allow HX013 -- polls the bots' background turns to quiesce
            busy = any(bot._background_tasks or bot._session_tasks for bot in self.bots.values())
            snapshot = self._outcome.summary()
            if busy or snapshot != last:
                last, still_since = snapshot, time.monotonic()
            elif time.monotonic() - still_since >= quiet:
                return
        raise AssertionError(f"the gateway did not settle within {timeout}s: {self._outcome.summary()}")

    # --- wiring ------------------------------------------------------------------------------------

    def _bot(self, name: str, token: str, bot_id: int, extra: dict):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import SendResult
        from plugins.platforms.discord.adapter import DiscordAdapter

        bot = DiscordAdapter(PlatformConfig(enabled=True, token=token, typing_indicator=False, extra=dict(extra)))
        bot._client = SimpleNamespace(user=SimpleNamespace(id=bot_id))
        bot._text_batch_delay_seconds = 0
        bot.gateway_runner = self.runner

        async def send(chat_id, content, reply_to=None, metadata=None, **_):
            text = str(content)
            if not any(marker in text for marker in _TIP_MARKERS):
                self._outcome.replies.append((name, text))
            return SendResult(success=True, message_id=f"s{len(self._outcome.replies)}")
        self.monkeypatch.setattr(bot, "send", send)
        real_handle = bot.handle_message

        async def handle_message(event):
            source = event.source
            self._outcome.handed.append(Handed(name, event.message_type, event.text, source.profile,
                                               bool(getattr(source, "profile_mention", False))))
            return await real_handle(event)
        self.monkeypatch.setattr(bot, "handle_message", handle_message)
        return bot

    def _spy_gateway(self) -> None:
        import traceback

        from hermes_constants import get_hermes_home
        from persistence.core.errors import AuthorityRefusal

        runner, real_init = self.runner, AuthorityRefusal.__init__

        def init(exc, reason, *args, **kwargs):
            # A refusal the GATEWAY decides (``gateway/``), handled later or not. A store-level answer
            # its caller branches on (``persistence/``: "no local receipt", ...) is not one.
            real_init(exc, reason, *args, **kwargs)
            stack = traceback.extract_stack()[:-1]
            if "/gateway/" in stack[-1].filename:
                chain = tuple(f.name for f in stack if "/gateway/" in f.filename)
                self._outcome.refusals.append(Refusal(str(reason), stack[-1].name, chain[-6:]))
        self.monkeypatch.setattr(AuthorityRefusal, "__init__", init)

        real_handle = runner._handle_message

        async def handle_message(event):
            try:
                return await real_handle(event)
            except Exception as exc:  # recorded, then re-raised: the adapter's own handling still runs
                self._outcome.errors.append(exc)
                raise
        self.monkeypatch.setattr(runner, "_handle_message", handle_message)

        async def run_agent(**kwargs):
            source = kwargs["source"]
            self._outcome.turns.append(Turn(
                getattr(source, "profile", None), Path(get_hermes_home()).resolve(), kwargs.get("session_key"),
                kwargs.get("session_id"), kwargs.get("message"), kwargs.get("context_prompt")))
            if self._hold is not None:
                await self._hold.wait()
            return {"final_response": "ok", "messages": [], "tools": [], "history_offset": 0, "last_prompt_tokens": 0}
        self.monkeypatch.setattr(runner, "_run_agent", run_agent)

        real_reset = runner._handle_reset_command

        async def reset(event):
            self._outcome.commands.append(("reset", event.source.profile, Path(get_hermes_home()).resolve()))
            return await real_reset(event)
        self.monkeypatch.setattr(runner, "_handle_reset_command", reset)


def _copy(message):
    """Each bot gets its own copy of the gateway event, as Discord delivers it to every bot."""
    clone = SimpleNamespace(**vars(message))
    clone.attachments = list(message.attachments)
    return clone


def text_attachment(rig: DiscordGatewayRig, body: str, filename: str = "notes.md"):
    """A text attachment every bot downloads and inlines into the event text, as a real ``.md`` upload."""
    from unittest.mock import AsyncMock

    import plugins.platforms.discord.adapter as discord_platform
    for bot in rig.bots.values():
        rig.monkeypatch.setattr(bot, "_cache_discord_document", AsyncMock(return_value=body.encode("utf-8")))
    cached = rig.root / "cache" / f"doc_{filename}"  # the gateway snapshots the cached file at admission
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(body, encoding="utf-8")
    rig.monkeypatch.setattr(discord_platform, "cache_document_from_bytes_async", AsyncMock(return_value=str(cached)))
    return SimpleNamespace(content_type="text/markdown", filename=filename, size=len(body), url="u")


@contextlib.asynccontextmanager
async def discord_gateway_rig(tmp_path: Path, monkeypatch, *, profiles: dict[str, Optional[dict]],
                              bots: tuple[str, ...] = ("default",), default_discord: Optional[dict] = None,
                              env: Optional[dict[str, Optional[str]]] = None):
    """A running multiplexed gateway (see the module docstring). *profiles* maps each served profile
    besides ``default`` to its own ``discord:`` config block (``None``: no block). *bots* names the
    profiles that own a Discord bot (``default`` first). *env* overrides the Discord gate variables
    (``None`` deletes one); the defaults are a free-response channel (``DISCORD_REQUIRE_MENTION=false``),
    no auto-thread and no history backfill, with user 42 allowed."""
    from agent.secret_scope import set_multiplex_active
    from gateway.config import GatewayConfig, Platform, load_gateway_config
    from gateway.run import GatewayRunner
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from persistence_assembly import open_persistence, profile_scope

    root = (tmp_path / "hh").resolve()
    rig = DiscordGatewayRig(root, monkeypatch)
    rig.homes = {"default": root, **{name: root / "profiles" / name for name in profiles}}
    configs = {}
    for name, discord in {"default": default_discord, **profiles}.items():
        write_config(rig.homes[name], {"model": {"default": "m"}, **({"discord": discord} if discord is not None else {})})
        with monkeypatch.context() as m:
            m.setenv("HERMES_HOME", str(rig.homes[name]))
            configs[name] = load_gateway_config()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(_discord(), "DMChannel", type("DMChannel", (), {}), raising=False)
    monkeypatch.setattr(_discord(), "Thread", type("Thread", (), {}), raising=False)
    gates = {"DISCORD_REQUIRE_MENTION": "false", "DISCORD_AUTO_THREAD": "false", "DISCORD_HISTORY_BACKFILL": "false",
             "DISCORD_ALLOWED_USERS": str(USER_ID), "DISCORD_FREE_RESPONSE_CHANNELS": None,
             "DISCORD_ALLOWED_CHANNELS": None, "DISCORD_IGNORED_CHANNELS": None, **(env or {})}
    for var, value in gates.items():
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    served = [(name, home) for name, home in rig.homes.items()]
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda *a, **k: list(served))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: rig.homes.get(n, root / "profiles" / n))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda n: n in rig.homes)
    set_multiplex_active(True)  # the real process latch (the conftest resets it)

    config = configs["default"]
    config.multiplex_profiles = True
    runner = rig.runner = GatewayRunner(config)
    runner.config._runtime_profile_homes = tuple(served)  # the server's reservation: what _binding trusts
    runner._served_profile_homes = dict(served)
    runner._profile_configs = {name: cfg for name, cfg in configs.items() if name != "default"}
    runner._primary_profile_name = "default"
    authorities = []
    for name, home in served:
        handle = open_persistence(profile_scope(home=home))
        rig._handles.append(handle)
        authorities.append((name, home, await initialize_session_authority(
            runner, profile_id=str(home), instance_id="owner", persistence=handle, register=name == "default")))
    runner.session_authorities = SessionAuthorities(root, multiplexed=True)
    for name, home, authority in authorities:
        runner.session_authorities.add(home, authority, **({} if name == "default" else {"name": name}))
    monkeypatch.setattr(runner, "_read_user_config", lambda: {"approvals": {"destructive_slash_confirm": False}})
    rig._spy_gateway()

    runner.adapters, runner._profile_adapters = {}, {name: {} for name in profiles}
    for index, name in enumerate(bots):
        extra = (configs[name].platforms.get(Platform.DISCORD).extra if Platform.DISCORD in configs[name].platforms else {})
        bot = rig.bots[name] = rig._bot(name, f"{name}-token", 999 - index, extra or {})
        if name == "default":
            runner.adapters[Platform.DISCORD] = bot
            runner._wire_adapter_handlers(bot)
        else:  # a secondary bot authorizes against its OWN profile's .env allowlist
            (rig.homes[name] / ".env").write_text(f"DISCORD_ALLOWED_USERS={USER_ID}\n", encoding="utf-8")
            runner._profile_adapters[name] = {Platform.DISCORD: bot}
            runner._configure_profile_adapter(bot, name, Platform.DISCORD)
    try:
        yield rig
    finally:
        with contextlib.suppress(Exception):
            runner.session_store.close_all_db_handles()
        for handle in rig._handles:
            with contextlib.suppress(Exception):
                handle.close()
