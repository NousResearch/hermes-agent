"""A running gateway starts, restarts and stops ONE platform adapter when the profile's platform config
changes, without a gateway restart, single-profile gateways included (no multiplexer involved).

Live: a real ``hermes gateway run`` on a disposable root; the webhook adapter binds a real loopback-
reachable port, so "started" is a socket that accepts and "stopped" is one that refuses.
"""
import asyncio
import os
import socket
import types
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from tests.gateway.test_multiplex_hot_serve_live import control, start_daemon, stop, tail
from tests.waiting import wait_until


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _reconcile(root, params=None):
    """Ask the gateway to apply the files just written. An apply past the verb's window answers ``pending``
    and goes on; asking again answers that same apply's receipt once it settles (the sync point: a loaded
    host takes longer than the window, never a different outcome)."""
    def _settled():
        answer = control(root, "reconcile-platforms", params, timeout=60)
        return None if answer.get("pending") else answer
    return wait_until(_settled, timeout=180, interval=0.1, what="the reconcile to settle")


def _accepts(port):
    with socket.socket() as peer:
        peer.settimeout(1)
        return peer.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_single_profile_gateway_starts_restarts_and_stops_a_platform_live(tmp_path):
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first, second = _free_port(), _free_port()
        dotenv = root / ".env"
        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={first}\n", encoding="utf-8")
        answer = _reconcile(root)
        assert answer["applied"] is True and answer["started"] == ["webhook"], (answer, tail(mux))
        assert answer["platforms"]["webhook"] == "connected"
        wait_until(lambda: _accepts(first), timeout=10, interval=0.1, what="webhook listening")

        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = _reconcile(root)
        assert answer["restarted"] == ["webhook"] and answer["platforms"]["webhook"] == "connected", answer
        wait_until(lambda: _accepts(second) and not _accepts(first), timeout=10, interval=0.1,
                   what="webhook moved to the new port")

        dotenv.write_text(f"WEBHOOK_ENABLED=false\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = _reconcile(root)
        assert answer["stopped"] == ["webhook"], answer
        wait_until(lambda: not _accepts(second), timeout=10, interval=0.1, what="webhook stopped")
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_a_platform_configured_by_the_launch_environment_survives_a_reconcile(tmp_path):
    """Docker ``-e`` / a unit's ``Environment=`` configures a platform the ``.env`` never names. A reconcile
    (a plugin loading, any ``.env`` save) re-read the ``.env`` the way ``/reload`` does, which drops every
    known key the file lacks: the launch-configured platform read as unwanted and was stopped."""
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n", encoding="utf-8")
    port = _free_port()
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1",
               WEBHOOK_ENABLED="true", WEBHOOK_PORT=str(port))
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        wait_until(lambda: _accepts(port), timeout=15, interval=0.1, what="webhook listening")
        (root / ".env").write_text("SOME_UNRELATED_SETTING=1\n", encoding="utf-8")
        answer = _reconcile(root)
        assert answer["stopped"] == [] and answer["platforms"].get("webhook") == "connected", (answer, tail(mux))
        assert _accepts(port), tail(mux)
    finally:
        stop(proc, expect=None)


def _closed_port():
    port = _free_port()  # nothing listens: an outbound API call through it is refused at once
    return f"http://127.0.0.1:{port}"


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_named_profile_restart_with_the_same_token_comes_back_up_live(tmp_path):
    """A multiplexed named profile's settings-only change (same channel token, new secret) restarts its
    adapter on the running gateway: the rebuild is not refused as a duplicate of the adapter it replaces."""
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    named = root / "profiles" / "b"
    named.mkdir(parents=True, mode=0o700)
    # ``config.token`` gives the adapter a credential claim; the .env pair is what the adapter reads.
    (named / "config.yaml").write_text("platforms:\n  line:\n    token: same-token\n", encoding="utf-8")
    dotenv = named / ".env"
    dotenv.write_text("LINE_CHANNEL_ACCESS_TOKEN=same-token\nLINE_CHANNEL_SECRET=first\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    closed = _closed_port()
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1",
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"), HTTPS_PROXY=closed, HTTP_PROXY=closed,
               https_proxy=closed, http_proxy=closed, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first = _reconcile(root, {"home": str(named)})
        assert first["profile"] == "b" and first["platforms"].get("line") == "connected", (first, tail(mux))
        dotenv.write_text("LINE_CHANNEL_ACCESS_TOKEN=same-token\nLINE_CHANNEL_SECRET=second\n", encoding="utf-8")
        answer = _reconcile(root, {"home": str(named)})
        assert answer["restarted"] == ["line"], (answer, tail(mux))
        assert answer["platforms"]["line"] == "connected", (answer, tail(mux))
        assert "with the same credential" not in tail(mux)
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_a_named_profile_rescan_leaves_its_restart_for_the_next_answer(tmp_path):
    """``rescan-profiles`` applies a named profile's new secret (restarts its adapter); the next
    ``reconcile-platforms`` for it reports that restart, and the one after reports nothing new."""
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    named = root / "profiles" / "b"
    named.mkdir(parents=True, mode=0o700)
    (named / "config.yaml").write_text("platforms:\n  line:\n    token: same-token\n", encoding="utf-8")
    dotenv = named / ".env"
    dotenv.write_text("LINE_CHANNEL_ACCESS_TOKEN=same-token\nLINE_CHANNEL_SECRET=first\n", encoding="utf-8")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    closed = _closed_port()
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               PYTHONUNBUFFERED="1", HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"), HTTPS_PROXY=closed,
               HTTP_PROXY=closed, https_proxy=closed, http_proxy=closed, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first = _reconcile(root, {"home": str(named)})
        assert first["platforms"].get("line") == "connected", (first, tail(mux))
        dotenv.write_text("LINE_CHANNEL_ACCESS_TOKEN=same-token\nLINE_CHANNEL_SECRET=second\n", encoding="utf-8")
        scan = control(root, "rescan-profiles", timeout=30)
        # A loaded host outlasts the rescan's 5 s window: it answers ``pending`` and applies on; either
        # way the restart is the next answer's.
        assert scan.get("pending") is True or "b" in scan["rescanned"], (scan, tail(mux))
        answer = _reconcile(root, {"home": str(named)})
        assert answer["restarted"] == ["line"] and answer["changed"] is True, (answer, tail(mux))
        assert answer["platforms"]["line"] == "connected", (answer, tail(mux))
        again = _reconcile(root, {"home": str(named)})
        assert again["restarted"] == [] and again["changed"] is False, (again, tail(mux))
    finally:
        stop(proc, expect=None)


def _runner(adapter):
    """Just the state the retire path reads; the retire logic under test is the real mixin's (the
    named-profile reconcile included: it is the platform reconcile's subclass)."""
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    class _Runner(_RunnerState, GatewayProfileReconcileMixin):
        pass

    return _Runner(adapter)


class _RunnerState:
    def __init__(self, adapter):
        self._running = True
        self.adapters = {Platform.TELEGRAM: adapter}
        self._failed_platforms = {}
        self.turns = {"agent:main:telegram:dm:42": object(), "agent:main:discord:dm:7": object()}
        self.torn_down_with = None
        self.status = {}
        self.sources = {}
        self._lock = asyncio.Lock()

    def _snapshot_running_agents(self):
        return dict(self.turns)

    def _get_cached_session_source(self, key):
        return self.sources.get(key)

    def _delivery_adapter_for(self, source):
        return source.answered_by

    async def _bounded_adapter_teardown(self, adapter, platform, *, profile=None):
        self.torn_down_with = dict(self.turns)

    def _update_platform_runtime_status(self, key, **fields):
        self.status[key] = fields

    def _reconcile_lock(self):
        return self._lock

    def _retain_background_task(self, task):
        pass


def test_disabled_platform_finishes_its_in_flight_turn_before_it_goes_down(monkeypatch):
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 30.0)
    runner = _runner(types.SimpleNamespace())

    async def scenario():
        retire = asyncio.ensure_future(runner._retire_primary_platform(Platform.TELEGRAM))
        await asyncio.sleep(1.2)  # health: allow HX013 -- still waiting after two poll intervals IS the behaviour
        assert not retire.done() and Platform.TELEGRAM in runner.adapters  # still delivering its turn
        runner.turns.pop("agent:main:telegram:dm:42")  # the turn finished; another platform's runs on
        await asyncio.wait_for(retire, timeout=5)

    asyncio.run(scenario())
    assert runner.torn_down_with == {"agent:main:discord:dm:7": runner.turns["agent:main:discord:dm:7"]}
    assert Platform.TELEGRAM not in runner.adapters
    assert runner.status["telegram"]["platform_state"] == "disabled"


def test_drain_waits_for_a_turn_this_bot_received_for_another_profile(monkeypatch):
    """A routed turn runs in another profile's namespace, yet this bot answers it: retiring waits."""
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 30.0)
    bot = types.SimpleNamespace()
    runner = _runner(bot)
    runner.turns = {"agent:other:discord:dm:9": object()}  # session key names neither this platform nor profile
    runner.sources = {"agent:other:discord:dm:9": types.SimpleNamespace(answered_by=bot)}

    async def scenario():
        retire = asyncio.ensure_future(runner._retire_primary_platform(Platform.TELEGRAM))
        await asyncio.sleep(1.2)  # health: allow HX013 -- still waiting after two poll intervals IS the behaviour
        assert not retire.done()
        runner.turns.clear()
        await asyncio.wait_for(retire, timeout=5)

    asyncio.run(scenario())
    assert runner.torn_down_with == {}


def test_build_print_tracks_env_only_credentials_and_typed_settings(tmp_path):
    """Slack reads SLACK_APP_TOKEN straight from the profile's .env and Telegram caches reply_to_mode:
    either change is a rebuild. Runner-read settings (home channel, allowlists) are not."""
    from gateway.config import PlatformConfig
    from gateway.run_platform_reconcile import platform_build_print
    env = tmp_path / ".env"

    def print_of(text, **settings):
        env.write_text(text, encoding="utf-8")
        from agent.secret_scope import invalidate_env_file_cache
        invalidate_env_file_cache(env)
        return platform_build_print(Platform.SLACK, PlatformConfig(enabled=True, token="xoxb-1", **settings), tmp_path)

    base = print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\n")
    assert print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-2\n").differs(base)
    assert print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\n", reply_to_mode="all").differs(base)
    assert not print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nOTHER_KEY=x\n").differs(base)


def test_only_adapters_a_build_attempt_owns_are_stopped():
    """The LOCAL session adapter is installed at runtime, never built from config: a reconcile whose
    config does not list it must leave it alone (it carries the app's live sessions)."""
    from gateway.run_platform_reconcile import BuildPrint, plan_platform_changes
    webhook = BuildPrint("w", "e")
    plan = plan_platform_changes({}, {Platform.WEBHOOK: webhook}, {Platform.WEBHOOK, Platform.LOCAL})
    assert plan.stop == (Platform.WEBHOOK,) and Platform.LOCAL not in (*plan.stop, *plan.restart)
    unchanged = plan_platform_changes({Platform.WEBHOOK: webhook}, {Platform.WEBHOOK: webhook}, {Platform.LOCAL})
    assert not unchanged and not unchanged.start  # attempted at boot and failed: the boot verdict stands


def test_secondary_restart_is_not_refused_as_a_duplicate_of_itself(monkeypatch):
    """A settings-only restart keeps the same token: the retired adapter's claim must leave the rescan's
    ``claimed`` map, or the rebuild is parked as a duplicate of the adapter it replaces."""
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 0.0)
    adapter = types.SimpleNamespace()
    runner = _runner(adapter)
    runner._profile_adapters = {"b": {Platform.TELEGRAM: adapter}}
    runner._profile_failed_platforms = {}
    claim = ("telegram", "token-hash")
    runner._adapter_claims = lambda platform, a: (claim, None)
    runner._hold_secondary_claims = lambda *a: None
    runner._release_secondary_claims = lambda *a: None
    claimed = {claim: "b", ("discord", "other"): "c"}
    asyncio.run(runner._retire_secondary_platform("b", Path("."), Platform.TELEGRAM, {}, claimed))
    assert claimed == {("discord", "other"): "c"}
    assert Platform.TELEGRAM not in runner._profile_adapters["b"]


@pytest.mark.asyncio
async def test_a_reconnect_superseded_while_connecting_never_publishes(monkeypatch):
    """A reconcile retires a queued platform (and may rebuild it) while the watcher's older attempt is
    still connecting with the old config: that attempt's adapter is discarded, never published."""
    from unittest.mock import AsyncMock
    from tests.gateway.test_platform_reconnect import StubAdapter, _make_runner
    runner = _make_runner()
    runner._update_platform_runtime_status = lambda *a, **k: None
    runner._install_reconnected_adapter = AsyncMock()
    runner._wire_adapter_handlers = lambda adapter: None
    stale = StubAdapter()
    stale.disconnect = AsyncMock()
    replacement = {"config": PlatformConfig(enabled=True, token="new"), "attempts": 0, "next_retry": 0}

    async def connect_then_superseded(adapter, platform, *, is_reconnect):
        runner._failed_platforms[platform] = replacement  # the reconcile's new slot for this platform
        return True

    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: stale)
    monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect_then_superseded)
    runner._failed_platforms[Platform.TELEGRAM] = {
        "config": PlatformConfig(enabled=True, token="old"), "attempts": 0, "next_retry": 0}
    await runner._reconnect_failed_platform(Platform.TELEGRAM, 0)
    runner._install_reconnected_adapter.assert_not_awaited()
    stale.disconnect.assert_awaited()
    assert runner._failed_platforms[Platform.TELEGRAM] is replacement and replacement["attempts"] == 0


def test_a_verb_answer_past_its_window_names_what_is_still_starting(monkeypatch):
    """A connect longer than the verb window answers ``pending`` with the in-flight plan, so a caller
    can say which platform is still connecting instead of printing nothing."""
    import threading
    from gateway import run_platform_reconcile as reconcile
    monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 0.2)
    runner = _runner(types.SimpleNamespace())
    runner._primary_platform_home = lambda: Path("primary-home")
    release = asyncio.Event()

    planned = threading.Event()

    async def slow_apply(name):
        runner._primary_platform_plan = reconcile.PlatformPlan((Platform.TELEGRAM,), (), ())
        planned.set()
        await release.wait()
        return {}

    runner.answer_platform_reconcile = slow_apply
    in_flight = runner.platform_plan_in_flight

    def plan_once_planned(name):  # the window may close before a loaded loop thread runs the apply at all
        assert planned.wait(30), "the apply never started"
        return in_flight(name)
    runner.platform_plan_in_flight = plan_once_planned
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        answer = reconcile.reconcile_platforms_verb(runner, loop)({"home": "primary-home"})
    finally:
        loop.call_soon_threadsafe(release.set)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    assert answer == {"applied": True, "pending": True, "starting": ["telegram"], "restarting": [], "stopping": [],
                      "installing": False}


class _FileDrivenPrimary:
    """A launch profile whose files drive the REAL reconcile (plan, signatures, the verb): ``.env`` line
    ``START_<platform>=1`` wants that platform, ``INSTALL_<name>=1`` makes the plugin recovery install
    ``name`` and write its marker to ``config.yaml`` (as the catalog recovery does). The adapter build is a
    stand-in that records what was really started and lets a test hold an apply mid-way.

    With ``monkeypatch``, a multiplexed named profile ``b`` is served beside it, its files driving the
    REAL named-profile paths (profile watcher tick, ``rescan-profiles``, a plugin-load trigger, the verb):
    the same ``START_``/``INSTALL_`` lines, plus ``SECRET=<v>`` (a new value restarts what runs)."""

    def __init__(self, home, loop, rng=None, monkeypatch=None):
        import threading
        from gateway import run_platform_reconcile as reconcile
        self.home, self.loop, self.rng = home, loop, rng
        self.started, self.installed = [], []
        self.named_started, self.named_installed, self.named_restarted = [], [], []
        self.release = threading.Event()
        self.release.set()
        self.mid_apply = None  # called (off-loop) while an apply is building its adapters
        self.states, self.named_states = {}, {}
        runner = self.runner = _runner(types.SimpleNamespace())
        runner.adapters, runner.config = {}, types.SimpleNamespace(platforms={})
        runner._primary_platform_home = lambda: home
        runner._primary_platform_states = lambda: dict(self.states)
        runner._run_housekeeping_in_executor = lambda fn, *a: asyncio.to_thread(fn, *a)
        runner._recover_primary_plugins = self._recover
        runner._read_primary_platforms = self._read
        runner._retire_primary_platform = self._retire
        runner._hot_start_primary_platforms = self._hot_start
        (home / "config.yaml").write_text("platforms: {}\n", encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
        if monkeypatch is not None:
            self._serve_named(monkeypatch)
        asyncio.run_coroutine_threadsafe(runner._note_boot_platform_signature(), loop).result(30)
        self.ask = reconcile.reconcile_platforms_verb(runner, loop)

    def _serve_named(self, monkeypatch):
        self.named = self.home / "profiles" / "b"
        self.named.mkdir(parents=True)
        (self.named / "config.yaml").write_text("platforms: {}\n", encoding="utf-8")
        (self.named / ".env").write_text("", encoding="utf-8")
        self.named_built = {}
        runner, homes = self.runner, [("default", self.home), ("b", self.named)]
        monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda *a, **k: list(homes))
        runner._primary_profile_name = "default"
        runner._multiplex_on = lambda: True
        runner._profile_adapters, runner._profile_failed_platforms = {"b": {}}, {}
        runner._parked_profile_names = list
        runner._drop_stale_platform_verdicts = lambda name: None
        runner._live_resource_claims = lambda active: {}
        runner._record_served_profiles = lambda active, items: runner._note_served_profiles(items)
        runner._secondary_platform_states = lambda name: dict(self.named_states)
        runner._start_one_profile_adapters = self._start_named
        runner._unserve_profile = self._unserve_named
        monkeypatch.setattr("gateway.run_runtime.park_profile", lambda runner_, name, why: self.parked.append(name))
        self.parked = []
        runner._note_served_profiles(homes)

    def edit(self, line, profile=None):  # whole-file replace, as config writers do: a racing read sees before or after
        self._append(".env", [line], profile)

    def _dir(self, profile):
        return self.named if profile == "b" else self.home

    def _append(self, name, lines, profile=None):
        path, staged = self._dir(profile) / name, self._dir(profile) / f".{name}.new"
        staged.write_text(path.read_text(encoding="utf-8") + "".join(f"{ln}\n" for ln in lines), encoding="utf-8")
        os.replace(staged, path)

    def _lines(self, name, prefix, profile=None):
        text = (self._dir(profile) / name).read_text(encoding="utf-8")
        return [ln.split("=")[0][len(prefix):] for ln in text.splitlines() if ln.startswith(prefix)]

    def _new_installs(self, profile=None):
        done = set(self._lines("config.yaml", "# installed ", profile))
        new = [n for n in self._lines(".env", "INSTALL_", profile) if n not in done]
        if new:
            self._append("config.yaml", [f"# installed {n}" for n in new], profile)
        return new

    def _recover(self, home):
        new = self._new_installs()
        self.installed += new
        return new

    def _read(self, home, installed):
        from gateway.run_platform_reconcile import BuildPrint
        wanted = {Platform(v.lower()): (PlatformConfig(enabled=True), BuildPrint("v1"))
                  for v in self._lines(".env", "START_")}
        return types.SimpleNamespace(platforms={p: c for p, (c, _bp) in wanted.items()}), wanted, {}, installed

    async def _retire(self, platform):
        self.runner.adapters.pop(platform, None)

    async def _interleave(self):
        await asyncio.to_thread(self.release.wait, 60)
        if self.mid_apply is not None:
            await asyncio.to_thread(self.mid_apply)
        for _ in range(self.rng.randrange(8) if self.rng is not None else 0):
            await asyncio.sleep(0)  # let the racing reconciles and asks interleave differently per seed

    async def _hot_start(self, items):
        await self._interleave()
        for platform, _config, build_print in items:
            self.runner._primary_build_prints[platform] = build_print
            self.runner.adapters[platform] = object()
            self.states[platform.value] = "retrying"
            self.started.append(platform.value)

    async def _unserve_named(self, name, home):
        self.runner._profile_adapters.pop(name, None)
        self.named_states.clear()

    async def _start_named(self, name, home, claimed):
        """Stand-in for the named profile's adapter build (its recovery, retire, rebuild), keeping the
        contract the real one keeps: the recovery's installs are noted, the plan is published, and a
        config the multiplexer refuses (``OPEN_POLICY=1``) raises before anything is built."""
        from gateway.run import MultiplexConfigError
        from gateway.run_platform_reconcile import BuildPrint, plan_platform_changes
        if self._lines(".env", "OPEN_POLICY", "b"):
            raise MultiplexConfigError("open dm policy")
        runner, live = self.runner, self.runner._profile_adapters.setdefault(name, {})
        installed = self._new_installs("b")
        if installed:
            runner._note_profile_installs(name, installed)
            self.named_installed += installed
        text = (self.named / ".env").read_text(encoding="utf-8")
        secret = ([ln for ln in text.splitlines() if ln.startswith("SECRET=")] or [""])[-1]
        wanted = {Platform(v.lower()): BuildPrint("v1", secret) for v in self._lines(".env", "START_", "b")}
        plan = plan_platform_changes(wanted, self.named_built, set(live))
        runner._profile_platform_plans = getattr(runner, "_profile_platform_plans", None) or {}
        runner._profile_platform_plans[name] = plan
        for platform in (*plan.stop, *plan.restart):
            live.pop(platform, None)
            self.named_built.pop(platform, None)
            self.named_states.pop(platform.value, None)
        await self._interleave()
        for platform in (*plan.restart, *plan.start):
            live[platform], self.named_built[platform] = object(), wanted[platform]
            self.named_states[platform.value] = "connected"
            (self.named_restarted if platform in plan.restart else self.named_started).append(platform.value)
        return len(live)

    def background(self, reason):  # a plugin-load trigger, or a watcher tick
        coro = (self.runner._reconcile_primary_platforms_if_changed(reason) if reason == "watcher"
                else self.runner.reconcile_profile_platforms(None, reason=reason))
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def named_background(self, op):
        """The named profile's paths: the profile watcher's tick, ``rescan-profiles``, a plugin load."""
        coro = {"scan_b": lambda: self.runner.reconcile_served_profiles(reason="watcher"),
                "rescan_b": lambda: self.runner.reconcile_served_profiles(reason="control-socket"),
                "plugin_b": lambda: self.runner.reconcile_profile_platforms("b", reason="plugin-loaded")}[op]()
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def ask_named(self):
        return self.ask({"home": str(self.named)})

    def settled_ask(self, ask=None):
        ask = ask or (lambda: self.ask({}))

        def _settled():
            answer = ask()
            return None if answer.get("pending") else answer
        return wait_until(_settled, timeout=60, interval=0.05, what="the ask to settle")


@pytest.fixture
def reconcile_loop():
    import threading
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _assert_reported_once(gw, answers):
    """Every platform a reconcile really started, and every plugin it installed, is in exactly one answer
    for its profile; every restart is reported, never more often than it happened."""
    applied = {"default": {"started": gw.started, "installed": gw.installed},
               "b": {"started": gw.named_started, "installed": gw.named_installed}}
    for profile, lists in applied.items():
        mine = [answer for answer in answers if answer["profile"] == profile]
        for key, done in lists.items():
            reported = [item for answer in mine for item in answer[key]]
            assert sorted(reported) == sorted(done), (profile, key, answers)
    restarted = [item for answer in answers if answer["profile"] == "b" for item in answer["restarted"]]
    assert set(restarted) == set(gw.named_restarted), answers
    assert all(restarted.count(p) <= gw.named_restarted.count(p) for p in restarted), answers


_FRESH = [p.value.upper() for p in (Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK, Platform.WEBHOOK,
                                    Platform.MATRIX, Platform.MATTERMOST, Platform.SIGNAL, Platform.EMAIL)]


@pytest.mark.parametrize("script", [
    # A plugin-triggered reconcile applies the edit; a second trigger (another plugin) finds nothing to do.
    "edit plugin plugin ask",
    "edit plugin watcher ask",
    # The plugin recovery writes its install marker to config.yaml during the apply: not a new edit.
    "install plugin watcher ask",
    # An ask past its window answers pending; asking again answers what that apply did.
    "install pending release ask",
    "edit pending plugin& release ask ask",
    # A separate edit landing while an apply builds its adapters is reconciled once more.
    "install midedit plugin watcher ask",
])
def test_every_applied_change_is_reported_to_exactly_one_ask(tmp_path, monkeypatch, reconcile_loop, script):
    from gateway import run_platform_reconcile as reconcile
    gw, answers, fresh, running = _FileDrivenPrimary(tmp_path, reconcile_loop), [], iter(_FRESH), []
    steps = {
        "plugin&": lambda: running.append(gw.background("plugin")),
        "edit": lambda: gw.edit(f"START_{next(fresh)}=1"),
        "install": lambda: (gw.edit(f"INSTALL_{len(gw.installed)}x=1"), gw.edit(f"START_{next(fresh)}=1")),
        "plugin": lambda: gw.background("plugin").result(60),
        "watcher": lambda: gw.background("watcher").result(60),
        "ask": lambda: answers.append(gw.settled_ask()),
        "release": gw.release.set,
        "midedit": lambda: setattr(gw, "mid_apply", lambda: (gw.edit(f"START_{next(fresh)}=1"),
                                                             setattr(gw, "mid_apply", None))),
    }

    def pending():
        gw.release.clear()
        monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 0.2)
        assert gw.ask({})["pending"] is True
        monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 30.0)
    steps["pending"] = pending
    for step in script.split():
        steps[step]()
    for future in running:
        future.result(60)
    gw.background("watcher").result(60)
    answers.append(gw.settled_ask())
    _assert_reported_once(gw, answers)
    edited = {name.lower() for name in gw._lines(".env", "START_")}
    assert {p.value for p in gw.runner.adapters} == edited == set(gw.started)  # every edit applied, once


@pytest.mark.parametrize("seed", range(12))
def test_racing_reconciles_report_each_change_to_exactly_one_ask(tmp_path, monkeypatch, reconcile_loop, seed):
    """Plugin triggers, watcher ticks and asks (some giving up as ``pending``) racing edits, in any order."""
    import random
    from gateway import run_platform_reconcile as reconcile
    rng = random.Random(seed)
    gw, answers, fresh, running = _FileDrivenPrimary(tmp_path, reconcile_loop, rng), [], iter(_FRESH), []
    for step in range(14):
        op = rng.choice(["edit", "install", "plugin", "watcher", "ask", "pending"])
        if op == "edit" or (op == "install" and rng.random() < 0.5):
            gw.edit(f"START_{next(fresh, 'SMS')}=1")
        elif op == "install":
            gw.edit(f"INSTALL_{seed}n{step}=1")
        elif op in ("plugin", "watcher"):
            running.append(gw.background(op))
        else:
            monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 30.0 if op == "ask" else 0.005)
            answer = gw.ask({})
            if not answer.get("pending"):
                answers.append(answer)
    monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 30.0)
    for future in running:
        future.result(60)
    gw.background("watcher").result(60)
    answers.append(gw.settled_ask())
    _assert_reported_once(gw, answers)
    assert gw.settled_ask()["started"] == []  # taken: asked again about the same files, nothing is new


_NAMED_FRESH = [p.value.upper() for p in (Platform.FEISHU, Platform.DINGTALK, Platform.WECOM, Platform.BLUEBUBBLES,
                                          Platform.QQBOT, Platform.WHATSAPP)]


def _named_steps(gw, monkeypatch, answers, running, fresh):
    """The named profile's ops, beside the launch profile's: edits, a plugin it installs, a new secret,
    and every path that applies it (watcher tick, ``rescan-profiles``, a plugin load, an ask, a pending ask)."""
    from gateway import run_platform_reconcile as reconcile
    secrets = iter(range(1000))

    def pending_b():
        gw.release.clear()
        monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 0.2)
        assert gw.ask_named()["pending"] is True
        monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 30.0)
    return {
        "edit_b": lambda: gw.edit(f"START_{next(fresh, 'WEIXIN')}=1", "b"),
        "install_b": lambda: (gw.edit(f"INSTALL_b{len(gw.named_installed)}x{next(secrets)}=1", "b"),
                              gw.edit(f"START_{next(fresh, 'WEIXIN')}=1", "b")),
        "secret_b": lambda: gw.edit(f"SECRET={next(secrets)}", "b"),
        "scan_b": lambda: gw.named_background("scan_b").result(60),
        "rescan_b": lambda: gw.named_background("rescan_b").result(60),
        "plugin_b": lambda: gw.named_background("plugin_b").result(60),
        "plugin_b&": lambda: running.append(gw.named_background("plugin_b")),
        "rescan_b&": lambda: running.append(gw.named_background("rescan_b")),
        "ask_b": lambda: answers.append(gw.settled_ask(gw.ask_named)),
        "pending_b": pending_b,
    }


def _settle_both(gw, answers, running):
    for future in running:
        future.result(60)
    gw.background("watcher").result(60)
    gw.named_background("scan_b").result(60)
    answers.append(gw.settled_ask())
    answers.append(gw.settled_ask(gw.ask_named))


@pytest.mark.parametrize("script", [
    # The reviewer's case: a rescan applies the new secret; the next ask reports the restart, once.
    "edit_b ask_b secret_b rescan_b ask_b ask_b",
    "edit_b scan_b secret_b scan_b ask_b",
    "install_b rescan_b plugin_b ask_b",
    "edit_b pending_b secret_b release rescan_b ask_b",
    "edit_b plugin_b& secret_b rescan_b& release ask_b",
    # Both profiles at once: each answer carries only its own profile's changes.
    "edit edit_b plugin rescan_b ask ask_b",
    "install install_b pending release rescan_b watcher ask_b ask",
])
def test_every_named_profile_change_is_reported_to_exactly_one_ask(tmp_path, monkeypatch, reconcile_loop, script):
    gw, answers, running = _FileDrivenPrimary(tmp_path, reconcile_loop, monkeypatch=monkeypatch), [], []
    fresh, named_fresh = iter(_FRESH), iter(_NAMED_FRESH)
    steps = _named_steps(gw, monkeypatch, answers, running, named_fresh)
    steps.update({
        "edit": lambda: gw.edit(f"START_{next(fresh)}=1"),
        "install": lambda: (gw.edit(f"INSTALL_{len(gw.installed)}x=1"), gw.edit(f"START_{next(fresh)}=1")),
        "plugin": lambda: gw.background("plugin").result(60),
        "watcher": lambda: gw.background("watcher").result(60),
        "ask": lambda: answers.append(gw.settled_ask()),
        "release": gw.release.set,
    })
    steps["pending"] = lambda: (gw.release.clear(), monkeypatch.setattr(
        "gateway.run_platform_reconcile._VERB_WAIT_SECS", 0.2), answers.append(gw.ask({})),
        monkeypatch.setattr("gateway.run_platform_reconcile._VERB_WAIT_SECS", 30.0))
    for step in script.split():
        steps[step]()
    answers[:] = [answer for answer in answers if not answer.get("pending")]
    _settle_both(gw, answers, running)
    _assert_reported_once(gw, answers)
    if script.startswith("edit_b ask_b secret_b rescan_b"):  # the restart the rescan applied, in the next answer
        assert [answer["restarted"] for answer in answers[:3]] == [[], ["feishu"], []], answers
    assert {p.value for p in gw.runner._profile_adapters["b"]} == {
        n.lower() for n in gw._lines(".env", "START_", "b")} == set(gw.named_started)


@pytest.mark.parametrize("seed", range(10))
def test_racing_named_and_launch_reconciles_report_each_change_once(tmp_path, monkeypatch, reconcile_loop, seed):
    """Both profiles' triggers, rescans, watcher ticks and asks (some giving up as ``pending``), in any order."""
    import random
    from gateway import run_platform_reconcile as reconcile
    rng = random.Random(seed)
    gw = _FileDrivenPrimary(tmp_path, reconcile_loop, rng, monkeypatch=monkeypatch)
    answers, running, fresh, named_fresh, secrets = [], [], iter(_FRESH), iter(_NAMED_FRESH), iter(range(100))
    def ask(op):
        monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 0.005 if op == "pending_b" else 30.0)
        answer = gw.ask({}) if op == "ask" else gw.ask_named()
        if not answer.get("pending"):
            answers.append(answer)
    ops = {
        "edit": lambda step: gw.edit(f"START_{next(fresh, 'SMS')}=1"),
        "edit_b": lambda step: gw.edit(f"START_{next(named_fresh, 'WEIXIN')}=1", "b"),
        "install_b": lambda step: gw.edit(f"INSTALL_{seed}b{step}=1", "b"),
        "secret_b": lambda step: gw.edit(f"SECRET={next(secrets)}", "b"),
        "plugin": lambda step: running.append(gw.background("plugin")),
        **{op: (lambda step, op=op: running.append(gw.named_background(op))) for op in ("scan_b", "rescan_b", "plugin_b")},
        **{op: (lambda step, op=op: ask(op)) for op in ("ask", "ask_b", "pending_b")},
    }
    for step in range(16):
        ops[rng.choice(sorted(ops))](step)
    monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 30.0)
    _settle_both(gw, answers, running)
    _assert_reported_once(gw, answers)
    again = gw.settled_ask(gw.ask_named)
    assert again["started"] == again["restarted"] == again["installed"] == []  # taken: nothing is new


def test_a_named_profile_its_apply_parks_answers_that_its_platforms_stopped(tmp_path, monkeypatch, reconcile_loop):
    """A save the multiplexer refuses parks the profile and stops its bots: the answer says so, never the
    ``connected`` an earlier ask's apply left in the ledger."""
    gw = _FileDrivenPrimary(tmp_path, reconcile_loop, monkeypatch=monkeypatch)
    gw.edit("START_FEISHU=1", "b")
    first = gw.settled_ask(gw.ask_named)
    assert first["started"] == ["feishu"] and first["platforms"] == {"feishu": "connected"}, first
    gw.edit("OPEN_POLICY=1", "b")
    answer = gw.settled_ask(gw.ask_named)
    assert gw.parked == ["b"] and "b" not in gw.runner._served_profile_homes
    assert answer["applied"] is False and answer["reason"] == "not_served", answer
    assert answer["platforms"] == {} and answer["stopped"] == ["feishu"], answer


def test_a_drained_change_comes_with_the_state_after_it(tmp_path, reconcile_loop):
    """A reconcile finishing between an ask's own apply and its drain: the answer reports its start with
    the state that reconcile left (``retrying``), not the older snapshot the ask's own apply read."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    gw = _FileDrivenPrimary(tmp_path, reconcile_loop)
    ledger = gw.runner.platform_change_ledger
    at_drain, release, drain = threading.Event(), threading.Event(), ledger.drain

    def drain_once_released(profile):
        at_drain.set()
        assert release.wait(30)
        return drain(profile)

    ledger.drain = drain_once_released
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gw.ask, {})
        try:
            assert at_drain.wait(30)
            gw.edit("START_WEBHOOK=1")
            applied = gw.background("plugin").result(30)
        finally:
            release.set()
        answer = future.result(30)
    assert applied["started"] == ["webhook"], applied
    assert answer["started"] == ["webhook"] and answer["platforms"] == {"webhook": "retrying"}, answer


def test_an_answer_returns_inside_the_callers_wait_and_a_pending_one_takes_nothing(
        tmp_path, monkeypatch, reconcile_loop):
    """Caller to socket: a reconcile holding the lock past the verb window answers ``pending`` over the real
    control socket inside the caller's wait (the answer never waits on that reconcile again), and drains
    nothing: the next ask reports the held reconcile's start and the ask's own, once."""
    import time
    from gateway import run_platform_reconcile as reconcile
    from gateway.control_socket import GatewayControlServer, query_gateway_control
    monkeypatch.setattr(reconcile, "_VERB_WAIT_SECS", 0.5)
    gw = _FileDrivenPrimary(tmp_path, reconcile_loop)
    sock_home = tmp_path / "sock"
    sock_home.mkdir(mode=0o700)
    server = GatewayControlServer(sock_home, verb_handlers={"reconcile-platforms": gw.ask})
    assert asyncio.run_coroutine_threadsafe(server.start(), reconcile_loop).result(30)
    caller_wait = 5.0  # production: 15 s against a 10 s window; the same margin, scaled
    try:
        gw.release.clear()
        gw.edit("START_WEBHOOK=1")
        held = gw.background("plugin")  # holds the reconcile lock until released
        gw.edit("START_SLACK=1")
        began = time.monotonic()
        answer = query_gateway_control(sock_home, "reconcile-platforms", timeout=caller_wait)
        elapsed = time.monotonic() - began
        assert answer is not None and answer.get("pending") is True, answer
        assert elapsed < caller_wait, elapsed
        gw.release.set()
        held.result(30)

        def settled():
            reply = query_gateway_control(sock_home, "reconcile-platforms", timeout=caller_wait)
            assert reply is not None, "an answer missed the caller's wait"
            return None if reply.get("pending") else reply
        reported = []
        for _ in range(2):
            reported += wait_until(settled, timeout=60, interval=0.05, what="a settled answer")["started"]
        assert sorted(reported) == ["slack", "webhook"] == sorted(gw.started), reported
    finally:
        gw.release.set()
        asyncio.run_coroutine_threadsafe(server.stop(), reconcile_loop).result(30)


def test_an_answer_reports_the_platform_state_as_it_is_now(tmp_path, reconcile_loop):
    """A reconcile whose start is still retrying leaves it for the next ask; by then the platform failed for
    good. The answer names the start once and the state the platform is in now, not the one it had then."""
    gw = _FileDrivenPrimary(tmp_path, reconcile_loop)
    gw.edit("START_WEBHOOK=1")
    gw.background("plugin").result(60)
    gw.states["webhook"] = "unavailable"
    answer = gw.settled_ask()
    assert answer["started"] == ["webhook"] and answer["platforms"]["webhook"] == "unavailable", answer


def _webhook_yaml(port, home_chat="old"):
    return (f"gateway:\n  multiplex_profiles: false\nplatforms:\n  webhook:\n    enabled: true\n"
            f"    extra:\n      port: {port}\n    home_channel:\n      platform: webhook\n      chat_id: {home_chat}\n")


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_an_unreadable_config_keeps_the_running_platforms_until_it_is_fixed(tmp_path):
    """A half-written config.yaml (an editor mid-save, a typo) must not read as "no platforms": the
    startup loader's env-only fallback stopped every YAML-configured adapter. The gateway keeps what
    runs, says it could not apply the edit, and applies the next readable one as usual."""
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    first, second = _free_port(), _free_port()
    config = root / "config.yaml"
    config.write_text(_webhook_yaml(first), encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        wait_until(lambda: _accepts(first), timeout=15, interval=0.1, what="webhook listening")
        config.write_text("platforms: [\n", encoding="utf-8")
        for _ in range(2):  # a second ask about the same broken file warns no more
            answer = _reconcile(root)
            assert answer["applied"] is False and answer["reason"] == "config_unreadable", (answer, tail(mux))
            assert answer["stopped"] == [] and answer["platforms"].get("webhook") == "connected", answer
            assert "config.yaml cannot be read" in answer["error"]
        assert _accepts(first), tail(mux)

        config.write_text(_webhook_yaml(second), encoding="utf-8")
        answer = _reconcile(root)
        assert answer["applied"] is True and answer["restarted"] == ["webhook"], (answer, tail(mux))
        wait_until(lambda: _accepts(second) and not _accepts(first), timeout=10, interval=0.1,
                   what="webhook moved to the port the fixed config names")
        # The log file lags the answers: once the later restart's line is in it, so is every warning before.
        wait_until(lambda: "restart=['webhook']" in tail(mux, n=10**7), timeout=30, interval=0.1,
                   what="the restart's log line")
        assert tail(mux, n=10**7).count("config.yaml cannot be read; keeping its current platforms") == 1, tail(mux)
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_an_unreadable_named_profile_config_keeps_its_platforms(tmp_path):
    """The same for a multiplexed named profile: its rescan keeps the last good config and adapters."""
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    named = root / "profiles" / "b"
    named.mkdir(parents=True, mode=0o700)
    good = "platforms:\n  line:\n    token: same-token\n"
    (named / "config.yaml").write_text(good, encoding="utf-8")
    (named / ".env").write_text("LINE_CHANNEL_ACCESS_TOKEN=same-token\nLINE_CHANNEL_SECRET=s\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    closed = _closed_port()
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1",
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"), HTTPS_PROXY=closed, HTTP_PROXY=closed,
               https_proxy=closed, http_proxy=closed, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first = _reconcile(root, {"home": str(named)})
        assert first["platforms"].get("line") == "connected", (first, tail(mux))
        (named / "config.yaml").write_text("platforms: [\n", encoding="utf-8")
        answer = _reconcile(root, {"home": str(named)})
        assert answer["applied"] is False and answer["reason"] == "config_unreadable", (answer, tail(mux))
        assert answer["stopped"] == [] and answer["platforms"].get("line") == "connected", (answer, tail(mux))
        (named / "config.yaml").write_text(good, encoding="utf-8")
        answer = _reconcile(root, {"home": str(named)})
        assert answer["applied"] is True and answer["changed"] is False, (answer, tail(mux))
        assert answer["platforms"].get("line") == "connected", (answer, tail(mux))
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)


def test_a_home_channel_change_is_live_without_a_reconnect(tmp_path, monkeypatch):
    """``home_channel`` is read by the runner (deliveries, notices), not the adapter: changing it is no
    rebuild, yet the running gateway must deliver to the new destination, not keep the old one."""
    from gateway.config import load_gateway_config
    from gateway.run import GatewayRunner
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(_webhook_yaml(_free_port()), encoding="utf-8")
    runner = GatewayRunner(load_gateway_config())
    live = types.SimpleNamespace(config=runner.config.platforms[Platform.WEBHOOK])
    runner.adapters[Platform.WEBHOOK] = live
    runner._primary_build_prints = {
        Platform.WEBHOOK: runner._primary_build_print(Platform.WEBHOOK, runner.config.platforms[Platform.WEBHOOK])}
    from gateway.run_platform_reconcile import _credential_print
    runner._primary_credential_print = _credential_print(tmp_path)  # no credential change: no plugin recovery
    runner._running = True
    config.write_text(_webhook_yaml(runner.config.platforms[Platform.WEBHOOK].extra["port"], home_chat="new"),
                      encoding="utf-8")
    answer = asyncio.run(runner.reconcile_profile_platforms(None, reason="test"))
    assert answer["applied"] is True and answer["changed"] is False, answer
    assert runner.adapters[Platform.WEBHOOK] is live  # not rebuilt
    assert runner.delivery_router.config.get_home_channel(Platform.WEBHOOK).chat_id == "new"
    assert live.config.home_channel.chat_id == "new"


def test_explicit_gateway_config_is_not_reconciled_from_home_files(tmp_path, monkeypatch):
    """A ``GatewayRunner`` launched with an explicit ``GatewayConfig`` (``gateway --config <file>`` or
    programmatic injection) is not backed by ``$HERMES_HOME/config.yaml``: unrelated writes to the home's
    config (such as onboarding state on the first turn) must never retire or rebuild its adapters."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platforms: {}\n", encoding="utf-8")
    explicit = GatewayConfig(platforms={Platform.WEBHOOK: PlatformConfig(enabled=True, token="explicit-token")})
    runner = GatewayRunner(explicit)
    live = types.SimpleNamespace(config=runner.config.platforms[Platform.WEBHOOK])
    runner.adapters[Platform.WEBHOOK] = live
    runner._primary_build_prints = {
        Platform.WEBHOOK: runner._primary_build_print(Platform.WEBHOOK, runner.config.platforms[Platform.WEBHOOK])}
    runner._running = True
    retired: list[Platform] = []

    async def _retire(platform):
        retired.append(platform)
        runner.adapters.pop(platform, None)

    runner._retire_primary_platform = _retire
    asyncio.run(runner._note_boot_platform_signature())
    # First turn writes unrelated state into $HERMES_HOME/config.yaml.
    (tmp_path / "config.yaml").write_text("onboarding:\n  seen:\n    profile_build_offered: true\n", encoding="utf-8")

    asyncio.run(runner._reconcile_primary_platforms_if_changed(reason="watcher"))
    assert retired == [] and runner.adapters.get(Platform.WEBHOOK) is live

    answer = asyncio.run(runner.reconcile_profile_platforms(None, reason="control-socket"))
    assert answer["applied"] is True and answer["changed"] is False and answer["stopped"] == [], answer
    assert retired == [] and runner.adapters.get(Platform.WEBHOOK) is live
