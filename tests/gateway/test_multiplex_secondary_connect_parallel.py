"""Regression tests for concurrent secondary-profile adapter connection (BM-8).

Secondary profiles used to be awaited one by one inside
``GatewayAdapterLifecycleMixin._start_secondary_profile_adapters()``, so N
profiles serialized their network connect() handshakes before the gateway
reached ``ready``. These tests prove:

1. Secondary-profile connects overlap concurrently (deterministic barrier proof,
   no flaky wall-clock sleeps) while keeping each task's ContextVar profile
   scope (``get_hermes_home()`` and ``get_secret()``) isolated.
2. Multiple platforms within a single secondary profile also connect
   concurrently.
3. All pre-connect and lifecycle safety invariants remain preserved: duplicate
   credential / listener refusal, provisional claim deferral letting a healthy
   sibling connect after a permanent predecessor failure, ``MultiplexConfigError``
   propagation (with staged holds released), cancellation cleanup across
   preparation / pre-gather / in-flight handshakes, transient profile failure
   isolation (omitted from ``_served_profile_signatures`` and
   ``_profile_configs``), per-adapter connect failure isolation + reconnect
   scheduling, parked-profile skipping, single-profile (multiplex off)
   unchanged, and systemd boot timeout
   renewal during preparation plus max-budget batch extension.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner, MultiplexConfigError
from gateway.status import flush_runtime_status, read_runtime_status
from hermes_constants import get_hermes_home

_BARRIER_TIMEOUT_SECS = 2.5


@pytest.fixture(autouse=True)
def _restore_multiplex_flag():
    prev = is_multiplex_active()
    yield
    set_multiplex_active(prev)


def _mkprofile(home: Path, name: str, env: str = "", config_yaml: str = "model: {default: m}\n") -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (profile_dir / ".env").write_text(env, encoding="utf-8")
    return profile_dir


class _FakeAsyncAdapter(BasePlatformAdapter):
    """Minimal async adapter with controllable connect() behavior."""

    def __init__(
        self,
        platform: Platform,
        config: PlatformConfig,
        connect_impl,
        *,
        sidecar_bind: str | None = None,
        sidecar_port: int | None = None,
        project_secret: str | None = None,
    ):
        super().__init__(config, platform)
        self.token = config.token
        self._connect_impl = connect_impl
        self.connect_calls = 0
        self.disconnect_calls = 0
        if sidecar_bind is not None:
            self._sidecar_bind = sidecar_bind
        if sidecar_port is not None:
            self._sidecar_port = sidecar_port
        if project_secret is not None:
            self._project_secret = project_secret

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self.connect_calls += 1
        return await self._connect_impl(self, is_reconnect=is_reconnect)

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_runner(tmp_path: Path, monkeypatch, *, multiplex: bool = True) -> tuple[GatewayRunner, Path]:
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("gateway.run_startup.recover_left_core_in", lambda *a, **kw: None)
    set_multiplex_active(multiplex)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex, systemd_watchdog_seconds=0)
    runner._running = False
    runner._primary_profile_name = "default"
    runner._platform_lock_takeover_on_start = False
    runner._shutdown_event = asyncio.Event()
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._failed_platforms = {}
    runner._profile_failed_platforms = {}
    runner._background_tasks = set()
    runner._busy_text_mode = "queue"
    runner._busy_text_modes_by_profile = {}
    runner._busy_input_modes_by_profile = {}
    runner.session_store = None
    runner.pairing_store = MagicMock()
    runner.pairing_stores = {}
    runner._register_config_hooks = lambda *a, **kw: None
    runner._sync_voice_mode_state_to_adapter = lambda *a: None
    runner._restore_secondary_completion_ledgers = lambda *a: None
    runner._subscribe_plugin_rewire = lambda *a, **kw: None
    runner._handle_active_session_busy_message = AsyncMock()
    runner._recover_telegram_topic_thread_id = AsyncMock(return_value=None)
    runner._make_adapter_auth_check = lambda platform, profile_name=None: (lambda *a, **kw: True)
    runner._adapter_disconnect_timeout_secs = lambda: 2.5
    return runner, home


async def _drain_background_tasks(runner: GatewayRunner) -> None:
    runner._shutdown_event.set()
    tasks = list(getattr(runner, "_background_tasks", ()))
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_secondary_profiles_connect_concurrently_with_isolated_scopes(tmp_path, monkeypatch):
    """Three secondary profiles must reach connect() together (barrier proof) while each
    task observes only its own HERMES_HOME and secret scope across the await boundary."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    profiles = {
        "alpha": _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n"),
        "beta": _mkprofile(home, "beta", "TELEGRAM_BOT_TOKEN=tok-beta\n"),
        "gamma": _mkprofile(home, "gamma", "TELEGRAM_BOT_TOKEN=tok-gamma\n"),
    }

    in_flight = 0
    max_in_flight = 0
    all_entered = asyncio.Event()
    observed_scopes: dict[str, tuple[Path, str | None]] = {}

    async def barrier_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight == len(profiles):
            all_entered.set()
        try:
            await asyncio.wait_for(all_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
            # Re-check ContextVar scope AFTER awaiting the shared barrier.
            observed_scopes[adapter._hermes_profile_name] = (
                Path(get_hermes_home()).resolve(),
                get_secret("TELEGRAM_BOT_TOKEN"),
            )
            return True
        finally:
            in_flight -= 1

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, barrier_connect)

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert max_in_flight == 3, f"expected 3 concurrent connects, got max_in_flight={max_in_flight}"
    assert connected == 3
    for name, expected_home in profiles.items():
        assert Platform.TELEGRAM in runner._profile_adapters[name]
        assert observed_scopes[name] == (expected_home.resolve(), f"tok-{name}")
        assert name in runner._served_profile_signatures
        assert name in runner._profile_configs


@pytest.mark.asyncio
async def test_single_profile_multiple_platforms_connect_concurrently(tmp_path, monkeypatch):
    """A single secondary profile with multiple platforms connects them concurrently."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    profile_home = _mkprofile(
        home,
        "multi",
        "TELEGRAM_BOT_TOKEN=tok-tg\nDISCORD_BOT_TOKEN=tok-dc\n",
    )

    in_flight = 0
    max_in_flight = 0
    both_entered = asyncio.Event()

    async def barrier_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight == 2:
            both_entered.set()
        try:
            await asyncio.wait_for(both_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
            return True
        finally:
            in_flight -= 1

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, barrier_connect)

    try:
        connected = await runner._start_one_profile_adapters("multi", profile_home, {})
    finally:
        await _drain_background_tasks(runner)

    assert max_in_flight == 2, f"expected 2 concurrent platform connects, got {max_in_flight}"
    assert connected == 2
    assert set(runner._profile_adapters["multi"]) == {Platform.TELEGRAM, Platform.DISCORD}


@pytest.mark.asyncio
async def test_duplicate_credential_refused_before_concurrent_connect(tmp_path, monkeypatch):
    """Two secondary profiles sharing a bot token must still collide deterministically in
    setup: the duplicate never calls connect() and is stamped fatal, while distinct siblings
    connect concurrently."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=shared-token\n")
    _mkprofile(home, "beta", "TELEGRAM_BOT_TOKEN=shared-token\n")
    _mkprofile(home, "gamma", "TELEGRAM_BOT_TOKEN=unique-gamma\n")

    entered_profiles: list[str] = []
    both_unique_entered = asyncio.Event()
    created: dict[str, _FakeAsyncAdapter] = {}

    async def barrier_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        entered_profiles.append(adapter._hermes_profile_name)
        if len(entered_profiles) == 2:
            both_unique_entered.set()
        await asyncio.wait_for(both_unique_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
        return True

    def make_adapter(platform, cfg):
        adapter = _FakeAsyncAdapter(platform, cfg, barrier_connect)
        created[Path(get_hermes_home()).name] = adapter
        return adapter

    runner._create_adapter = make_adapter

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 2
    assert set(entered_profiles) == {"alpha", "gamma"}
    assert created["beta"].connect_calls == 0
    assert Platform.TELEGRAM in runner._profile_adapters["alpha"]
    assert Platform.TELEGRAM in runner._profile_adapters["gamma"]
    assert Platform.TELEGRAM not in runner._profile_adapters.get("beta", {})

    flush_runtime_status()
    state = read_runtime_status(home / "gateway_state.json")
    beta_status = (state.get("platforms") or {}).get("beta:telegram") or {}
    assert beta_status.get("state") == "fatal"
    assert beta_status.get("error_code") == "duplicate_credential"


@pytest.mark.asyncio
async def test_whatsapp_secondaries_connect_one_per_wave_in_profile_order(tmp_path, monkeypatch):
    """A WhatsApp secondary picks its bridge port from the ports bound when it connects, so a sibling's
    configured port is safe only once that sibling's bridge is up. WhatsApp adapters carry no credential or
    listener claim, so they connect one per wave in profile order, as the serial sweep did, while other
    platforms keep connecting in the first wave."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    for name in ("alpha", "beta", "gamma"):
        session = _mkprofile(home, name, "") / "platforms" / "whatsapp" / "session"
        session.mkdir(parents=True)
        (session / "creds.json").write_text("{}", encoding="utf-8")

    def load_cfg():
        name = Path(get_hermes_home()).name
        cfg = GatewayConfig(multiplex_profiles=True)
        cfg.platforms = {Platform.WHATSAPP: PlatformConfig(enabled=True)}
        if name == "beta":
            cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="tok-beta")
        return cfg

    monkeypatch.setattr("gateway.config.load_gateway_config", load_cfg)
    started: list[str] = []
    in_flight = {"whatsapp": 0, "max": 0}

    async def connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        started.append(f"{adapter.profile}:{adapter.platform.value}")
        if adapter.platform is Platform.WHATSAPP:
            in_flight["whatsapp"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["whatsapp"])
        for _ in range(20):  # yield so every other connect in this wave starts meanwhile
            await asyncio.sleep(0)
        if adapter.platform is Platform.WHATSAPP:
            in_flight["whatsapp"] -= 1
        return True

    def make_adapter(platform, cfg):
        adapter = _FakeAsyncAdapter(platform, cfg, connect)
        adapter.profile = Path(get_hermes_home()).name
        return adapter

    runner._create_adapter = make_adapter

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 4
    assert in_flight["max"] == 1
    assert [s for s in started if s.endswith(":whatsapp")] == ["alpha:whatsapp", "beta:whatsapp", "gamma:whatsapp"]
    assert started.index("beta:telegram") < started.index("beta:whatsapp")  # not held back behind alpha


@pytest.mark.asyncio
async def test_duplicate_listener_refused_before_concurrent_connect(tmp_path, monkeypatch):
    """Two secondary Photon adapters on the same bind:port with distinct secrets collide
    deterministically before connect()."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    photon = Platform("photon")
    _mkprofile(home, "alpha", "")
    _mkprofile(home, "beta", "")

    def load_cfg():
        name = Path(get_hermes_home()).name
        cfg = GatewayConfig(multiplex_profiles=True)
        cfg.platforms = {photon: PlatformConfig(enabled=True, token=f"secret-{name}")}
        return cfg

    monkeypatch.setattr("gateway.config.load_gateway_config", load_cfg)

    created: dict[str, _FakeAsyncAdapter] = {}

    async def ok_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        return True

    def make_adapter(platform, cfg):
        name = Path(get_hermes_home()).name
        adapter = _FakeAsyncAdapter(
            platform,
            cfg,
            ok_connect,
            sidecar_bind="127.0.0.1",
            sidecar_port=8789,
            project_secret=f"secret-{name}",
        )
        created[name] = adapter
        return adapter

    runner._create_adapter = make_adapter

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 1
    assert created["alpha"].connect_calls == 1
    assert created["beta"].connect_calls == 0
    assert photon in runner._profile_adapters["alpha"]
    assert photon not in runner._profile_adapters.get("beta", {})

    flush_runtime_status()
    state = read_runtime_status(home / "gateway_state.json")
    beta_status = (state.get("platforms") or {}).get("beta:photon") or {}
    assert beta_status.get("state") == "fatal"
    assert beta_status.get("error_code") == "duplicate_listener"


@pytest.mark.asyncio
async def test_multiplex_config_error_raises_and_releases_staged_claims(tmp_path, monkeypatch):
    """A MultiplexConfigError in a later profile aborts boot before any staged adapter
    connects and releases already-held secondary claims."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    _mkprofile(
        home,
        "beta",
        "WHATSAPP_ENABLED=true\nWHATSAPP_DM_POLICY=open\n",
    )

    connect_calls = []

    async def unexpected_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        connect_calls.append(adapter._hermes_profile_name)
        return True

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, unexpected_connect)

    try:
        with pytest.raises(MultiplexConfigError, match="beta"):
            await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connect_calls == []
    assert not getattr(runner, "_secondary_held_claims", {})


@pytest.mark.asyncio
async def test_transient_profile_failure_omitted_while_siblings_connect_concurrently(tmp_path, monkeypatch):
    """A transient failure loading one profile omits it from _served_profile_signatures and
    _profile_configs while healthy siblings still connect concurrently."""
    from hermes_cli import env_loader

    runner, home = _make_runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    _mkprofile(home, "broken", "TELEGRAM_BOT_TOKEN=tok-broken\n")
    _mkprofile(home, "gamma", "TELEGRAM_BOT_TOKEN=tok-gamma\n")

    real_hydrate = env_loader.hydrate_profile_secret_sources

    def flaky_hydrate(profile_home):
        if Path(profile_home).name == "broken":
            raise OSError("secret backend unreachable")
        return real_hydrate(profile_home)

    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", flaky_hydrate)

    in_flight = 0
    max_in_flight = 0
    both_entered = asyncio.Event()

    async def barrier_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight == 2:
            both_entered.set()
        try:
            await asyncio.wait_for(both_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
            return True
        finally:
            in_flight -= 1

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, barrier_connect)

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert max_in_flight == 2
    assert connected == 2
    assert Platform.TELEGRAM in runner._profile_adapters["alpha"]
    assert Platform.TELEGRAM in runner._profile_adapters["gamma"]
    assert "broken" not in runner._served_profile_signatures
    assert "broken" not in runner._profile_configs
    assert "alpha" in runner._served_profile_signatures and "alpha" in runner._profile_configs
    assert "gamma" in runner._served_profile_signatures and "gamma" in runner._profile_configs


@pytest.mark.asyncio
async def test_connect_failure_and_exception_do_not_block_siblings(tmp_path, monkeypatch):
    """Per-adapter connect() returning False or raising isolates failure, disconnects
    defensively, schedules startup reconnect, and lets healthy siblings connect concurrently."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    _mkprofile(home, "beta", "TELEGRAM_BOT_TOKEN=tok-beta\n")
    _mkprofile(home, "gamma", "TELEGRAM_BOT_TOKEN=tok-gamma\n")

    in_flight = 0
    all_entered = asyncio.Event()
    created: dict[str, _FakeAsyncAdapter] = {}

    async def mixed_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        nonlocal in_flight
        in_flight += 1
        if in_flight == 3:
            all_entered.set()
        try:
            await asyncio.wait_for(all_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
            name = adapter._hermes_profile_name
            if name == "alpha":
                return False
            if name == "beta":
                raise RuntimeError("transient network reset")
            return True
        finally:
            in_flight -= 1

    def make_adapter(platform, cfg):
        adapter = _FakeAsyncAdapter(platform, cfg, mixed_connect)
        created[Path(get_hermes_home()).name] = adapter
        return adapter

    runner._create_adapter = make_adapter

    try:
        connected = await runner._start_secondary_profile_adapters()
        scheduled_names = {t.get_name() for t in runner._background_tasks}
    finally:
        await _drain_background_tasks(runner)

    assert connected == 1
    assert Platform.TELEGRAM in runner._profile_adapters["gamma"]
    assert Platform.TELEGRAM not in runner._profile_adapters.get("alpha", {})
    assert Platform.TELEGRAM not in runner._profile_adapters.get("beta", {})
    assert created["alpha"].disconnect_calls == 1
    assert created["beta"].disconnect_calls == 1
    assert "secondary-startup-reconnect:alpha:telegram" in scheduled_names
    assert "secondary-startup-reconnect:beta:telegram" in scheduled_names


@pytest.mark.asyncio
async def test_parked_profiles_skipped_and_single_profile_unchanged(tmp_path, monkeypatch):
    """Parked secondary profiles stay skipped while unparked siblings connect concurrently,
    and multiplex=False clears served_profiles without touching any secondary."""
    runner, home = _make_runner(tmp_path, monkeypatch, multiplex=True)
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    snoozed = _mkprofile(home, "snoozed", "TELEGRAM_BOT_TOKEN=tok-snoozed\n")
    (snoozed / "gateway.parked").write_text("", encoding="utf-8")
    _mkprofile(home, "gamma", "TELEGRAM_BOT_TOKEN=tok-gamma\n")

    in_flight = 0
    both_entered = asyncio.Event()

    async def barrier_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        nonlocal in_flight
        in_flight += 1
        if in_flight == 2:
            both_entered.set()
        try:
            await asyncio.wait_for(both_entered.wait(), timeout=_BARRIER_TIMEOUT_SECS)
            return True
        finally:
            in_flight -= 1

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, barrier_connect)

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 2
    assert set(runner._profile_adapters) == {"alpha", "gamma"}
    assert "snoozed" not in runner._profile_adapters

    # Single-profile (multiplex off) returns 0 and clears stale served_profiles.
    solo_runner, solo_home = _make_runner(tmp_path / "solo", monkeypatch, multiplex=False)
    (solo_home / "gateway_state.json").write_text(
        json.dumps({"pid": 1, "gateway_state": "stopped", "served_profiles": ["default", "alpha"]}),
        encoding="utf-8",
    )
    assert await solo_runner._start_secondary_profile_adapters() == 0
    flush_runtime_status()
    assert read_runtime_status(solo_home / "gateway_state.json")["served_profiles"] == []


@pytest.mark.asyncio
async def test_systemd_boot_extension_uses_max_budget_once_for_batch(tmp_path, monkeypatch):
    """Each adapter in a concurrent connect wave extends with the wave's largest platform
    budget, so a shorter-budget sibling (Discord) prepared last cannot shorten Telegram's
    parallel connect deadline."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    runner.config.systemd_watchdog_seconds = 30.0
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    _mkprofile(home, "beta", "DISCORD_BOT_TOKEN=tok-beta\n")

    extensions: list[float] = []
    monkeypatch.setattr("gateway.systemd_notify.extend_start_timeout", lambda secs: extensions.append(secs) or True)

    async def ok_connect(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        return True

    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, ok_connect)

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 2
    tg_budget = runner._platform_connect_timeout_secs(Platform.TELEGRAM, initial=True)
    dc_budget = runner._platform_connect_timeout_secs(Platform.DISCORD, initial=True)
    expected = max(tg_budget, dc_budget) + 30.0
    assert extensions[-2:] == [expected, expected]


@pytest.mark.asyncio
async def test_slow_wall_clock_preparation_renews_systemd_boot_deadline(tmp_path, monkeypatch):
    """When preparing the first secondary profile consumes >= 30s of monotonic wall-clock time,
    systemd's start deadline is renewed again before the second profile loads."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    runner.config.systemd_watchdog_seconds = 30.0
    _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
    _mkprofile(home, "beta", "DISCORD_BOT_TOKEN=tok-beta\n")

    clock = 100.0
    extensions_at_load: dict[str, int] = {}
    extensions: list[float] = []
    real_load = runner._load_secondary_profile_config

    monkeypatch.setattr("gateway.run_adapters.time.monotonic", lambda: clock)
    monkeypatch.setattr("gateway.systemd_notify.extend_start_timeout", lambda secs: extensions.append(secs) or True)

    async def slow_first_load(profile_name, profile_home):
        nonlocal clock
        extensions_at_load[profile_name] = len(extensions)
        cfg = await real_load(profile_name, profile_home)
        if profile_name == "alpha":
            clock += 35.0
        return cfg

    runner._load_secondary_profile_config = slow_first_load
    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 2
    assert extensions_at_load["alpha"] == 1
    assert extensions_at_load["beta"] == 3
    assert len(extensions) == 6


@pytest.mark.asyncio
async def test_slow_multi_platform_single_profile_prep_renews_systemd_boot_deadline(tmp_path, monkeypatch):
    """Three slow platforms (35s each) in ONE secondary profile must renew systemd's start
    deadline between platforms during preparation so 105s of prep never crosses a 90s timeout."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    runner.config.systemd_watchdog_seconds = 30.0
    _mkprofile(
        home,
        "solo",
        "TELEGRAM_BOT_TOKEN=tok-tg\nDISCORD_BOT_TOKEN=tok-dc\nSLACK_BOT_TOKEN=tok-sl\nSLACK_APP_TOKEN=xapp-sl\n",
    )

    now = 0.0
    deadline = 90.0
    expired: list[tuple[str, float, float]] = []
    extensions: list[float] = []
    extensions_at_create: list[int] = []

    monkeypatch.setattr("gateway.run_adapters.time.monotonic", lambda: now)

    def notify(secs: float) -> bool:
        nonlocal deadline
        extensions.append(secs)
        deadline = max(deadline, now + secs)
        return True

    monkeypatch.setattr("gateway.systemd_notify.extend_start_timeout", notify)

    def slow_create(platform, cfg):
        nonlocal now
        extensions_at_create.append(len(extensions))
        now += 35.0
        if now >= deadline:
            expired.append((platform.value, now, deadline))
        return _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))

    runner._create_adapter = slow_create

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 3
    assert now == 105.0
    assert not expired, f"crossed systemd start deadline during single-profile prep: {expired}"
    assert extensions_at_create == [2, 3, 4]
    assert len(extensions) == 7


@pytest.mark.asyncio
async def test_slow_preparation_renews_systemd_boot_deadline(tmp_path, monkeypatch):
    """Slow sequential preparation renews systemd's start deadline as each profile prepares
    so 3x35s preparations never exhaust a 90s unit start timeout before notification."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    runner.config.systemd_watchdog_seconds = 30.0
    for name in ("alpha", "beta", "gamma"):
        _mkprofile(home, name, f"TELEGRAM_BOT_TOKEN=tok-{name}\n")

    now = 0.0
    deadline = 90.0
    expired: list[tuple[str, float, float]] = []
    real_load = runner._load_secondary_profile_config

    async def slow_load(profile_name, profile_home):
        nonlocal now
        now += 35.0
        if now >= deadline:
            expired.append((profile_name, now, deadline))
        return await real_load(profile_name, profile_home)

    def notify(secs: float) -> bool:
        nonlocal deadline
        deadline = max(deadline, now + secs)
        return True

    runner._load_secondary_profile_config = slow_load
    monkeypatch.setattr("gateway.systemd_notify.extend_start_timeout", notify)
    runner._create_adapter = lambda platform, cfg: _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 3
    assert not expired, f"crossed systemd start deadline before notification: {expired}"


@pytest.mark.asyncio
@pytest.mark.parametrize("second_step", ["profile_load", "platform_build"])
async def test_each_preparation_step_renews_a_partly_spent_start_budget(tmp_path, monkeypatch, second_step):
    """Slow primary connects can leave the secondary sweep part of systemd's 90s start timeout. Each
    preparation step renews the deadline before it runs, as the serial sweep did before each connect:
    a 29s build and then a 32s profile load (or a 32s build of the profile's next platform) must not
    cross a deadline that the start of preparation set 60s ahead."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    runner.config.systemd_watchdog_seconds = 30.0
    if second_step == "profile_load":
        _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-alpha\n")
        _mkprofile(home, "beta", "DISCORD_BOT_TOKEN=tok-beta\n")
    else:
        _mkprofile(home, "alpha", "TELEGRAM_BOT_TOKEN=tok-tg\nDISCORD_BOT_TOKEN=tok-dc\n")

    now, deadline = 60.0, 90.0  # the primary phase already used 60s of the unit's start timeout
    expired: list[tuple[str, float, float]] = []
    steps = iter([29.0, 32.0, 0.0])
    real_load = runner._load_secondary_profile_config

    def notify(secs: float) -> bool:
        nonlocal deadline
        deadline = max(deadline, now + secs)
        return True

    def spend(what: str, secs: float) -> None:
        nonlocal now
        now += secs
        if now >= deadline:
            expired.append((what, now, deadline))

    async def load(profile_name, profile_home):
        if profile_name == "beta":
            spend("load beta", next(steps))
        return await real_load(profile_name, profile_home)

    def create(platform, cfg):
        spend(f"build {platform.value}", next(steps))
        return _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))

    monkeypatch.setattr("gateway.systemd_notify.extend_start_timeout", notify)
    runner._load_secondary_profile_config = load
    runner._create_adapter = create

    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 2
    assert not expired, f"crossed systemd start deadline during preparation: {expired}"


@pytest.mark.asyncio
async def test_cancellation_during_prepare_releases_staged_claims_without_disconnect(tmp_path, monkeypatch):
    """Cancellation while preparing a later profile releases earlier provisional claims
    without disconnecting never-started adapters."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    for name in ("alpha", "beta"):
        _mkprofile(home, name, f"TELEGRAM_BOT_TOKEN=tok-{name}\n")

    created: dict[str, _FakeAsyncAdapter] = {}

    def make_adapter(platform, cfg):
        adapter = _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))
        created[Path(get_hermes_home()).name] = adapter
        return adapter

    runner._create_adapter = make_adapter
    real_load = runner._load_secondary_profile_config

    async def cancelling_load(profile_name, profile_home):
        if profile_name == "beta":
            raise asyncio.CancelledError
        return await real_load(profile_name, profile_home)

    runner._load_secondary_profile_config = cancelling_load

    try:
        with pytest.raises(asyncio.CancelledError):
            await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert not getattr(runner, "_secondary_held_claims", {})
    assert created["alpha"].connect_calls == 0
    assert created["alpha"].disconnect_calls == 0


@pytest.mark.asyncio
async def test_cancellation_before_gather_workers_releases_all_claims_without_disconnect(tmp_path, monkeypatch):
    """Cancellation right before concurrent connect workers start releases every staged
    provisional hold and never calls disconnect() on never-started adapters."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    for name in ("alpha", "beta"):
        _mkprofile(home, name, f"TELEGRAM_BOT_TOKEN=tok-{name}\n")

    created: dict[str, _FakeAsyncAdapter] = {}

    def make_adapter(platform, cfg):
        adapter = _FakeAsyncAdapter(platform, cfg, AsyncMock(return_value=True))
        created[Path(get_hermes_home()).name] = adapter
        return adapter

    runner._create_adapter = make_adapter
    real_prepare, prepared = runner._prepare_one_profile_adapters, []

    async def prepare(profile_name, profile_home, claimed, batch):
        await real_prepare(profile_name, profile_home, claimed, batch)
        prepared.append(profile_name)

    runner._prepare_one_profile_adapters = prepare

    def cancel_before_wave(platforms):
        if len(prepared) == 2:  # preparation renewals look the budget up too; the wave's lookup comes last
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        return next(iter(platforms), None)

    runner._max_boot_timeout_platform = cancel_before_wave

    try:
        with pytest.raises(asyncio.CancelledError):
            await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert not getattr(runner, "_secondary_held_claims", {})
    assert created["alpha"].connect_calls == 0 and created["alpha"].disconnect_calls == 0
    assert created["beta"].connect_calls == 0 and created["beta"].disconnect_calls == 0


@pytest.mark.asyncio
async def test_cancellation_during_handshake_disconnects_untracked_and_preserves_holds(tmp_path, monkeypatch):
    """Cancellation mid-handshake disconnects started-but-untracked adapters under their own
    profile scopes, releases abandoned provisional holds, leaves already-tracked adapters
    alone, and preserves holds already transferred to reconnect bridge tasks."""
    runner, home = _make_runner(tmp_path, monkeypatch)
    homes = {
        name: _mkprofile(home, name, f"TELEGRAM_BOT_TOKEN=tok-{name}\n")
        for name in ("alpha", "beta", "gamma", "delta")
    }

    boot = asyncio.current_task()
    alpha_done = asyncio.Event()
    beta_done = asyncio.Event()
    delta_in_flight = asyncio.Event()
    disconnect_scopes: dict[str, Path] = {}
    created: dict[str, _FakeAsyncAdapter] = {}

    class _ScopedAdapter(_FakeAsyncAdapter):
        async def disconnect(self) -> None:
            disconnect_scopes[self._hermes_profile_name] = Path(get_hermes_home()).resolve()
            await super().disconnect()

    async def handshake(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        name = adapter._hermes_profile_name
        if name == "alpha":
            alpha_done.set()
            return True  # already tracked before cancellation
        if name == "beta":
            beta_done.set()
            return False  # retryable failure transfers hold to reconnect bridge
        if name == "delta":
            delta_in_flight.set()
            await asyncio.Event().wait()
        # gamma waits for alpha (connected), beta (queued), and delta (in flight), then cancels boot.
        await asyncio.wait_for(alpha_done.wait(), timeout=_BARRIER_TIMEOUT_SECS)
        await asyncio.wait_for(beta_done.wait(), timeout=_BARRIER_TIMEOUT_SECS)
        await asyncio.wait_for(delta_in_flight.wait(), timeout=_BARRIER_TIMEOUT_SECS)
        # Let beta finish its post-failure disconnect + reconnect-bridge handoff before cancelling boot.
        for _ in range(5):
            if ("beta", Platform.TELEGRAM) in getattr(runner, "_secondary_held_claims", {}) and (
                runner._secondary_held_claims[("beta", Platform.TELEGRAM)][0] is not created["beta"]
            ):
                break
            await asyncio.sleep(0)
        asyncio.get_running_loop().call_soon(boot.cancel)
        await asyncio.Event().wait()
        return False

    def make_adapter(platform, cfg):
        adapter = _ScopedAdapter(platform, cfg, handshake)
        created[Path(get_hermes_home()).name] = adapter
        return adapter

    runner._create_adapter = make_adapter

    try:
        with pytest.raises(asyncio.CancelledError):
            await runner._start_secondary_profile_adapters()
        held_after_cancel = dict(getattr(runner, "_secondary_held_claims", {}))
    finally:
        await _drain_background_tasks(runner)

    # Tracked alpha was NOT disconnected; in-flight gamma/delta WERE disconnected under their scopes.
    assert runner._profile_adapters["alpha"][Platform.TELEGRAM] is created["alpha"]
    assert created["alpha"].disconnect_calls == 0
    assert created["gamma"].disconnect_calls == 1
    assert created["delta"].disconnect_calls == 1
    assert disconnect_scopes["gamma"] == homes["gamma"].resolve()
    assert disconnect_scopes["delta"] == homes["delta"].resolve()
    # Provisional holds for gamma/delta were released; beta's reconnect-bridge hold was preserved.
    assert ("gamma", Platform.TELEGRAM) not in held_after_cancel
    assert ("delta", Platform.TELEGRAM) not in held_after_cancel
    assert ("beta", Platform.TELEGRAM) in held_after_cancel
    assert held_after_cancel[("beta", Platform.TELEGRAM)][0] is not created["beta"]


@pytest.mark.asyncio
async def test_provisional_claim_released_on_permanent_failure_lets_sibling_connect(tmp_path, monkeypatch):
    """A provisional listener/credential conflict is deferred instead of stamped fatal upfront:
    if alpha fails permanently (non-retryable) and releases its hold, beta connects in the next
    wave; if alpha fails retryably and keeps a reconnect hold, beta is refused as duplicate."""
    photon = Platform("photon")

    # 1. Permanent failure on alpha releases the shared listener -> beta connects.
    runner, home = _make_runner(tmp_path / "perm", monkeypatch)
    for name in ("alpha", "beta"):
        _mkprofile(home, name)

    def load_cfg():
        name = Path(get_hermes_home()).name
        cfg = GatewayConfig(multiplex_profiles=True)
        cfg.platforms = {photon: PlatformConfig(enabled=True, token=f"key-{name}")}
        return cfg

    monkeypatch.setattr("gateway.config.load_gateway_config", load_cfg)

    async def perm_handshake(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        if adapter._hermes_profile_name == "alpha":
            adapter._set_fatal_error("MISSING_CREDENTIALS", "missing project ID", retryable=False)
            return False
        return True

    runner._create_adapter = lambda p, c: _FakeAsyncAdapter(
        p, c, perm_handshake, sidecar_bind="127.0.0.1", sidecar_port=8789, project_secret=c.token,
    )
    try:
        connected = await runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(runner)

    assert connected == 1
    assert photon not in runner._profile_adapters.get("alpha", {})
    assert photon in runner._profile_adapters["beta"]

    # 2. Retryable failure on alpha transfers hold to reconnect bridge -> beta is refused fatal.
    retry_runner, retry_home = _make_runner(tmp_path / "retry", monkeypatch)
    for name in ("alpha", "beta"):
        _mkprofile(retry_home, name)
    created_retry: dict[str, _FakeAsyncAdapter] = {}

    async def retry_handshake(adapter: _FakeAsyncAdapter, *, is_reconnect: bool = False) -> bool:
        return adapter._hermes_profile_name != "alpha"

    def make_retry_adapter(p, c):
        adapter = _FakeAsyncAdapter(
            p, c, retry_handshake, sidecar_bind="127.0.0.1", sidecar_port=8789, project_secret=c.token,
        )
        created_retry[Path(get_hermes_home()).name] = adapter
        return adapter

    retry_runner._create_adapter = make_retry_adapter
    try:
        retry_connected = await retry_runner._start_secondary_profile_adapters()
    finally:
        await _drain_background_tasks(retry_runner)

    assert retry_connected == 0
    assert created_retry["beta"].connect_calls == 0
    assert photon not in retry_runner._profile_adapters.get("beta", {})
    flush_runtime_status()
    state = read_runtime_status(retry_home / "gateway_state.json")
    beta_status = (state.get("platforms") or {}).get("beta:photon") or {}
    assert beta_status.get("state") == "fatal"
    assert beta_status.get("error_code") == "duplicate_listener"
