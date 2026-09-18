"""Hot-serve invariants for ``gateway.multiplex_profiles`` (``gateway/run_profile_reconcile.py``).

The multiplexer used to enumerate ``profiles/`` once at boot; these pin the runtime reconcile: a
profile created afterwards is served, a deleted one is torn down and unrouted, a served profile whose
config/.env changed (bot token added after create) gets its adapters, and none of it touches the other
profiles' live adapters. The cron ticker's live enumerator is covered in ``tests/cron``.

The runner is set up the way ``run_bootstrap.start_gateway`` leaves it: the boot set reserved through
``process_ownership`` and frozen into ``config._runtime_profile_homes``. The reconcile must diff the
LIVE ``profiles/`` against that reservation and grow/shrink it — a reconcile that reads the snapshot
back as "what exists" can never see a new profile (the regression these tests used to pass through).
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.run_profile_reconcile import profile_serve_signature


class _Adapter:
    platform = Platform.DISCORD

    def __init__(self, token):
        self.token = token
        self.disconnected = False
        self.cancelled = False

    async def disconnect(self):
        self.disconnected = True

    async def cancel_background_tasks(self):
        self.cancelled = True


def _runner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_authorities = None  # adapters-only runner: no per-profile session authority
    runner._running = True
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._failed_platforms = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner.pairing_store = MagicMock()
    runner.pairing_stores = {}
    runner._adapter_disconnect_timeout_secs = lambda: 0.5
    started = []

    async def _start(profile_name, profile_home, claimed):
        started.append(profile_name)
        token = (profile_home / ".env").read_text(encoding="utf-8") if (profile_home / ".env").exists() else ""
        if "DISCORD_BOT_TOKEN" not in token:
            return 0
        runner._profile_adapters.setdefault(profile_name, {})[Platform.DISCORD] = _Adapter(token)
        return 1

    runner._start_one_profile_adapters = _start
    runner._adapter_credential_fingerprint = lambda adapter: getattr(adapter, "token", None)
    runner._started = started
    return runner, home


def _boot(runner, home):
    """What ``start_gateway`` does before the runner exists: reserve every profile that exists now and
    freeze that set as the reservation the process owns."""
    from gateway.run import _multiplex_profile_homes
    from gateway.runtime_ownership import process_ownership
    boot_set = _multiplex_profile_homes(runner.config)
    process_ownership.reserve([h for _n, h in boot_set])
    runner.config._runtime_profile_homes = tuple(boot_set)


@pytest.fixture(autouse=True)
def _release_reservations():
    from gateway.runtime_ownership import process_ownership
    yield
    for reserved in process_ownership.homes:
        process_ownership.release(reserved)


def _reserved_names(runner):
    return sorted(name for name, _home in runner.config._runtime_profile_homes)


def _mkprofile(home, name, env=""):
    d = home / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (d / ".env").write_text(env, encoding="utf-8")
    return d


def _served_record(home):
    return json.loads((home / "gateway_state.json").read_text(encoding="utf-8")).get("served_profiles")


@pytest.mark.asyncio
async def test_created_then_credentialed_profile_is_served_without_restart(tmp_path, monkeypatch):
    runner, home = _runner(tmp_path, monkeypatch)
    alpha_dir = _mkprofile(home, "alpha", "DISCORD_BOT_TOKEN=alpha-token\n")
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        _boot(runner, home)
        await runner._start_secondary_profile_adapters()
        alpha_adapter = runner._profile_adapters["alpha"][Platform.DISCORD]
        assert _served_record(home) == ["default", "alpha"]

        # 1. Created while running, no token yet: served (routes/prefixes/cron), zero adapters, and the
        #    reservation grew to include it (the next restart's reserve, the served set, the cron ticker).
        gamma_dir = _mkprofile(home, "gamma")
        result = await runner.reconcile_served_profiles()
        assert result["added"] == ["gamma"]
        assert _served_record(home) == ["default", "alpha", "gamma"]
        assert _reserved_names(runner) == ["alpha", "default", "gamma"]
        from gateway.runtime_ownership import process_ownership
        assert process_ownership.owns(gamma_dir)
        assert "gamma" in runner.pairing_stores
        assert Platform.DISCORD not in runner._profile_adapters.get("gamma", {})

        # 2. Token added afterwards: the rescan builds the adapter (never "adapter-less forever").
        (gamma_dir / ".env").write_text("DISCORD_BOT_TOKEN=gamma-token\n", encoding="utf-8")
        result = await runner.reconcile_served_profiles()
        assert result["rescanned"] == ["gamma"]
        assert runner._profile_adapters["gamma"][Platform.DISCORD].token.strip().endswith("gamma-token")

        # 3. A no-op rescan and the whole sequence never touched alpha's live adapter.
        assert await runner.reconcile_served_profiles() == {
            "added": [], "removed": [], "rescanned": [], "parked": [], "reason": "request",
            "served_profiles": ["default", "alpha", "gamma"],
        }
        assert runner._profile_adapters["alpha"][Platform.DISCORD] is alpha_adapter
        assert alpha_adapter.disconnected is False
        assert runner._started.count("alpha") == 1
        assert profile_serve_signature(alpha_dir) == runner._served_profile_signatures["alpha"]


@pytest.mark.asyncio
async def test_deleted_profile_is_torn_down_and_unrouted_others_untouched(tmp_path, monkeypatch):
    runner, home = _runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "DISCORD_BOT_TOKEN=alpha-token\n")
    gamma_dir = _mkprofile(home, "gamma", "DISCORD_BOT_TOKEN=gamma-token\n")
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        _boot(runner, home)
        await runner._start_secondary_profile_adapters()
        alpha_adapter = runner._profile_adapters["alpha"][Platform.DISCORD]
        gamma_adapter = runner._profile_adapters["gamma"][Platform.DISCORD]
        runner._agent_cache = {"agent:gamma:discord:dm:1": ("agent",), "agent:alpha:discord:dm:1": ("agent",)}
        evicted = []
        runner._evict_cached_agent = evicted.append
        reconnect = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
        runner._profile_failed_platforms = {"gamma": {Platform.TELEGRAM: reconnect}}

        from hermes_constants import mark_named_profile_deleted
        mark_named_profile_deleted(gamma_dir)  # what ``delete_profile`` does before rmtree
        result = await runner.reconcile_served_profiles()

    assert result["removed"] == ["gamma"]
    assert gamma_adapter.disconnected is True and gamma_adapter.cancelled is True
    assert "gamma" not in runner._profile_adapters
    assert "gamma" not in runner.pairing_stores
    assert reconnect.cancelled()
    assert evicted == ["agent:gamma:discord:dm:1"]
    assert _served_record(home) == ["default", "alpha"]
    # The reservation shrank with it: the next restart must not reserve a home that is gone.
    assert _reserved_names(runner) == ["alpha", "default"]
    from gateway.runtime_ownership import process_ownership
    assert not process_ownership.owns(gamma_dir)
    assert runner._profile_adapters["alpha"][Platform.DISCORD] is alpha_adapter
    assert alpha_adapter.disconnected is False


@pytest.mark.asyncio
async def test_hot_added_profile_cannot_double_claim_a_live_secondary_token(tmp_path, monkeypatch):
    """Boot's duplicate-credential guard sees every profile at once; a hot add must see the LIVE
    secondaries' claims too, or the new profile starts a second poller on alpha's bot."""
    runner, home = _runner(tmp_path, monkeypatch)
    _mkprofile(home, "alpha", "DISCORD_BOT_TOKEN=shared\n")
    seen_claims = {}

    async def _start(profile_name, profile_home, claimed):
        seen_claims[profile_name] = dict(claimed)
        runner._profile_adapters.setdefault(profile_name, {})[Platform.DISCORD] = _Adapter("shared")
        return 1

    runner._start_one_profile_adapters = _start
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        _boot(runner, home)
        await runner._start_secondary_profile_adapters()
        _mkprofile(home, "dupe", "DISCORD_BOT_TOKEN=shared\n")
        await runner.reconcile_served_profiles()
    fp = GatewayRunner._adapter_credential_fingerprint(_Adapter("shared"))
    assert seen_claims["dupe"].get((Platform.DISCORD, fp)) == "alpha"
