"""Plugin module-level caches must not hand profile A's state to profile B under a multiplexed
HERMES_HOME override (``hermes_constants.set_hermes_home_override``).

One invariant per mechanism: home-keyed slot with the unscoped module slot intact (router; yuanbao's
ClassVar twin), credential-fingerprinted catalog keys (openrouter), per-home registries (memory
provider skills), collect-all atexit (openviking), lru_cache keyed by the home (disk-cleanup).
Only HTTP transports are faked; the caches themselves are exercised for real.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Profile A (launch home, ``HERMES_HOME``) and profile B with different config/.env values."""
    root = tmp_path / ".hermes"
    a, b = root, root / "profiles" / "B"
    for home, tag in ((a, "A"), (b, "B")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(f"memory:\n  provider: prov{tag}\n", encoding="utf-8")
        (home / ".env").write_text(
            f"RAMP_ROUTER_API_KEY=router-key-{tag}\nRAMP_ROUTER_BASE_URL=https://{tag.lower()}.router.test/v1\n",
            encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    for var in ("RAMP_ROUTER_API_KEY", "RAMP_ROUTER_BASE_URL", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(var, raising=False)
    return a, b


@contextlib.contextmanager
def scoped(home: Path):
    t_home = set_hermes_home_override(str(home))
    t_secret = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(t_secret)
        reset_hermes_home_override(t_home)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def read(self):
        return json.dumps(self._payload).encode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _router():
    from providers import get_provider_profile

    profile = get_provider_profile("router")
    return profile, sys.modules[type(profile).__module__]


def test_router_efforts_cache_and_base_url_follow_the_active_profile(homes, monkeypatch):
    """Efforts map + once-only flags are per home under an override (and the warm thread inherits the
    scope), while the unscoped path keeps using the module slots; the base URL comes from the
    profile's .env."""
    import hermes_cli.urllib_security as urllib_security

    a, b = homes
    profile, mod = _router()
    fetched: list[str] = []

    def fake_open(req, *, timeout, **_kw):
        tag = (req.get_header("Authorization") or "").rsplit("-", 1)[-1]
        fetched.append(req.full_url)
        return _Resp({"data": [{"id": f"model-{tag}", "router": {"capabilities": {"reasoning": {
            "supported": True, "efforts": [{"value": "low" if tag == "A" else "high"}]}}}}]})

    monkeypatch.setattr(urllib_security, "open_credentialed_url", fake_open)
    monkeypatch.setattr(mod, "_efforts_cache", None)
    monkeypatch.setattr(mod, "_disk_checked", False)
    monkeypatch.setattr(mod, "_warm_started", False)

    with scoped(a):
        assert mod._base_url() == "https://a.router.test/v1"
        profile.fetch_models()
        assert profile.supported_reasoning_efforts("model-A") == ("low",)
    with scoped(b):
        assert mod._base_url() == "https://b.router.test/v1"
        # B never fetched: A's verdicts must not be visible, and B's disk mirror (absent) is what it reads.
        assert profile.supported_reasoning_efforts("model-A") is None
        profile.fetch_models()
        assert profile.supported_reasoning_efforts("model-B") == ("high",)
    with scoped(a):
        assert profile.supported_reasoning_efforts("model-A") == ("low",)
    # Unscoped: the module slot is untouched by the scoped fetches.
    assert mod._efforts_cache is None

    # Warm thread launched from B's turn fetches with B's key/base URL. pytest re-sets
    # PYTEST_CURRENT_TEST per phase; the warmer's pytest guard reads it at call time.
    fetched.clear()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with scoped(b):
        mod._warm_efforts_async()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not fetched:
        time.sleep(0.02)
    assert fetched and all(url.startswith("https://b.router.test/") for url in fetched)


def test_credentialed_catalog_probe_failure_is_not_cached_across_keys(monkeypatch):
    """A 401 under one key must not pin a sibling profile (same base URL, valid key) to the empty
    catalog for the TTL."""
    import requests

    import plugins.image_gen.openrouter as orp

    def fake_get(url, headers=None, timeout=None, **_kw):
        if (headers or {}).get("Authorization") == "Bearer good-key":
            return _Resp({"data": [{"id": "google/gemini-image"}]})
        return _Resp({"error": "unauthorized"}, status=401)

    monkeypatch.setattr(requests, "get", fake_get)
    orp._CATALOG_CACHE.clear()
    try:
        assert orp._fetch_image_api_catalog("https://openrouter.ai/api/v1", "bad-key") == frozenset()
        assert "google/gemini-image" in orp._fetch_image_api_catalog("https://openrouter.ai/api/v1", "good-key")
    finally:
        orp._CATALOG_CACHE.clear()


def test_memory_provider_skill_prune_only_touches_the_active_home(homes, monkeypatch):
    """Pruning under profile B (whose active provider differs) must leave profile A's registered
    provider skill in place; A's own later prune still retracts it."""
    import plugins.memory as mem
    from hermes_cli.plugins import _reset_plugin_managers_for_tests, get_plugin_manager

    a, b = homes
    _reset_plugin_managers_for_tests()
    mem._REGISTERED_MEMORY_PROVIDER_SKILLS.clear()
    skill_dir = a / "plugins" / "provA" / "skills" / "maint"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: maint\ndescription: x\n---\nbody\n", encoding="utf-8")
    try:
        with scoped(a):
            mem._ProviderCollector("provA").register_skill("maint", skill_dir)
            assert get_plugin_manager().find_plugin_skill("provA:maint") is not None
        with scoped(b):
            mem._prune_inactive_memory_provider_skills("provB")
        with scoped(a):
            assert get_plugin_manager().find_plugin_skill("provA:maint") is not None
            mem._prune_inactive_memory_provider_skills("provOther")
            assert get_plugin_manager().find_plugin_skill("provA:maint") is None
    finally:
        mem._REGISTERED_MEMORY_PROVIDER_SKILLS.clear()
        _reset_plugin_managers_for_tests()


def test_openviking_atexit_commits_every_profile_provider(homes):
    """Two profiles' providers initialized in one process both get the atexit commit."""
    import plugins.memory.openviking as ov

    a, b = homes
    committed: list[object] = []
    providers = []
    try:
        for home in (a, b):
            with scoped(home):
                provider = ov.OpenVikingMemoryProvider()
                provider.initialize(session_id=f"s-{home.name}", hermes_home=str(home))
                provider.on_session_end = lambda _msgs, _p=provider: committed.append(_p)
                providers.append(provider)
        ov._atexit_commit_sessions()
        assert committed == providers
    finally:
        for provider in providers:
            with contextlib.suppress(Exception):
                provider._release_run_lock()


def test_disk_cleanup_protected_cron_paths_follow_the_active_home(homes):
    """The protected-path guard must protect the ACTIVE profile's cron dir, not the first one asked."""
    spec = importlib.util.spec_from_file_location(
        "disk_cleanup_mux_scope", REPO / "plugins" / "disk-cleanup" / "disk_cleanup.py")
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    a, b = homes
    for home in (a, b):
        (home / "cron").mkdir()
    with scoped(a):
        assert dc._is_protected_cron_path(a / "cron")
    with scoped(b):
        assert dc._is_protected_cron_path(b / "cron")
        assert not dc._is_protected_cron_path(a / "cron")


def test_yuanbao_active_adapter_resolves_per_profile(homes, monkeypatch):
    """Each profile's turn reads back its own adapter; the unscoped slot still serves single-profile."""
    from gateway.config import PlatformConfig
    from gateway.platforms.yuanbao import YuanbaoAdapter

    a, b = homes
    cfg = PlatformConfig(enabled=True, extra={"app_id": "x", "app_secret": "y"})
    monkeypatch.setattr(YuanbaoAdapter, "_active_instance", None)
    monkeypatch.setattr(YuanbaoAdapter, "_active_instances", {})
    with scoped(a):
        adapter_a = YuanbaoAdapter(cfg)
        YuanbaoAdapter.set_active(adapter_a)
    with scoped(b):
        adapter_b = YuanbaoAdapter(cfg)
        YuanbaoAdapter.set_active(adapter_b)
    with scoped(a):
        assert YuanbaoAdapter.get_active() is adapter_a
    with scoped(b):
        assert YuanbaoAdapter.get_active() is adapter_b
    assert YuanbaoAdapter.get_active() is None  # scoped adapters never claim the unscoped slot
    unscoped = YuanbaoAdapter(cfg)
    YuanbaoAdapter.set_active(unscoped)
    assert YuanbaoAdapter.get_active() is unscoped


def test_honcho_loopback_flow_status_is_per_profile(homes, monkeypatch):
    """Profile B's connect must not be refused as 'pending' because profile A's flow is running."""
    import plugins.memory.honcho.oauth_flow as flow

    a, b = homes
    gate = threading.Event()
    started: list[Path] = []

    def fake_authorize(**kwargs):
        started.append(kwargs["config_path"])
        gate.wait(5)

    monkeypatch.setattr(flow, "authorize_via_loopback", fake_authorize)
    monkeypatch.setattr(flow, "_status", flow.FlowStatus())
    monkeypatch.setattr(flow, "_flow_thread", None)
    for home in (a, b):
        (home / "honcho.json").write_text("{}", encoding="utf-8")
    try:
        with scoped(a):
            assert flow.start_loopback_flow_background()["state"] == "pending"
        with scoped(b):
            assert flow.get_flow_status()["state"] == "idle"
            assert flow.start_loopback_flow_background()["state"] == "pending"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(started) < 2:
            time.sleep(0.02)
        assert sorted(started) == sorted([a / "honcho.json", b / "honcho.json"])
    finally:
        gate.set()
        getattr(flow, "_flows_by_target", {}).clear()
