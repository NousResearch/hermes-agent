"""Multiplexed gateway: hermes_cli's process-wide caches must not hand profile A's value to profile B.

Every test builds two real profile homes (config.yaml / .env / cache files that differ), warms a
cache under ``set_hermes_home_override(A)`` and reads under B. Only the HTTP transport is canned —
its payload depends on the Authorization header or URL so a leaked entry is observable.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time

import httpx
import pytest

from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _json_resp(payload) -> _Resp:
    return _Resp(json.dumps(payload).encode())


class _Scoped:
    """Run a block as one profile's multiplexed turn (home override + its .env secret scope)."""

    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self._t = set_hermes_home_override(self.home)
        self._s = set_secret_scope(build_profile_secret_scope(self.home))
        return self

    def __exit__(self, *a):
        reset_secret_scope(self._s)
        reset_hermes_home_override(self._t)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    a = tmp_path / "hermes"
    b = a / "profiles" / "B"
    for home in (a, b):
        (home / "cache").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(a))
    for var in ("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL", "NOUS_INFERENCE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return a, b


def test_deepinfra_catalog_is_fetched_with_each_profiles_key(homes, monkeypatch):
    a, b = homes
    (a / ".env").write_text("DEEPINFRA_API_KEY=key-A\n", encoding="utf-8")
    (b / ".env").write_text("DEEPINFRA_API_KEY=key-B\n", encoding="utf-8")
    from application_deepinfra_catalog import deepinfra_model_ids
    from models.catalog_deepinfra import reset_catalog_cache
    from providers import get_provider_profile

    reset_catalog_cache()

    def fetch_catalog(*, api_key="", base_url="", timeout=5.0):
        who = api_key.rsplit("-", 1)[-1] or "anon"
        return [{"id": f"di/model-{who}", "metadata": {"tags": ["chat"]}}]

    monkeypatch.setattr(get_provider_profile("deepinfra"), "fetch_catalog", fetch_catalog)
    with _Scoped(a):
        assert deepinfra_model_ids("chat") == ["di/model-A"]
    with _Scoped(b):
        assert deepinfra_model_ids("chat") == ["di/model-B"]


def test_copilot_context_cache_hit_requires_same_api_key(homes, monkeypatch):
    import models.catalog_github as catalog
    import models.metadata.github as github

    catalog.reset_github_model_catalog_cache()
    github.reset_github_context_cache()

    def transport(_url, *, timeout, headers):
        limit = 111 if headers.get("Authorization", "").endswith("copilot-A") else 222
        return {"data": [{"id": "gpt-x", "model_picker_enabled": True,
                          "supported_endpoints": ["/chat/completions"],
                          "capabilities": {"type": "chat", "limits": {"max_prompt_tokens": limit}}}]}

    monkeypatch.setattr(catalog, "_fetch_json", transport)
    assert github.github_model_context_length("gpt-x", api_key="copilot-A") == 111
    assert github.github_model_context_length("gpt-x", api_key="copilot-B") == 222
    assert github.github_model_context_length("gpt-x", api_key="copilot-A") == 111


def test_nous_reasoning_caps_follow_each_profiles_portal(homes, monkeypatch):
    a, b = homes
    (a / ".env").write_text("NOUS_INFERENCE_BASE_URL=https://portal-a.example/v1\n", encoding="utf-8")
    (b / ".env").write_text("NOUS_INFERENCE_BASE_URL=https://portal-b.example/v1\n", encoding="utf-8")
    import hermes_cli.models as models
    import models.metadata.reasoning as caps

    for attr, value in (("_nous_reasoning_caps_cache", None), ("_nous_reasoning_caps_failed_at", None),
                        ("_nous_caps_disk_checked", False), ("_nous_caps_warm_started", False)):
        monkeypatch.setattr(caps, attr, value)

    def transport(req, *, timeout, **kw):
        effort = "low" if "portal-a" in req.full_url else "high"
        return _json_resp({"data": [{"id": "nous/m", "supported_parameters": ["reasoning"],
                                     "reasoning": {"supported_efforts": [effort]}}]})

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", transport)
    with _Scoped(a):
        assert caps.nous_model_reasoning_capabilities("nous/m", allow_fetch=True).supported_efforts == ("low",)
    with _Scoped(b):
        assert caps.nous_model_reasoning_capabilities("nous/m", allow_fetch=True).supported_efforts == ("high",)


def test_swr_refresh_runs_as_the_profile_that_spawned_it(homes):
    a, b = homes
    import hermes_cli.models as models

    seen: dict[str, str] = {}
    done = threading.Event()

    def refresh():
        seen["home"] = str(get_hermes_home())
        done.set()
        return {"fp": "fp", "at": time.time(), "models": ["m"]}

    with _Scoped(b):
        models._spawn_swr_refresh("custom:https://gw.example/v1#fp", refresh)
    assert done.wait(5)
    assert seen["home"] == str(b)
    deadline = time.monotonic() + 5
    while not (b / "provider_models_cache.json").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (b / "provider_models_cache.json").exists()
    assert not (a / "provider_models_cache.json").exists()


def _write_manifest(home, model_id: str, mtime: float) -> None:
    path = home / "cache" / "model_catalog.json"
    path.write_text(json.dumps({"version": 1, "providers": {"openrouter": {"models": [
        {"id": model_id, "description": "x", "default": True}]}}}), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_model_catalog_in_process_copy_is_bound_to_its_cache_file(homes, monkeypatch):
    a, b = homes
    from models import catalog_runtime as mc
    from models.catalog_manifest import catalog_settings

    for home in (a, b):
        (home / "config.yaml").write_text("model_catalog:\n  ttl_minutes: 600\n", encoding="utf-8")
    same_mtime = time.time() - 5  # identical mtimes: only the path can tell the two files apart
    _write_manifest(a, "vendor/a-model", same_mtime)
    _write_manifest(b, "vendor/b-model", same_mtime)
    mc.reset_cache()
    settings = catalog_settings({})
    catalog_path_b = b / "cache" / "model_catalog.json"
    with _Scoped(a):
        assert [m["id"] for m in mc.get_catalog(settings, a / "cache" / "model_catalog.json")["providers"]["openrouter"]["models"]] == ["vendor/a-model"]
    with _Scoped(b):
        assert [m["id"] for m in mc.get_catalog(settings, b / "cache" / "model_catalog.json")["providers"]["openrouter"]["models"]] == ["vendor/b-model"]
        assert mc.cached_default_model(catalog_path_b, "openrouter") == "vendor/b-model"


def test_openrouter_curated_list_is_per_profile(homes, monkeypatch):
    a, b = homes
    import hermes_cli.models as models

    for home in (a, b):
        (home / "config.yaml").write_text("model_catalog:\n  ttl_minutes: 600\n", encoding="utf-8")
    now = time.time()
    for home, mid in ((a, "vendor/a-model"), (b, "vendor/b-model")):
        (home / "cache" / "openrouter_curated_catalog.json").write_text(
            json.dumps({"fetched_at": now, "curated": [[mid, "free"]]}), encoding="utf-8")
    monkeypatch.setattr(models, "_openrouter_catalog_cache", None)
    with _Scoped(a):
        assert [m for m, _ in models.fetch_openrouter_models()] == ["vendor/a-model"]
    with _Scoped(b):
        assert [m for m, _ in models.fetch_openrouter_models()] == ["vendor/b-model"]


def test_banner_skills_are_the_routed_profiles(homes):
    a, b = homes
    import hermes_cli.banner as banner

    for home, tag in ((a, "a"), (b, "b")):
        skill = home / "skills" / f"skill_{tag}"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: skill_{tag}\ndescription: {tag}\n---\nbody\n", encoding="utf-8")
    banner._available_skills_cache = None
    try:
        with _Scoped(a):
            assert sorted(sum(banner.get_available_skills().values(), [])) == ["skill_a"]
        with _Scoped(b):
            assert sorted(sum(banner.get_available_skills().values(), [])) == ["skill_b"]
    finally:
        banner._available_skills_cache = None


def test_failed_guest_mint_only_suppresses_that_profile(homes, monkeypatch, tmp_path):
    a, b = homes
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    import hermes_cli.anon_auth as anon
    import hermes_cli.auth_nous as auth_nous

    anon.reset_mint_memo_for_tests()
    status = {"code": 429}
    attempts: list[str] = []

    def client(timeout_seconds, verify):
        def handler(request):
            attempts.append(str(get_hermes_home()))
            if status["code"] == 429:
                return httpx.Response(429, json={"error": "rate"})
            return httpx.Response(201, json={"token": "anon_b", "user_id": "u", "org_id": "o"})
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(auth_nous, "_nous_http_client", client)
    with _Scoped(a), pytest.raises(Exception):
        anon.ensure_portal_identity(explicit=True, timeout_seconds=1)
    assert attempts == [str(a)]
    status["code"] = 201
    with _Scoped(b):
        assert anon.ensure_portal_identity(explicit=True, timeout_seconds=1) is not None
    assert attempts[-1] == str(b)


def test_active_skin_is_per_profile_and_leaves_launch_slot_alone(homes):
    a, b = homes
    from hermes_cli import skin_engine

    (a / "config.yaml").write_text("display:\n  skin: ares\n", encoding="utf-8")
    (b / "config.yaml").write_text("display:\n  skin: mono\n", encoding="utf-8")
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    getattr(skin_engine, "_active_skin_by_home", {}).clear()
    try:
        with _Scoped(a):
            skin_engine.init_skin_from_config({"display": {"skin": "ares"}})
            assert skin_engine.get_active_skin().name == "ares"
        with _Scoped(b):
            assert skin_engine.get_active_skin().name == "mono"  # B's own display.skin, never A's
        with _Scoped(a):
            assert skin_engine.get_active_skin().name == "ares"
        assert skin_engine.get_active_skin_name() == "default"  # routed turns never touch the launch slot
    finally:
        skin_engine._active_skin = None
        skin_engine._active_skin_name = "default"
        getattr(skin_engine, "_active_skin_by_home", {}).clear()
