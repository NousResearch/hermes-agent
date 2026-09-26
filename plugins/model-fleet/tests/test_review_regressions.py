"""Regression tests for the review findings on PR #123636.

Each test names the finding it pins, so a future refactor that reintroduces the bug
fails with a comment saying which one came back.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hermes_cli").is_dir() and (parent / "cron").is_dir():
            import sys
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break
    spec = importlib.util.spec_from_file_location(
        "model_fleet_review", PLUGIN_DIR / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Finding: profile_blocklist ignored when profile_allowlist is non-empty ------


def test_blocklist_wins_over_allowlist():
    """allowlist: [alpha, beta] + blocklist: [beta] must NOT select beta."""
    mod = _load()
    settings = {"profile_allowlist": ["alpha", "beta"], "profile_blocklist": ["beta"]}
    assert mod._profile_selected("alpha", settings) is True
    assert mod._profile_selected("beta", settings) is False, "blocklist was ignored"
    assert mod._profile_selected("gamma", settings) is False


def test_allowlist_alone_still_narrows():
    mod = _load()
    settings = {"profile_allowlist": ["alpha"], "profile_blocklist": []}
    assert mod._profile_selected("alpha", settings) is True
    assert mod._profile_selected("beta", settings) is False


# --- Finding: numbered pick resolves from a stale cache after allowlist change ---


def test_numeric_pick_does_not_use_stale_cache_after_filter_change():
    """A number typed after the rows changed must index the CURRENT filtered list."""
    mod = _load()
    live_rows = [{"slug": "nous", "models": ["a1"]}, {"slug": "z-ai", "models": ["z1"]}]
    # Cache was populated when BOTH providers were listed.
    mod._CACHE.clear()
    mod._CACHE.update({"providers": list(live_rows), "at": time.time()})
    # Now the allowlist hides "nous"; the live rows shrink to one entry.
    filtered = [{"slug": "z-ai", "models": ["z1"]}]

    picked = mod._match_provider(filtered, "1")
    assert picked is not None
    assert picked["slug"] == "z-ai", (
        f"numbered pick returned {picked['slug']!r} from a stale cache; "
        "the allowlist change was bypassed")


def test_expired_cache_is_not_used():
    """A cache older than the TTL must not resolve a numbered pick."""
    mod = _load()
    rows = [{"slug": "nous", "models": ["a1"]}]
    mod._CACHE.clear()
    mod._CACHE.update({"providers": [dict(r) for r in rows],
                       "at": time.time() - mod.CACHE_TTL_SECONDS - 60})
    assert mod._cache_fresh("providers", rows) is False
    picked = mod._match_provider(rows, "1")
    assert picked is not None and picked["slug"] == "nous"  # resolved from live rows


def test_model_pick_revalidates_cache_against_live_list():
    mod = _load()
    row = {"slug": "nous", "models": ["m1", "m2"]}
    mod._CACHE.clear()
    # Cache claims a DIFFERENT model list for the same provider.
    mod._CACHE.update({"models": ["mX"], "provider": "nous", "at": time.time()})
    assert mod._match_model(row, "1") == "m1", "resolved from a divergent cache"


# --- Finding: cron load/mutate/save split across two lock acquisitions ----------


def test_cron_write_happens_inside_the_jobs_lock(tmp_path, monkeypatch):
    """save_jobs must be called while the same lock that load_jobs ran under is held."""
    mod = _load()
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(mod, "_install_root", lambda: home)
    (home / "cron").mkdir(parents=True)
    # jobs.json is an object with a "jobs" list, not a bare list.
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "j1", "provider": "old", "model": "old", "no_agent": False,
         "enabled": True, "schedule": {"kind": "every", "every": 3600}},
    ]}), encoding="utf-8")

    state = {"depth": 0, "max_depth": 0, "saved": 0, "loaded": 0}
    import cron.jobs as cj

    real_lock_factory = cj._jobs_lock

    class TracingLock:
        """Wraps the real lock without pre-acquiring it (the factory is stateful:
        _jobs_lock() tracks depth in a thread-local, so nesting two real contexts
        would desync the counter and the write would land nowhere)."""

        def __enter__(self):
            self._real = real_lock_factory()
            entered = self._real.__enter__()
            state["depth"] += 1
            state["max_depth"] = max(state["max_depth"], state["depth"])
            return entered

        def __exit__(self, *a):
            state["depth"] -= 1
            return self._real.__exit__(*a)

    monkeypatch.setattr(cj, "_jobs_lock", TracingLock)
    orig_save = cj.save_jobs
    orig_load = cj.load_jobs

    def save_spy(jobs, *a, **k):
        # Recording the lock depth at save time is the assertion: it must be > 0.
        state["saved"] += 1
        assert state["depth"] > 0, "save_jobs ran OUTSIDE the jobs lock"
        return orig_save(jobs, *a, **k)

    def load_spy(*a, **k):
        state["loaded"] += 1
        return orig_load(*a, **k)

    monkeypatch.setattr(cj, "save_jobs", save_spy)
    monkeypatch.setattr(cj, "load_jobs", load_spy)

    settings = {"include_profiles": False, "include_cron": True, "backup": True}
    mod._apply_crons("new", "new-model", settings, "STAMP", dry_run=False)

    assert state["saved"] == 1, "no save happened; the test did not exercise the path"
    assert state["max_depth"] >= 1
    # The scheduler-owned fields must survive: a last-writer-wins save of a stale
    # snapshot would have reverted next_run_at / fire_claim set by a tick mid-flight.
    out = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    job = out["jobs"][0]
    assert job["model"] == "new-model" and job["provider"] == "new"
    assert job["enabled"] is True


# --- Finding: fleet enumeration anchors to the caller's scoped profile home ------


def test_profile_homes_anchor_to_installation_root_not_scoped_profile(monkeypatch, tmp_path):
    """Invoked from a named profile, enumeration must still find the real root."""
    mod = _load()
    root = tmp_path / "root"
    (root / "profiles" / "alpha").mkdir(parents=True)
    (root / "profiles" / "beta").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    for name in ("alpha", "beta"):
        (root / "profiles" / name / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    # Pretend the call is scoped to the "alpha" profile.
    monkeypatch.setattr(mod, "_hermes_home", lambda: root / "profiles" / "alpha")

    settings = {"include_profiles": True, "profile_allowlist": [], "profile_blocklist": []}
    homes = dict(mod._profile_homes(settings))
    labels = set(homes)
    assert "default" in labels, f"real default home missing; got {labels}"
    assert {"alpha", "beta"} <= labels, f"sibling profiles missing; got {labels}"
    # The active config was NOT mistaken for the "default" profile's config.
    assert homes["default"] == root


# --- Finding: backup must be taken before the first write, once per target ------


def test_backup_of_active_config_holds_prechange_content(tmp_path, monkeypatch):
    mod = _load()
    from hermes_cli.model_switch import ModelSwitchResult
    monkeypatch.setattr(mod, "_switch_result",
                        lambda p, mo: ModelSwitchResult(
                            success=True, new_model=mo, target_provider=p,
                            provider_changed=True, base_url="https://s.invalid", api_mode="chat"))
    home = tmp_path / "h"
    home.mkdir(parents=True)
    original = "model:\n  provider: nous\n  default: alpha-base\n"
    (home / "config.yaml").write_text(original, encoding="utf-8")
    monkeypatch.setattr(mod, "_hermes_home", lambda: home)
    monkeypatch.setattr(mod, "_install_root", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(mod, "_settings", lambda: {
        "include_profiles": False, "include_cron": False, "include_auxiliary": False,
        "profile_allowlist": [], "profile_blocklist": [], "model_allowlist": [], "backup": True})
    # Keep the REAL _apply_profiles: the clobber happened there. Only stub the
    # per-profile route resolution (network), which returns this profile's own result.
    from hermes_cli.model_switch import ModelSwitchResult
    _r = ModelSwitchResult(success=True, new_model="claude-two", target_provider="anthropic",
                           base_url="https://s.invalid", api_mode="chat")
    monkeypatch.setattr(mod, "_resolve_in_profile",
                        lambda h, p, mo: (_r, {"provider": p, "default": mo}))

    out = mod._apply_sync("anthropic", "claude-two", dry_run=False, with_auxiliary=False)
    assert "aborted" not in out.lower(), out
    backups = list(home.glob("*.bak-model-fleet-*"))
    assert backups, "no backup written"
    assert original in backups[0].read_text(encoding="utf-8"), (
        "backup holds POST-change content — restore would be a silent no-op")


def test_backup_failure_aborts_instead_of_writing(tmp_path, monkeypatch):
    """With backup: true and a failing copy, the switch must refuse."""
    mod = _load()
    from hermes_cli.model_switch import ModelSwitchResult
    monkeypatch.setattr(mod, "_switch_result",
                        lambda p, mo: ModelSwitchResult(success=True, new_model=mo,
                                                        target_provider=p))
    home = tmp_path / "h"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model:\n  default: keepme\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_hermes_home", lambda: home)
    monkeypatch.setattr(mod, "_install_root", lambda: home)
    monkeypatch.setattr(mod, "_settings", lambda: {
        "include_profiles": False, "include_cron": False, "include_auxiliary": False,
        "profile_allowlist": [], "profile_blocklist": [], "model_allowlist": [], "backup": True})
    monkeypatch.setattr(mod, "_backup", lambda p, s: None)  # simulate failure

    out = mod._apply_sync("anthropic", "claude-two", dry_run=False, with_auxiliary=False)
    assert "aborted" in out.lower(), out
    assert "keepme" in (home / "config.yaml").read_text(encoding="utf-8"), \
        "config was rewritten despite the aborted backup"


# --- Finding: per-profile resolution reuses another profile's route --------------


def test_profiles_skip_when_target_does_not_resolve_there(tmp_path, monkeypatch):
    """A profile where the target does not resolve must be skipped, not given
    another profile's endpoint."""
    mod = _load()
    settings = {"include_profiles": True, "include_cron": False, "backup": True,
                "profile_allowlist": [], "profile_blocklist": []}
    root = tmp_path / "root"
    other = root / "profiles" / "alpha"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_profile_homes", lambda s: [("alpha", other)])

    from hermes_cli.model_switch import ModelSwitchResult
    caller_result = ModelSwitchResult(success=True, new_model="m", target_provider="p",
                                      base_url="https://CALLER-ENDPOINT", api_mode="chat")
    # This profile has no credentials / no such provider: resolution must fail.
    monkeypatch.setattr(mod, "_resolve_in_profile", lambda h, p, mo: None)

    changed, backups = mod._apply_profiles("p", "m", caller_result, settings, "STAMP", False)
    assert any("SKIPPED" in c for c in changed), changed
    text = (other / "config.yaml").read_text(encoding="utf-8")
    assert "CALLER-ENDPOINT" not in text, "wrote the caller's endpoint into a sibling profile"
    assert backups == []
