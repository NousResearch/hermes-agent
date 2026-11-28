"""Tests for foreground default-plugins preflight (ensure, settled hash, profile create, offline)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
import hermes_cli.plugin_catalog as pc
import pm.plugins_state
from hermes_cli import default_plugins
from hermes_cli.plugin_catalog import entry_from_mapping


def _entry_dict(name: str, *, tier: str = "official", sha: str = "a" * 40, default: bool = True) -> dict:
    return {
        "name": name,
        "repo": "https://github.com/NousResearch/hermes-official-plugins",
        "sha": sha,
        "description": f"{name} desc",
        "maintainer": "NousResearch",
        "tier": tier,
        "default": default,
    }


def _entry(name: str, *, tier: str = "official", sha: str = "a" * 40, default: bool = True):
    return entry_from_mapping(_entry_dict(name, tier=tier, sha=sha, default=default), name)


def _setup_env(tmp_path: Path, monkeypatch, entries_ref: list, live_ref: list):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-ensure-1")
    monkeypatch.setattr(default_plugins, "default_entries", lambda: list(entries_ref))
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **kwargs: live_ref[0])
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    return home


def _fake_install_into(calls: list[tuple[Path, str]], *, fail: bool = False, noisy: bool = False):
    def install_into(home: Path, consent=None):
        def install(name: str, discard=None, on_step=None):
            if noisy:
                print(f"noisy stdout from installing {name}")
            calls.append((home, name))
            if fail:
                return {"ok": False, "error": "fatal: unable to access 'https://github.com/'"}
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return {"ok": True}
        return install
    return install_into


class _FinishedChild:
    pid = 0

    def __init__(self, code: int) -> None:
        self.returncode = code

    def poll(self) -> int:
        return self.returncode


def _inline_installer_child(monkeypatch) -> None:
    """Run the ensure child's own work in this process, so the fakes patched here apply. The real
    process boundary (detached child, result file, timeout hand-off) is covered in
    test_default_plugins_timeout.py."""
    monkeypatch.setattr(default_plugins, "_launch_ensure_child", lambda run_id, home: _FinishedChild(
        default_plugins._ensure_child_work(run_id, home, say=print)))


@pytest.fixture(autouse=True)
def inline_installer_child(monkeypatch):
    _inline_installer_child(monkeypatch)


def _parse_single_json_line(stdout: str) -> dict:
    lines = stdout.splitlines()
    assert len(lines) == 1, f"expected exactly 1 JSON line on stdout, got {len(lines)}: {stdout!r}"
    return json.loads(lines[0])


def test_ensure_fast_path_does_no_install_and_no_lock(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    # First pass installs and settles.
    assert default_plugins.main(["ensure", "--json"]) == 0
    first = _parse_single_json_line(capsys.readouterr().out)
    assert first["state"] == "installed"
    assert first["installed"] == ["hermes-security"]
    assert calls == [(home, "hermes-security")]

    # Second pass must take the fast path: no lock and no install.
    def fail_lock(_wait):
        raise AssertionError("installer_lock must not be acquired on fast path")

    def fail_install(_home, consent=None):
        raise AssertionError("_install_into must not be called on fast path")

    monkeypatch.setattr(default_plugins, "installer_lock", fail_lock)
    monkeypatch.setattr(lcm, "_install_into", fail_install)

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0
    assert payload["state"] == "nothing-pending"
    assert payload["pending_before"] == []
    assert payload["installed"] == []
    assert payload["owed"] == []
    assert isinstance(payload["elapsed_ms"], int) and payload["elapsed_ms"] >= 0


def test_ensure_pending_installs_records_and_user_removal_sticks(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0
    assert payload["state"] == "installed"
    assert payload["pending_before"] == ["hermes-security"]
    assert payload["installed"] == ["hermes-security"]
    assert payload["owed"] == []
    assert (home / "plugins" / "hermes-security").is_dir()
    assert "hermes-security" in default_plugins._recorded(home)

    # User removes the default plugin; it must never be reinstalled.
    (home / "plugins" / "hermes-security").rmdir()
    code2 = default_plugins.main(["ensure", "--json"])
    payload2 = _parse_single_json_line(capsys.readouterr().out)
    assert code2 == 0
    assert payload2["state"] == "nothing-pending"
    assert payload2["installed"] == []
    assert not (home / "plugins" / "hermes-security").exists()
    assert calls == [(home, "hermes-security")]


def test_ensure_deadline_returns_3_and_leaves_the_child_running(tmp_path, monkeypatch, capsys):
    """At the deadline ensure returns 3 without starting a second installer: the child it started keeps
    going."""
    entries = [_entry("langfuse")]
    live = [{"entries": [_entry_dict("langfuse")]}]
    _setup_env(tmp_path, monkeypatch, entries, live)
    launched: list = []

    class _RunningChild:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(default_plugins, "_launch_ensure_child",
                        lambda run_id, home: launched.append((run_id, home)) or _RunningChild())
    monkeypatch.setattr(default_plugins, "spawn_install", lambda home=None: pytest.fail("no continuation spawn"))

    code = default_plugins.main(["ensure", "--json", "--timeout", "0.05"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 3
    assert payload["state"] == "timeout"
    assert payload["pending_before"] == ["langfuse"]
    assert payload["owed"] == ["langfuse"]
    assert len(launched) == 1 and launched[0][1] is None


def test_ensure_policy_denied_returns_4(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")

    monkeypatch.setattr(default_plugins, "installer_lock", lambda _w: (_ for _ in ()).throw(AssertionError("lock")))
    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("install")))

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 4
    assert payload["state"] == "skipped"
    assert payload["pending_before"] == []
    assert payload["installed"] == []
    assert payload["owed"] == []

    # Also test env opt-out with a fresh allowed config.
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    code2 = default_plugins.main(["ensure", "--json"])
    payload2 = _parse_single_json_line(capsys.readouterr().out)
    assert code2 == 4
    assert payload2["state"] == "skipped"


def test_ensure_owed_returns_2_on_install_failure(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls, fail=True))

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 2
    assert payload["state"] == "owed"
    assert payload["pending_before"] == ["hermes-security"]
    assert payload["installed"] == []
    assert "hermes-security" in payload["owed"]


def test_new_default_in_catalog_after_settled_unsettles_and_installs(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    assert default_plugins.main(["ensure", "--json"]) == 0
    _parse_single_json_line(capsys.readouterr().out)
    assert default_plugins.step_install_default_plugins() == {"ok": True}

    settled_path = home / "cache" / "default-plugins.settled"
    settled_snapshot = settled_path.read_text(encoding="utf-8")

    # Live catalog adds a new default without any app identity change.
    entries.append(_entry("langfuse", sha="b" * 40))
    live[0] = {"entries": [_entry_dict("hermes-security"), _entry_dict("langfuse", sha="b" * 40)]}

    # step_install_default_plugins treats the hash mismatch as unsettled.
    spawned: list = []
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: spawned.append(h) or True)
    step_res = default_plugins.step_install_default_plugins()
    assert step_res.get("owed") is True
    assert len(spawned) == 1

    # Restore the old settled file to verify ensure's fast path also treats it as unsettled.
    settled_path.write_text(settled_snapshot, encoding="utf-8")
    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0
    assert payload["state"] == "installed"
    assert payload["pending_before"] == ["langfuse"]
    assert payload["installed"] == ["langfuse"]
    assert (home / "plugins" / "langfuse").is_dir()

    # After installing, both step_install_default_plugins and ensure are settled again.
    assert default_plugins.step_install_default_plugins() == {"ok": True}
    monkeypatch.setattr(default_plugins, "installer_lock", lambda _w: (_ for _ in ()).throw(AssertionError("lock")))
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert _parse_single_json_line(capsys.readouterr().out)["state"] == "nothing-pending"


def test_legacy_identity_only_settled_file_is_unsettled_safely(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    # Simulate an older Hermes install: plugin already recorded, legacy plain-text settled file.
    default_plugins._record(home, {"hermes-security"})
    settled_path = home / "cache" / "default-plugins.settled"
    settled_path.parent.mkdir(parents=True, exist_ok=True)
    settled_path.write_text("rev-ensure-1", encoding="utf-8")

    # step_install_default_plugins treats identity-only settled file as unsettled (no crash).
    spawned: list = []
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: spawned.append(h) or True)
    assert default_plugins.step_install_default_plugins().get("owed") is True

    # Restore legacy file and run ensure: one extra pass upgrades the marker without reinstalling.
    settled_path.write_text("rev-ensure-1", encoding="utf-8")
    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0
    assert payload["state"] == "nothing-pending"
    assert calls == []

    # Marker is now upgraded: next ensure takes the fast path without lock.
    monkeypatch.setattr(default_plugins, "installer_lock", lambda _w: (_ for _ in ()).throw(AssertionError("lock")))
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert _parse_single_json_line(capsys.readouterr().out)["state"] == "nothing-pending"


def test_bare_main_keeps_detached_child_behavior(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(default_plugins, "_run_child", lambda: called.append("child") or 0)
    monkeypatch.setattr(default_plugins, "ensure_defaults", lambda **_k: (_ for _ in ()).throw(AssertionError("ensure")))
    assert default_plugins.main([]) == 0
    assert called == ["child"]


def test_ensure_json_stdout_clean_when_installer_prints(tmp_path, monkeypatch, capsys):
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls, noisy=True))

    code = default_plugins.main(["ensure", "--json"])
    captured = capsys.readouterr()
    payload = _parse_single_json_line(captured.out)
    assert code == 0
    assert payload["state"] == "installed"
    assert "noisy stdout from installing hermes-security" in captured.err


def test_profile_provision_runs_foreground_ensure_for_new_home(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    monkeypatch.setattr(default_plugins, "default_entries", lambda: [_entry("hermes-security")])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **kwargs: {"entries": [_entry_dict("hermes-security")]})

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))
    monkeypatch.setattr(default_plugins, "spawn_install", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("spawned")))

    result = provision_profile(ProfileCreateRequest(name="work", alias=False, mirror_credentials=False))
    assert result.path == home / "profiles" / "work"
    assert (result.path / "plugins" / "hermes-security").is_dir()
    assert calls == [(result.path, "hermes-security")]
    assert result.warnings == []


def test_profile_provision_timeout_warns_and_leaves_the_child_running(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "PROFILE_CREATE_ENSURE_TIMEOUT", 0.05)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    (home / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    monkeypatch.setattr(default_plugins, "default_entries", lambda: [_entry("langfuse")])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **kwargs: {"entries": [_entry_dict("langfuse")]})

    launched: list = []

    class _RunningChild:
        pid = 4243
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(default_plugins, "_launch_ensure_child",
                        lambda run_id, home: launched.append(home) or _RunningChild())
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: pytest.fail("no continuation spawn"))

    result = provision_profile(ProfileCreateRequest(name="slowprof", alias=False, mirror_credentials=False))

    assert result.path == home / "profiles" / "slowprof"
    assert launched == [result.path]
    assert any("timed out" in w and "continuing in background" in w for w in result.warnings)


def test_cli_subprocess_ensure_json_with_cached_catalog_and_fast_path(tmp_path):
    """Real subprocess check: `python -m hermes_cli.default_plugins ensure --json` end to end."""
    home = tmp_path / "hermes_home"
    cache = home / "cache"
    cache.mkdir(parents=True)
    (cache / "plugin-catalog.json").write_text(
        json.dumps({
            "generated_at": "2099-01-01T00:00:00+00:00",
            "entries": [_entry_dict("hermes-security", default=False)],
            "removed": [],
        }),
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    # The ensure must run (and settle), not skip under the suite's no-fetch rule.
    env.update(HERMES_HOME=str(home), HOME=str(tmp_path), PYTHONPATH=str(repo), HERMES_DISABLE_LAZY_INSTALLS="0")

    # First run reads fresh local cache and writes settled marker.
    p1 = subprocess.run(
        [sys.executable, "-m", "hermes_cli.default_plugins", "ensure", "--json", "--timeout", "30"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert p1.returncode == 0, p1.stderr
    d1 = _parse_single_json_line(p1.stdout)
    assert d1["state"] == "nothing-pending"
    assert (home / "cache" / "default-plugins.settled").is_file()

    # Second run takes the fast path.
    p2 = subprocess.run(
        [sys.executable, "-m", "hermes_cli.default_plugins", "ensure", "--json", "--timeout", "30"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert p2.returncode == 0, p2.stderr
    d2 = _parse_single_json_line(p2.stdout)
    assert d2["state"] == "nothing-pending"
    assert d2["pending_before"] == []
    assert d2["installed"] == []
    assert d2["owed"] == []

    # Policy opt-out returns exit 4.
    env_denied = {**env, "HERMES_DISABLE_LAZY_INSTALLS": "1"}
    p3 = subprocess.run(
        [sys.executable, "-m", "hermes_cli.default_plugins", "ensure", "--json", "--timeout", "30"],
        cwd=repo, env=env_denied, capture_output=True, text=True, timeout=30, check=False,
    )
    assert p3.returncode == 4, p3.stderr
    assert _parse_single_json_line(p3.stdout)["state"] == "skipped"


def test_fast_settled_hit_invalidates_when_live_cache_file_changes(tmp_path, monkeypatch, capsys):
    """Updating cache/plugin-catalog.json on disk invalidates the zero-parse fast path."""
    home = tmp_path / "home"
    cache = home / "cache"
    cache.mkdir(parents=True)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-fast-1")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path / "empty-in-tree")
    monkeypatch.setattr(pc, "_in_tree_catalog_time", None)

    cache_file = cache / "plugin-catalog.json"
    cache_file.write_text(
        json.dumps({
            "generated_at": "2099-01-01T00:00:00+00:00",
            "entries": [_entry_dict("hermes-security", default=True)],
            "removed": [],
        }),
        encoding="utf-8",
    )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    # Pass 1 installs hermes-security and writes live_cache_sig into default-plugins.settled.
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert _parse_single_json_line(capsys.readouterr().out)["installed"] == ["hermes-security"]
    assert default_plugins._fast_settled_hit(home, [home]) is True

    # Pass 2 hits the zero-parse fast path without touching load_catalog_live or lock.
    monkeypatch.setattr(default_plugins, "installer_lock", lambda _w: (_ for _ in ()).throw(AssertionError("lock")))
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert _parse_single_json_line(capsys.readouterr().out)["state"] == "nothing-pending"

    # Live cache file changes to add a second default -> _fast_settled_hit returns False -> installs it.
    monkeypatch.undo()
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-fast-1")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path / "empty-in-tree")
    monkeypatch.setattr(pc, "_in_tree_catalog_time", None)
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))
    _inline_installer_child(monkeypatch)  # undone with the rest above

    cache_file.write_text(
        json.dumps({
            "generated_at": "2099-01-01T00:00:00+00:00",
            "entries": [
                _entry_dict("hermes-security", default=True),
                _entry_dict("langfuse", sha="b" * 40, default=True),
            ],
            "removed": [],
        }),
        encoding="utf-8",
    )
    assert default_plugins._fast_settled_hit(home, [home]) is False
    assert default_plugins.main(["ensure", "--json"]) == 0
    out = _parse_single_json_line(capsys.readouterr().out)
    assert out["state"] == "installed"
    assert out["installed"] == ["langfuse"]
    assert (home / "plugins" / "langfuse").is_dir()


def test_ensure_offline_fails_open_without_attempting_installs(tmp_path, monkeypatch, capsys):
    """An unreachable live catalog means every git/uv install would time out too: owed at once."""
    entries = [_entry("hermes-security"), _entry("langfuse")]
    live = [None]  # the live catalog read failed
    _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    assert default_plugins.main(["ensure", "--json"]) == 2
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert payload["state"] == "owed" and calls == []
    assert set(payload["owed"]) == {"live catalog", "hermes-security", "langfuse"}

    live[0] = {"entries": [_entry_dict("hermes-security"), _entry_dict("langfuse")]}  # back online
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert sorted(name for _, name in calls) == ["hermes-security", "langfuse"]


def test_ensure_offline_with_expired_stale_cache_fails_open_without_installing(tmp_path, monkeypatch, capsys):
    """Returning user goes offline with an expired (>6h, <24h) stale cache/plugin-catalog.json:
    ensure_defaults delegates catalog discovery to the ensure child (bounding slow HTTP by --timeout),
    the child detects the failed fetch via live_fetch_recently_failed(), and fails open with exit=2
    and zero installs."""
    home = tmp_path / "home"
    cache = home / "cache"
    cache.mkdir(parents=True)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-stale-offline")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path / "empty-in-tree")
    monkeypatch.setattr(pc, "_in_tree_catalog_time", None)
    monkeypatch.setattr(pc, "_live_fetch_failed_until", 0.0)

    cache_file = cache / "plugin-catalog.json"
    cache_file.write_text(
        json.dumps({
            "generated_at": "2026-01-01T00:00:00+00:00",
            "entries": [_entry_dict("hermes-security", default=True)],
            "removed": [],
        }),
        encoding="utf-8",
    )
    # Past LIVE_CATALOG_TTL_SECONDS (6h), within LIVE_CATALOG_MAX_STALE_SECONDS (24h).
    stale_mtime = time.time() - pc.LIVE_CATALOG_TTL_SECONDS - 600
    os.utime(cache_file, (stale_mtime, stale_mtime))

    monkeypatch.setattr(
        pc,
        "_download_live_catalog",
        lambda _httpx: (_ for _ in ()).throw(OSError("network unreachable")),
    )
    monkeypatch.setattr(
        lcm,
        "_install_into",
        lambda *_a, **_k: pytest.fail("must not install when offline with stale cache"),
    )

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == default_plugins.ENSURE_EXIT_OWED
    assert payload["state"] == "owed"
    assert payload["installed"] == []
    assert set(payload["owed"]) == {"live catalog", "hermes-security"}
    assert pc.live_fetch_recently_failed() is True


def test_manually_placed_unrecorded_default_is_recorded_so_removal_sticks(tmp_path, monkeypatch, capsys):
    """A manually placed default tree with no _default_plugins_installed entry must NOT be treated
    as settled by either step_install_default_plugins or ensure_defaults until recorded; once
    recorded, removing the tree keeps it removed."""
    entries = [_entry("hermes-security")]
    live = [{"entries": [_entry_dict("hermes-security")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(lcm, "_install_into", _fake_install_into(calls))

    # Manually placed plugin tree + pre-existing settled marker matching identity & defaults hash,
    # but config.yaml has NO _default_plugins_installed entry for hermes-security.
    (home / "plugins" / "hermes-security").mkdir(parents=True)
    default_plugins._write_settled(
        home / "cache" / "default-plugins.settled",
        "rev-ensure-1",
        default_plugins._defaults_hash(entries),
        default_names=["hermes-security"],
    )

    # step_install_default_plugins must NOT treat this home as settled yet.
    spawned: list = []
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: spawned.append(h) or True)
    assert default_plugins.step_install_default_plugins().get("owed") is True
    assert len(spawned) == 1

    # Restore settled marker; ensure_defaults must record hermes-security (without reinstalling).
    default_plugins._write_settled(
        home / "cache" / "default-plugins.settled",
        "rev-ensure-1",
        default_plugins._defaults_hash(entries),
        default_names=["hermes-security"],
    )
    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0 and payload["state"] == "nothing-pending"
    assert calls == []
    assert "hermes-security" in default_plugins._recorded(home)

    # Now user removes the plugin: both ensure_defaults and step_install_default_plugins keep it removed.
    (home / "plugins" / "hermes-security").rmdir()
    assert default_plugins.main(["ensure", "--json"]) == 0
    assert _parse_single_json_line(capsys.readouterr().out)["state"] == "nothing-pending"
    assert not (home / "plugins" / "hermes-security").exists()
    assert calls == []
    assert default_plugins.step_install_default_plugins() == {"ok": True}


def test_disabled_plugin_with_stale_installing_marker_is_not_reinstalled(tmp_path, monkeypatch, capsys):
    """If a previous automatic install crashed leaving installing-<name>, and the user subsequently
    added the plugin to plugins.disabled, retry must treat it as settled, never delete/reinstall/
    re-enable it, and clear the stale installing marker."""
    entries = [_entry("langfuse")]
    live = [{"entries": [_entry_dict("langfuse")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)

    plugin_dir = home / "plugins" / "langfuse"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "keep.txt").write_text("user tree", encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  disabled:\n    - langfuse\n", encoding="utf-8")
    marker = default_plugins._installing(home, entries[0])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: lambda *_n, **_k2: pytest.fail("must not reinstall disabled plugin"))
    assert default_plugins._settled(home, entries[0]) is True
    assert default_plugins._pending_names([home], entries) == []

    code = default_plugins.main(["ensure", "--json"])
    payload = _parse_single_json_line(capsys.readouterr().out)
    assert code == 0 and payload["state"] == "nothing-pending"
    assert (plugin_dir / "keep.txt").is_file()
    assert "langfuse" in default_plugins._recorded(home)
    assert not marker.exists()


def test_record_only_pass_never_installs_in_process_when_new_home_appears_mid_lock(tmp_path, monkeypatch):
    """When pending_before is empty before taking the lock, a newly created profile discovered after
    acquiring installer_lock must route to the detached child instead of calling _install_into in-process."""
    from contextlib import contextmanager

    entries = [_entry("langfuse")]
    live = [{"entries": [_entry_dict("langfuse")]}]
    home = _setup_env(tmp_path, monkeypatch, entries, live)
    (home / "plugins" / "langfuse").mkdir(parents=True)

    new_profile = tmp_path / "new-profile"
    new_profile.mkdir()
    homes_list = [home]
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: list(homes_list))

    @contextmanager
    def lock_with_mid_lock_profile(_wait):
        if new_profile not in homes_list:
            homes_list.append(new_profile)
        yield

    monkeypatch.setattr(default_plugins, "installer_lock", lock_with_mid_lock_profile)
    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: pytest.fail("record-only pass must never install in-process"))

    launched: list[tuple[str, Path | None]] = []

    def fake_launch(run_id: str, target_home: Path | None):
        launched.append((run_id, target_home))
        default_plugins._write_child_result(home, run_id, {"installed": ["langfuse"], "owed": []})
        return _FinishedChild(0)

    monkeypatch.setattr(default_plugins, "_launch_ensure_child", fake_launch)

    code, payload = default_plugins.ensure_defaults(timeout=30, say=lambda _m: None)
    assert code == 0
    assert payload["state"] == "installed"
    assert payload["installed"] == ["langfuse"]
    assert payload["pending_before"] == ["langfuse"]
    assert len(launched) == 1 and launched[0][1] is None


def test_install_and_record_reports_every_pass_of_a_rescanned_run(tmp_path, monkeypatch):
    """A rescan requested mid-run usually finds nothing: the run still reports what it installed."""
    _setup_env(tmp_path, monkeypatch, [_entry("hermes-security")], [{"entries": []}])
    from pm.environments import dependency_home_root
    rerun = default_plugins._state(Path(dependency_home_root()), "rerun")
    passes = iter([{"installed": ["hermes-security"], "owed": []}, {"installed": [], "owed": []}])

    def one_pass(homes, **kwargs):
        result = next(passes)
        if result["installed"]:  # a profile created during the first pass asks for a rescan
            rerun.parent.mkdir(parents=True, exist_ok=True)
            rerun.touch()
        return result

    monkeypatch.setattr(default_plugins, "_install_defaults_for_homes", one_pass)
    assert default_plugins.install_and_record(wait=1.0)["installed"] == ["hermes-security"]
