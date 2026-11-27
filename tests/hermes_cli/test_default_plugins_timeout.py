"""The ensure timeout path never abandons an install, and a tree an interrupted install left is retried.

A root-home ``ensure`` runs the install in a detached child (it survives the ensure process and its
Desktop parent being killed) and waits for that child's result until the deadline. A default whose
install was cut off (kill, crash, power loss) keeps its ``installing`` marker, so the next pass reinstalls
it instead of recording the half-installed tree as the user's.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
import hermes_cli.plugin_catalog as pc
import pm.plugins_state
from hermes_cli import default_plugins
from hermes_cli.plugin_catalog import entry_from_mapping
from tests.waiting import wait_until

REPO = Path(__file__).resolve().parents[2]


def _entry_dict(name: str) -> dict:
    return {
        "name": name, "repo": "https://github.com/NousResearch/hermes-official-plugins", "sha": "a" * 40,
        "description": f"{name} desc", "maintainer": "NousResearch", "tier": "official", "default": True,
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-timeout-1")
    monkeypatch.setattr(default_plugins, "default_entries", lambda: [entry_from_mapping(_entry_dict("langfuse"), "langfuse")])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **kwargs: {"entries": [_entry_dict("langfuse")]})
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    return home


def _child_script(tmp_path: Path, args: tuple[str, ...], *, sleep: float = 0.0, installed=(), owed=(),
                  write: bool = True, wait_for_file: Path | None = None) -> list[str]:
    """A real child process standing in for the installer: it parses ``_child --run-id [--home]`` from
    argv, logs its start, optionally waits for *wait_for_file* or sleeps, then reports through the
    module's own result writer."""
    starts = tmp_path / "child-starts.txt"
    gate = str(wait_for_file) if wait_for_file is not None else ""
    body = textwrap.dedent(f"""
        import argparse, sys, time
        from pathlib import Path
        sys.path.insert(0, {str(REPO)!r})
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        c = sub.add_parser("_child")
        c.add_argument("--run-id", required=True)
        c.add_argument("--home", default="")
        ns = p.parse_args(sys.argv[1:])
        with open({str(starts)!r}, "a", encoding="utf-8") as fh:
            fh.write(ns.run_id + " " + (ns.home or "-") + "\\n")
        gate = {gate!r}
        if gate:
            cap = time.monotonic() + 30.0
            while not Path(gate).exists() and time.monotonic() < cap:
                time.sleep(0.02)
        elif {sleep!r} > 0:
            time.sleep({sleep!r})
        if {write!r}:
            from hermes_cli import default_plugins as d
            from pm.environments import dependency_home_root
            d._write_child_result(Path(dependency_home_root()), ns.run_id,
                                  {{"installed": {list(installed)!r}, "owed": {list(owed)!r}}})
        sys.exit(1 if {list(owed)!r} else 0)
    """)
    return [sys.executable, "-I", "-c", body, *args]


def _result_files(home: Path) -> list[Path]:
    cache = home / "cache"
    return sorted(cache.glob("default-plugins.result-*.json")) if cache.is_dir() else []


def _result(home: Path, run_id: str | None = None) -> dict:
    path = default_plugins._result_path(home, run_id) if run_id else (_result_files(home)[0] if _result_files(home) else None)
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else {}


def test_timed_out_ensure_leaves_no_install_running_in_its_own_process(home, tmp_path, monkeypatch):
    """On timeout ensure returns without a daemon thread still inside the install or a duplicate spawn."""
    release = threading.Event()
    inflight = [0]

    def install_into(h, consent=None):
        def install(name, discard=None, on_step=None):
            inflight[0] += 1
            try:
                (h / "plugins" / name).mkdir(parents=True, exist_ok=True)
                release.wait(timeout=5.0)
                return {"ok": True}
            finally:
                inflight[0] -= 1
        return install

    second_installers: list = []
    release_file = tmp_path / "release-child-1"
    monkeypatch.setattr(lcm, "_install_into", install_into)
    monkeypatch.setattr(default_plugins, "spawn_install", lambda home=None: second_installers.append(home) or True)
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, wait_for_file=release_file, write=False))
    try:
        started = time.monotonic()
        code, payload = default_plugins.ensure_defaults(timeout=0.3, say=lambda _m: None)
        returned_after = time.monotonic() - started
        still_installing_here = inflight[0]
    finally:
        release.set()
        release_file.touch()
    assert code == default_plugins.ENSURE_EXIT_TIMEOUT and payload["state"] == "timeout"
    assert returned_after < 10.0
    assert still_installing_here == 0, "ensure returned while its own thread was still installing"
    assert second_installers == [], "a timeout must not start a second installer beside the first"


class _Killed(BaseException):
    """The process dying mid-install: nothing after this point of the install runs."""


def test_tree_left_by_an_interrupted_install_is_reinstalled_not_recorded(home, monkeypatch):
    calls: list[str] = []

    def killed_install_into(h, consent=None):
        def install(name, discard=None, on_step=None):
            calls.append(f"killed:{name}")
            (h / "plugins" / name).mkdir(parents=True, exist_ok=True)  # tree placed, then the kill
            raise _Killed()
        return install

    monkeypatch.setattr(lcm, "_install_into", killed_install_into)
    with pytest.raises(_Killed):
        default_plugins.install_defaults(home, say=lambda _m: None)
    assert (home / "plugins" / "langfuse").is_dir()
    assert default_plugins._recorded(home) == set()

    discarded: list = []

    def good_install_into(h, consent=None):
        def install(name, discard=None, on_step=None):
            calls.append(f"ok:{name}")
            discarded.append(discard)
            (h / "plugins" / name).mkdir(parents=True, exist_ok=True)
            (h / f"admitted-{name}").touch()
            return {"ok": True}
        return install

    monkeypatch.setattr(lcm, "_install_into", good_install_into)
    assert default_plugins._pending_names([home], default_plugins.default_entries()) == ["langfuse"]
    result = default_plugins.install_defaults(home, say=lambda _m: None)
    assert calls == ["killed:langfuse", "ok:langfuse"], "the half-installed tree was recorded without its install"
    assert discarded == [home / "plugins" / "langfuse"]  # the leftover tree is discarded first
    assert result["installed"] == ["langfuse"] and (home / "admitted-langfuse").exists()
    assert default_plugins._recorded(home) == {"langfuse"}
    assert not (home / "cache" / "default-plugins.installing-langfuse").exists()


def test_timeout_returns_promptly_and_the_detached_child_completes(home, tmp_path, monkeypatch):
    release_file = tmp_path / "release-child"
    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: pytest.fail("ensure must not install in-process"))
    monkeypatch.setattr(default_plugins, "spawn_install", lambda home=None: pytest.fail("no continuation spawn"))
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, wait_for_file=release_file, installed=["langfuse"]))
    started = time.monotonic()
    try:
        code, payload = default_plugins.ensure_defaults(timeout=0.3, say=lambda _m: None)
        elapsed = time.monotonic() - started
        assert code == default_plugins.ENSURE_EXIT_TIMEOUT
        assert payload["state"] == "timeout" and payload["pending_before"] == ["langfuse"]
        assert elapsed < 10.0
        assert not _result(home), "child must still be waiting on the release file when ensure returns"
        assert not [t for t in threading.enumerate() if t.name.startswith("default-plugins")]
    finally:
        release_file.touch()

    wait_until(lambda: bool(_result(home)), timeout=20, what="detached ensure child result")
    result = _result(home)
    assert result.get("installed") == ["langfuse"], "the child did not finish after ensure returned"
    starts = (tmp_path / "child-starts.txt").read_text(encoding="utf-8").splitlines()
    assert len(starts) == 1, f"expected exactly one child, got {starts}"
    assert starts[0].split()[0] == result["run_id"] and starts[0].split()[1] == "-"


def test_slow_catalog_discovery_is_bounded_by_timeout_and_returns_3(tmp_path, monkeypatch):
    """When cache/plugin-catalog.json is missing/expired, ensure_defaults delegates catalog discovery
    to the detached child instead of blocking synchronously on HTTP, so --timeout bounds it."""
    home = tmp_path / "uncached-home"
    home.mkdir()
    release_file = tmp_path / "release-discovery"
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-uncached-1")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **_k: pytest.fail("parent must not fetch catalog synchronously"))
    monkeypatch.setattr(pc, "_download_live_catalog", lambda *_a, **_k: pytest.fail("parent must not do HTTP"))
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, wait_for_file=release_file, installed=["langfuse"]))
    started = time.monotonic()
    try:
        code, payload = default_plugins.ensure_defaults(timeout=0.3, say=lambda _m: None)
        elapsed = time.monotonic() - started
        assert code == default_plugins.ENSURE_EXIT_TIMEOUT
        assert payload["state"] == "timeout"
        assert payload["owed"] == ["live catalog"]
        assert elapsed < 10.0
        assert not _result(home)
    finally:
        release_file.touch()

    wait_until(lambda: bool(_result(home)), timeout=20, what="detached discovery child result")
    assert _result(home).get("installed") == ["langfuse"]


@pytest.mark.parametrize("installed,owed,code,state", [
    (["langfuse"], [], 0, "installed"),
    ([], [], 0, "nothing-pending"),
    ([], ["langfuse"], 2, "owed"),
])
def test_child_result_maps_to_the_exit_code_and_payload(home, tmp_path, monkeypatch, installed, owed, code, state):
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, sleep=0.0, installed=installed, owed=owed))
    got, payload = default_plugins.ensure_defaults(timeout=30, say=lambda _m: None)
    assert (got, payload["state"]) == (code, state)
    assert payload["installed"] == installed and payload["owed"] == owed
    assert payload["pending_before"] == ["langfuse"]
    assert set(payload) == {"state", "pending_before", "installed", "owed", "elapsed_ms"}


def test_a_child_that_dies_without_a_result_is_owed(home, tmp_path, monkeypatch):
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, sleep=0.0, owed=["x"], write=False))
    code, payload = default_plugins.ensure_defaults(timeout=30, say=lambda _m: None)
    assert code == default_plugins.ENSURE_EXIT_OWED and payload["state"] == "owed"
    assert payload["owed"] == ["langfuse"]


def test_a_stale_result_from_another_run_is_ignored(home, tmp_path, monkeypatch):
    default_plugins._write_child_result(home, "someone-else", {"installed": ["langfuse"], "owed": []})
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, sleep=0.0, write=False))
    code, payload = default_plugins.ensure_defaults(timeout=30, say=lambda _m: None)
    assert code == default_plugins.ENSURE_EXIT_OWED and payload["installed"] == []


def test_the_ensure_child_installs_and_writes_its_result_with_step_timing(home, monkeypatch):
    lines: list[str] = []

    def install_into(h, consent=None):
        def install(name, discard=None, on_step=None):
            assert (h / "cache" / f"default-plugins.installing-{name}").exists()  # marked before installing
            on_step("downloading")
            (h / "plugins" / name).mkdir(parents=True, exist_ok=True)
            on_step("python_packages")
            on_step("loading_tools")
            return {"ok": True}
        return install

    monkeypatch.setattr(lcm, "_install_into", install_into)
    assert default_plugins._ensure_child_work("run-1", None, say=lines.append) == 0
    assert _result(home) == {"run_id": "run-1", "installed": ["langfuse"], "owed": []}
    assert not (home / "cache" / "default-plugins.installing-langfuse").exists()
    timing = [line for line in lines if "timing langfuse" in line]
    assert len(timing) == 1 and "fetch+place" in timing[0] and "enable+deps" in timing[0]
    assert sum("default plugins pass:" in line for line in lines) == 1


def test_the_profile_child_installs_only_its_home(home, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home, other])
    calls: list = []

    def install_into(h, consent=None):
        def install(name, discard=None, on_step=None):
            calls.append((h, name))
            (h / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return {"ok": True}
        return install

    monkeypatch.setattr(lcm, "_install_into", install_into)
    assert default_plugins._ensure_child_work("run-2", other, say=lambda _m: None) == 0
    assert calls == [(other, "langfuse")]
    assert _result(home)["run_id"] == "run-2"


def test_profile_create_timeout_hands_off_to_one_child_for_that_home(tmp_path, monkeypatch):
    from hermes_cli import profiles
    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(default_plugins, "PROFILE_CREATE_ENSURE_TIMEOUT", 0.3)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    (root / "config.yaml").write_text("model:\n  default: m\n  provider: openrouter\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    monkeypatch.setattr(default_plugins, "default_entries", lambda: [entry_from_mapping(_entry_dict("langfuse"), "langfuse")])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **kwargs: {"entries": [_entry_dict("langfuse")]})
    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: pytest.fail("profile create must not install in-process"))
    monkeypatch.setattr(default_plugins, "spawn_install", lambda home=None: pytest.fail("no continuation spawn"))
    monkeypatch.setattr(default_plugins, "_child_argv",
                        lambda args=(): _child_script(tmp_path, args, sleep=1.0, installed=["langfuse"]))

    started = time.monotonic()
    result = provision_profile(ProfileCreateRequest(name="slowprof", alias=False, mirror_credentials=False))
    assert time.monotonic() - started < 10
    assert any("timed out" in w and "continuing in background" in w for w in result.warnings)
    wait_until(lambda: bool(_result(root)), timeout=20, what="detached profile child result")
    assert _result(root).get("installed") == ["langfuse"]
    starts = (tmp_path / "child-starts.txt").read_text(encoding="utf-8").splitlines()
    assert len(starts) == 1 and starts[0].split()[1] == str(result.path)


def test_main_child_subcommand_dispatches_to_the_ensure_child(home, monkeypatch):
    seen: list = []
    monkeypatch.setattr(default_plugins, "_ensure_child_work",
                        lambda run_id, target, *, say: seen.append((run_id, target)) or 0)
    monkeypatch.setattr(default_plugins, "install_and_record", lambda **_k: pytest.fail("plain child path"))
    assert default_plugins.main(["_child", "--run-id", "run-3", "--home", str(home)]) == 0
    assert seen == [("run-3", home)]


def test_nothing_pending_but_unsettled_records_in_process_without_a_child(home, monkeypatch):
    """A pass with nothing to fetch (a default already on disk, settled marker not written yet) only writes
    records and the settled marker: it stays in this process, so no second interpreter start is paid."""
    (home / "plugins" / "langfuse").mkdir(parents=True)  # the user's own tree: settled, unrecorded
    monkeypatch.setattr(default_plugins, "_launch_ensure_child", lambda *_a: pytest.fail("no child for a record-only pass"))
    monkeypatch.setattr(lcm, "_install_into", lambda *_a, **_k: lambda *_n, **_k2: pytest.fail("nothing to install"))
    code, payload = default_plugins.ensure_defaults(timeout=30, say=lambda _m: None)
    assert (code, payload["state"], payload["pending_before"]) == (0, "nothing-pending", [])
    assert default_plugins._recorded(home) == {"langfuse"}
    assert (home / "cache" / "default-plugins.settled").is_file()
    assert not [t for t in threading.enumerate() if t.name.startswith("default-plugins")]


def test_concurrent_runs_keep_isolated_result_files_and_unlink_on_read(home):
    """Per-run-id result files prevent concurrent runs from clobbering each other, unlink once read,
    and prune orphaned result files older than an hour."""
    import os

    stale = default_plugins._result_path(home, "stale-old-run")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"run_id": "stale-old-run", "installed": [], "owed": []}), encoding="utf-8")
    old_ts = time.time() - default_plugins._RESULT_MAX_AGE - 60
    os.utime(stale, (old_ts, old_ts))

    default_plugins._write_child_result(home, "run-a", {"installed": ["plugin-a"], "owed": []})
    default_plugins._write_child_result(home, "run-b", {"installed": [], "owed": ["plugin-b"]})

    assert not stale.exists(), "stale result file older than 1 hour should be pruned"
    assert default_plugins._read_child_result(home, "run-a") == {"installed": ["plugin-a"], "owed": []}
    assert not default_plugins._result_path(home, "run-a").exists()
    assert default_plugins._result_path(home, "run-b").is_file()
    assert default_plugins._read_child_result(home, "run-b") == {"installed": [], "owed": ["plugin-b"]}
    assert not default_plugins._result_path(home, "run-b").exists()
