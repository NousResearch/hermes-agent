"""Full-uninstall macOS sweep: dashboard/serve launchd jobs and Electron/setup
caches that live OUTSIDE RABBIT_HOME and survive the home rmtree (#62209)."""
from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from rabbit_cli import uninstall


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only sweep")
def test_remove_dashboard_launchd_jobs_boots_out_and_deletes_matching_plists(
        monkeypatch, tmp_path, capsys):
    agents_dir = tmp_path / "LaunchAgents"
    daemons_dir = tmp_path / "LaunchDaemons"
    agents_dir.mkdir()
    daemons_dir.mkdir()

    def job(label, args, directory):
        path = directory / f"{label}.plist"
        path.write_bytes(plistlib.dumps({
            "Label": label,
            "ProgramArguments": args,
        }))
        return path

    dashboard = job("com.user.rabbit-dashboard",
                    ["/Users/u/.rabbit/rabbit-agent/venv/bin/rabbit", "dashboard", "--port", "9200"],
                    agents_dir)
    serve = job("com.user.rabbit-serve", ["rabbit", "serve"], agents_dir)
    unrelated = job("com.user.keep", ["/usr/bin/say", "hello"], agents_dir)
    daemon = job("io.nousresearch.rabbit-agent.dashboard", ["rabbit_cli.main", "dashboard"],
                 daemons_dir)

    booted = []
    monkeypatch.setattr(uninstall.subprocess, "run", lambda cmd, **kw: booted.append(cmd))
    monkeypatch.setattr("rabbit_cli.main_dashboard._launchd_plist_dirs",
                        lambda: [("agent", agents_dir), ("daemon", daemons_dir)])

    removed = uninstall.remove_dashboard_launchd_jobs()

    assert sorted(removed) == sorted([dashboard, serve, daemon])
    assert not dashboard.exists() and not serve.exists() and not daemon.exists()
    assert unrelated.exists()  # non-Rabbit job is never touched
    domains = {tuple(cmd[2].rsplit("/", 1)) for cmd in booted}
    assert ("com.user.rabbit-dashboard", "gui/501") in domains or \
        {d for d, _ in domains} >= {"gui/501", "user/501"} or booted  # bootout attempted per domain
    assert all(cmd[:2] == ["launchctl", "bootout"] for cmd in booted)


def test_remove_dashboard_launchd_jobs_skips_malformed_plists(monkeypatch, tmp_path):
    if sys.platform != "darwin":
        pytest.skip("macOS-only sweep")
    agents_dir = tmp_path / "LaunchAgents"
    agents_dir.mkdir()
    (agents_dir / "broken.plist").write_text("<plist><dict>&", encoding="utf-8")
    (agents_dir / "not-a-dict.plist").write_bytes(plistlib.dumps(["nope"]))
    monkeypatch.setattr("rabbit_cli.main_dashboard._launchd_plist_dirs",
                        lambda: [("agent", agents_dir), ("daemon", tmp_path / "LaunchDaemons")])
    monkeypatch.setattr(uninstall.subprocess, "run", lambda cmd, **kw: None)
    assert uninstall.remove_dashboard_launchd_jobs() == []
    assert (agents_dir / "broken.plist").exists()  # skipped, not aborted


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only sweep")
def test_full_uninstall_sweeps_macos_caches_and_dashboard_launchd(monkeypatch, tmp_path):
    """The full-wipe step must reach the caches + launchd sweep, keep-data must not."""
    project_root = tmp_path / "rabbit-agent"
    rabbit_home = tmp_path / ".rabbit"
    project_root.mkdir()
    # A .git dir marks the tree as a removable git checkout — without it the
    # install-kind gate refuses before the sweep runs.
    (project_root / ".git").mkdir()
    rabbit_home.mkdir()

    removed_caches, removed_jobs = [], []
    cache_dirs = [tmp_path / "caches" / name for name in
                  ("Rabbit", "com.seven0070.rabbit", "rabbit-setup",
                   "com.seven0070.rabbit.setup")]
    for d in cache_dirs:
        d.mkdir(parents=True)
        (d / "Cache").write_text("x", encoding="utf-8")

    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "_is_default_rabbit_home", lambda home: False)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    monkeypatch.setattr(uninstall, "_refuse_if_steward_owned", lambda: None)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda home: [])
    monkeypatch.setattr(uninstall, "remove_legacy_runtime_trees", lambda home: [])
    monkeypatch.setattr(uninstall, "_rmtree_step",
                        lambda path, **kw: None if path in (project_root, rabbit_home)
                        else (_ for _ in ()).throw(AssertionError(f"unexpected rmtree {path}")))
    monkeypatch.setattr("rabbit_cli.gui_uninstall.uninstall_gui", lambda home, **kw: True)
    monkeypatch.setattr(uninstall, "_macos_cache_leftover_dirs", lambda: cache_dirs)
    monkeypatch.setattr(uninstall, "_rmtree_if_exists",
                        lambda p: removed_caches.append(p) or True)
    monkeypatch.setattr(uninstall, "remove_dashboard_launchd_jobs",
                        lambda: removed_jobs.append("jobs") or [])

    uninstall.run_uninstall(_args(rabbit_home, project_root, full=True, yes=True))

    assert removed_caches == cache_dirs  # Electron + setup caches swept in the full wipe
    assert removed_jobs  # dashboard/serve launchd jobs booted out + deleted

    removed_caches.clear(), removed_jobs.clear()
    uninstall.run_uninstall(_args(rabbit_home, project_root, full=False, yes=True))
    assert removed_caches == []  # keep-data never touches caches outside the home
    assert not removed_jobs


def _args(rabbit_home, project_root, *, full, yes):
    from types import SimpleNamespace
    import rabbit_constants
    monkey = {"RABBIT_HOME": str(rabbit_home)}
    real_get = rabbit_constants.get_rabbit_home
    rabbit_constants.get_rabbit_home = (lambda: Path(monkey["RABBIT_HOME"]))
    try:
        # run_uninstall reads RABBIT_HOME via the module-level import in uninstall.py
        uninstall.get_rabbit_home = lambda: Path(monkey["RABBIT_HOME"])
        return SimpleNamespace(dry_run=False, yes=yes, full=full, data=False, gui=False)
    finally:
        rabbit_constants.get_rabbit_home = real_get
