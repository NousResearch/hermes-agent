"""Other-install discovery, the launch notice, and tree-scoped removal.

Several Hermes installs can share one ``HERMES_HOME``. These tests pin the
contracts that matter when one install removes another: the running install
is never removable, only the removed tree and its own launchers go, and the
state every install shares is left alone.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import installs

# The autouse fixture below stubs discovery; one test needs the real function.
_REAL_PACKAGED_APP_DIRS = installs._packaged_app_dirs


def _checkout(root: Path) -> Path:
    (root / ".git").mkdir(parents=True)
    (root / "hermes_cli").mkdir()
    return root


def _sealed(root: Path, distribution: str = "desktop-app", version: str = "0.21.5") -> Path:
    (root / "hermes_cli").mkdir(parents=True)
    (root / "install-stamp.json").write_text(
        json.dumps({
            "commit": "a" * 40,
            "updateMechanism": "microsoft-store",
            "distribution": distribution,
            "displayVersion": version,
        }),
        encoding="utf-8",
    )
    return root


def _counts(state: dict) -> dict:
    """The host-independent part of a notice state: ``partial`` is True on Windows only."""
    return {key: state[key] for key in ("count", "dismissed")}


def _ok(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _quiet(*args, **kwargs) -> subprocess.CompletedProcess:
    return _ok("")


@pytest.fixture(autouse=True)
def _no_real_machine_state(monkeypatch):
    monkeypatch.setattr(installs, "_packaged_app_dirs", lambda: [])
    monkeypatch.setattr(installs, "_path_entries", lambda windows: [])
    monkeypatch.setattr(installs, "_launcher_bin_dirs", lambda default_root: ())
    monkeypatch.setattr("hermes_cli.boot_bootstrap._git_binary", lambda: "git")


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "hermes"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


@pytest.fixture
def running(tmp_path, monkeypatch):
    root = _checkout(tmp_path / "running" / "hermes-agent")
    monkeypatch.setattr(installs, "current_root", lambda: root.resolve())
    return root


class TestDiscovery:
    @pytest.mark.platforms("windows")
    def test_bundled_per_user_install_is_discovered_but_not_uninstalled_by_gui_uninstall(
            self, tmp_path, monkeypatch):
        from hermes_cli import gui_uninstall

        bundled = tmp_path / "Programs" / "HermesBundled"
        bundled.mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        assert bundled in _REAL_PACKAGED_APP_DIRS()
        assert bundled not in gui_uninstall.packaged_gui_app_paths()

    def test_running_and_home_checkout_are_found_and_current_is_marked(self, home, running):
        managed = _checkout(home / "hermes-agent")

        found = installs.discover(appx=False)

        by_root = {i.root: i for i in found}
        assert by_root[running.resolve()].current is True
        assert by_root[managed.resolve()].current is False
        assert {i.steward for i in found} == {"git"}

    def test_steward_comes_from_the_stamp(self, home, running, tmp_path):
        sealed = _sealed(tmp_path / "bundle" / "hermes-agent", distribution="nix")
        registry = {"seen": {"x": {"root": str(sealed)}}}

        found = {i.root: i for i in installs.discover(registry=registry, appx=False)}

        assert found[sealed.resolve()].steward == "nix"

    def test_version_label_matches_the_apps_version_with_the_distance_past_the_release(self, tmp_path):
        stamped = tmp_path / "bundle" / "hermes-agent"
        stamped.mkdir(parents=True)
        (stamped / "install-stamp.json").write_text(
            json.dumps({"commit": "a" * 40, "updateMechanism": "microsoft-store",
                        "baseVersion": "0.21.5", "distance": 12}), encoding="utf-8")
        bare = tmp_path / "bare"
        bare.mkdir()

        assert installs._version_label(stamped) == "0.21.5+12"

        (stamped / "install-stamp.json").write_text(
            json.dumps({"commit": "b" * 40, "updateMechanism": "microsoft-store", "source": "commit-build",
                        "baseVersion": "0.0.0"}), encoding="utf-8")
        assert installs._version_label(stamped) == "commit build bbbbbbb"
        assert installs._version_label(bare) is None

    def test_registry_entry_for_a_deleted_tree_is_dropped(self, home, running, tmp_path):
        registry = {"seen": {"x": {"root": str(tmp_path / "gone")}}}

        roots = [i.root for i in installs.discover(registry=registry, appx=False)]

        assert tmp_path / "gone" not in roots

    def test_one_tree_found_by_two_sources_is_one_install(self, home, running):
        managed = _checkout(home / "hermes-agent")
        registry = {"seen": {"x": {"root": str(managed)}}}

        found = installs.discover(registry=registry, appx=False)

        matches = [i for i in found if i.root == managed.resolve()]
        assert len(matches) == 1
        assert set(matches[0].sources) == {"home", "registry"}

    @pytest.mark.parametrize("as_list", [True, False])
    def test_appx_packages_become_installs_with_their_package_name(self, home, running, tmp_path, as_list):
        package = tmp_path / "WindowsApps" / "Nous.Hermes_1_arm64__abc"
        payload = package / "app" / "resources" / "agent-payload"
        _sealed(payload / "hermes-agent")
        (payload / "manifest.json").write_text(
            json.dumps({"runtime": {"repoDir": "hermes-agent"}}), encoding="utf-8")
        entry = {"PackageFullName": "Nous.Hermes_1_arm64__abc", "InstallLocation": str(package)}

        found = installs.discover(
            appx=True, windows=True, run=lambda *a, **k: _ok(json.dumps([entry] if as_list else entry)))

        packaged = [i for i in found if i.package_full_name]
        assert [i.package_full_name for i in packaged] == ["Nous.Hermes_1_arm64__abc"]
        assert packaged[0].root == (payload / "hermes-agent").resolve()
        assert packaged[0].steward == "desktop-app"

    def test_appx_is_not_queried_off_windows(self, home, running):
        def boom(*a, **k):
            raise AssertionError("PowerShell must not run off Windows")

        installs.discover(appx=True, windows=False, run=boom)


class TestRegistry:
    def test_record_launch_adds_this_install_once_per_interval(self, home, running):
        installs.record_launch(now=1000)
        installs.record_launch(now=1000 + installs._TOUCH_INTERVAL_S - 1)
        first = installs.read_registry()["seen"]
        installs.record_launch(now=1000 + installs._TOUCH_INTERVAL_S + 1)
        later = installs.read_registry()["seen"]

        [entry] = first.values()
        assert entry["last_launched"] == 1000
        assert entry["root"] == str(running.resolve())
        [entry] = later.values()
        assert entry["last_launched"] == 1000 + installs._TOUCH_INTERVAL_S + 1

    def test_record_launch_skips_a_linked_git_worktree(self, home, tmp_path):
        worktree = tmp_path / "wt"
        (worktree / "hermes_cli").mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")

        installs.record_launch(worktree)

        assert installs.read_registry()["seen"] == {}

    def test_record_launch_leaves_a_missing_home_alone(self, tmp_path, monkeypatch, running):
        missing = tmp_path / "no-home-yet"
        monkeypatch.setenv("HERMES_HOME", str(missing))

        installs.record_launch()

        assert not missing.exists()

    def test_unreadable_registry_reads_as_empty(self, home):
        (home / installs.REGISTRY_NAME).write_text("{not json", encoding="utf-8")

        assert installs.read_registry() == {"schema": 1, "seen": {}, "dismissed": ""}


class TestNotice:
    def test_no_notice_without_other_installs(self, home, running):
        assert installs.launch_notice() is None

    def test_notice_counts_other_installs(self, home, running):
        _checkout(home / "hermes-agent")

        notice = installs.launch_notice()

        assert notice is not None
        assert "1 other Hermes install " in notice
        assert "hermes installs" in notice

    def test_notice_hedges_the_count_only_when_the_launch_path_skips_a_query(self):
        assert installs._notice_line(2, partial=True).startswith("At least 2 other Hermes installs ")
        assert installs._notice_line(2, partial=False).startswith("2 other Hermes installs ")

    def test_notice_ignores_installs_that_remove_would_refuse(self, home, running, tmp_path):
        _sealed(tmp_path / "store" / "hermes-agent", distribution="nix")
        installs.record_launch(root=tmp_path / "store" / "hermes-agent", now=1)

        assert installs.launch_notice() is None
        assert _counts(installs.notice_state()) == {"count": 0, "dismissed": False}

    def test_dismissal_holds_until_the_set_of_installs_changes(self, home, running, tmp_path):
        _checkout(home / "hermes-agent")
        assert installs.dismiss() == 1
        assert installs.launch_notice() is None

        third = _checkout(tmp_path / "third" / "hermes-agent")
        installs.record_launch(root=third, now=1)

        notice = installs.launch_notice()
        assert notice is not None
        assert "2 other Hermes installs " in notice


class TestLaunchers:
    def test_launchers_come_in_path_order_with_their_owner(self, home, running, tmp_path):
        managed = _checkout(home / "hermes-agent")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "hermes.cmd").write_text("@echo off\r\nrem unrelated\r\n", encoding="utf-8")
        owned_dir = tmp_path / "owned"
        owned_dir.mkdir()
        (owned_dir / "hermes.cmd").write_text(f'"{managed.resolve()}\\venv\\python.exe"', encoding="utf-8")
        found = installs.discover(appx=False)

        launchers = installs.path_launchers(
            found, windows=True, pathext=(".cmd",),
            entries=[str(bin_dir), str(tmp_path / "missing"), str(owned_dir)])

        assert [Path(entry.path).parent for entry in launchers] == [bin_dir, owned_dir]
        owner = {entry.path.parent: entry.owner for entry in launchers}
        assert owner[bin_dir] is None
        assert owner[owned_dir] == installs.install_id(managed)

    def test_a_directory_listed_twice_on_path_reports_its_launcher_once(self, home, running, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "hermes.cmd").write_text("@echo off\r\n", encoding="utf-8")

        launchers = installs.path_launchers(
            installs.discover(appx=False), windows=True, pathext=(".cmd",),
            entries=[str(bin_dir), str(bin_dir) + "\\", str(bin_dir)])

        assert [entry.path for entry in launchers] == [bin_dir / "hermes.cmd"]

    def test_windowsapps_alias_belongs_to_the_package_install(self, home, running, tmp_path):
        package = tmp_path / "pkg"
        payload = package / "app" / "resources" / "agent-payload"
        _sealed(payload / "hermes-agent")
        (payload / "manifest.json").write_text(
            json.dumps({"runtime": {"repoDir": "hermes-agent"}}), encoding="utf-8")
        entry = {"PackageFullName": "Nous.Hermes_1_arm64__abc", "InstallLocation": str(package)}
        found = installs.discover(appx=True, windows=True, run=lambda *a, **k: _ok(json.dumps(entry)))
        alias_dir = tmp_path / "Microsoft" / "WindowsApps"
        alias_dir.mkdir(parents=True)
        (alias_dir / "hermes.exe").write_bytes(b"")

        [launcher] = installs.path_launchers(
            found, windows=True, pathext=(".exe",), entries=[str(alias_dir)])

        [packaged] = [i for i in found if i.package_full_name]
        assert launcher.owner == packaged.id


class TestRemovalPlan:
    def _install(self, root, *, steward="git", package=None):
        return installs.Install(
            id=installs.install_id(root), root=root.resolve(), steward=steward,
            sources=("test",), current=False, package_full_name=package)

    def _plan(self, install, home, current, *, windows=False, run=None, **options):
        return installs.plan_removal(
            install, current=current, hermes_home=home, default_root=home, windows=windows,
            run=run or (lambda *a, **k: _ok("")), **options)

    def test_checkout_with_uncommitted_changes_is_refused_and_left_alone(self, home, running, tmp_path):
        dev = _checkout(tmp_path / "dev" / "hermes-agent")
        probes = []

        def dirty(argv, **kwargs):
            probes.append(argv)
            return _ok(" ?? WIP.txt\n")

        plan = self._plan(self._install(dev), home, running.resolve(), run=dirty)

        assert plan.action == "refuse"
        assert "uncommitted changes" in plan.refusal
        assert probes and probes[0][1:3] == ["-C", str(dev.resolve())]
        assert dev.exists()

    def test_clean_checkout_with_work_no_remote_has_is_refused_and_left_alone(self, home, running, tmp_path):
        def git(repo, *args):
            subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
                           check=True, capture_output=True)

        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
        dev = tmp_path / "dev" / "hermes-agent"
        (dev / "hermes_cli").mkdir(parents=True)
        git(dev, "init", "-q", "-b", "main")
        (dev / "hermes_cli" / "__init__.py").write_text("x\n", encoding="utf-8")
        git(dev, "add", "-A")
        git(dev, "commit", "-qm", "base")
        git(dev, "remote", "add", "origin", str(remote))
        git(dev, "push", "-q", "origin", "main")
        install = self._install(dev)
        assert self._plan(install, home, running.resolve(), run=subprocess.run).action == "tree"

        git(dev, "checkout", "-q", "-b", "feature")
        (dev / "feature.py").write_text("work\n", encoding="utf-8")
        git(dev, "add", "-A")
        git(dev, "commit", "-qm", "local only")
        unpushed = self._plan(install, home, running.resolve(), run=subprocess.run)
        git(dev, "push", "-q", "origin", "feature")
        (dev / "feature.py").write_text("wip\n", encoding="utf-8")
        git(dev, "stash", "-q")
        stashed = self._plan(install, home, running.resolve(), run=subprocess.run)

        assert unpushed.action == "refuse"
        assert "no remote has" in (unpushed.refusal or "")
        assert stashed.action == "refuse"
        assert "stashed" in (stashed.refusal or "")
        assert dev.exists()

    def test_a_probe_that_cannot_run_refuses_instead_of_deleting(self, home, running, tmp_path):
        dev = _checkout(tmp_path / "dev" / "hermes-agent")

        def broken(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr="fatal: dubious ownership")

        plan = self._plan(self._install(dev), home, running.resolve(), run=broken)

        assert plan.action == "refuse"
        assert "could not be checked" in plan.refusal

    def test_tree_holding_another_install_is_refused(self, home, running, tmp_path):
        outer = _checkout(tmp_path / "dev" / "hermes-agent")
        inner = _checkout(outer / ".worktrees" / "wt1")
        outer_install, inner_install = self._install(outer), self._install(inner)

        plan = self._plan(outer_install, home, running.resolve(), found=[outer_install, inner_install])

        assert plan.action == "refuse"
        assert str(inner.resolve()) in plan.refusal
        assert outer.exists() and inner.exists()

    def test_running_install_is_not_removable(self, home, running):
        plan = self._plan(self._install(running), home, running.resolve())

        assert plan.action == "refuse"
        assert "running" in plan.refusal

    def test_running_package_install_is_not_removable(self, home, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")

        plan = self._plan(install, home, bundle.resolve(), windows=True)

        assert plan.action == "refuse"
        assert plan.command is None
        assert "running" in plan.refusal

    def test_tree_containing_hermes_home_is_not_removable(self, tmp_path, monkeypatch, running):
        outer = _checkout(tmp_path / "outer")
        inner_home = outer / "state"
        inner_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(inner_home))

        plan = self._plan(self._install(outer), inner_home, running.resolve())

        assert plan.action == "refuse"
        assert "HERMES_HOME" in plan.refusal

    def test_sealed_steward_is_refused_with_its_message(self, home, running, tmp_path):
        nix = _sealed(tmp_path / "store" / "hermes-agent", distribution="nix")

        plan = self._plan(self._install(nix, steward="nix"), home, running.resolve())

        assert plan.action == "refuse"
        assert "Nix" in plan.refusal

    def test_package_install_uses_remove_appxpackage_on_windows_only(self, home, running, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")

        on_windows = self._plan(install, home, running.resolve(), windows=True)
        elsewhere = self._plan(install, home, running.resolve(), windows=False)

        assert on_windows.action == "appx"
        assert on_windows.command[-1] == "Remove-AppxPackage -Package Nous.Hermes_1_arm64__abc"
        assert elsewhere.action == "refuse"

    def test_all_users_package_removal_needs_an_elevated_terminal(self, home, running, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")

        plan = self._plan(install, home, running.resolve(), windows=True, all_users=True, elevated=False)

        assert plan.action == "refuse"
        assert plan.command is None
        assert "elevated" in plan.refusal

    def test_all_users_package_removal_adds_the_flag_when_elevated(self, home, running, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")

        plan = self._plan(install, home, running.resolve(), windows=True, all_users=True, elevated=True)

        assert plan.action == "appx"
        assert plan.command[-1] == "Remove-AppxPackage -Package Nous.Hermes_1_arm64__abc -AllUsers"
        assert any("all users" in step for step in plan.steps)

    @pytest.mark.parametrize(("is_admin", "action"), [(False, "refuse"), (True, "appx")])
    def test_all_users_removal_asks_the_system_whether_the_terminal_is_elevated(
            self, home, running, tmp_path, monkeypatch, is_admin, action):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")
        monkeypatch.setattr(installs, "_is_elevated", lambda windows=None: is_admin)

        plan = self._plan(install, home, running.resolve(), windows=True, all_users=True)

        assert plan.action == action

    def test_running_package_install_is_refused_even_for_all_users(self, home, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="Nous.Hermes_1_arm64__abc")

        plan = self._plan(install, home, bundle.resolve(), windows=True, all_users=True, elevated=True)

        assert plan.action == "refuse"
        assert plan.command is None

    def test_elevation_is_never_reported_off_windows(self):
        assert installs._is_elevated(windows=False) is False

    def test_unsafe_package_name_is_refused(self, home, running, tmp_path):
        bundle = _sealed(tmp_path / "pkg" / "hermes-agent")
        install = self._install(bundle, steward="desktop-app", package="x; Remove-Item C:\\ -Recurse")

        plan = self._plan(install, home, running.resolve(), windows=True)

        assert plan.action == "refuse"
        assert plan.command is None

    def test_posix_removal_takes_only_the_launchers_that_belong_to_the_tree(self, home, running, tmp_path, monkeypatch):
        managed = _checkout(home / "hermes-agent")
        user_bin = tmp_path / "user-bin"
        user_bin.mkdir()
        for name in ("hermes", "hermes-acp", "hermes-agent"):
            (user_bin / name).write_text(f"owner={managed.resolve()}\n", encoding="utf-8")
        (user_bin / "uv").write_text("unrelated\n", encoding="utf-8")
        bin_dir = home / "bin"
        bin_dir.mkdir()
        (bin_dir / "hermes").write_text("owner=another-tree\n", encoding="utf-8")
        (bin_dir / "hermes-acp").write_text(f"owner={managed.resolve()}\n", encoding="utf-8")
        monkeypatch.setattr(installs, "_launcher_bin_dirs", lambda default_root: (user_bin, default_root / "bin"))
        monkeypatch.setattr(
            "hermes_cli._launchers._owns_launcher",
            lambda target, root: f"owner={root}" in target.read_text(encoding="utf-8-sig"))
        plan = self._plan(self._install(managed), home, running.resolve(), windows=False)

        assert sorted(p.name for p in plan.launchers) == ["hermes", "hermes-acp", "hermes-acp", "hermes-agent"]
        assert installs.execute_removal(plan) == 0

        assert not managed.exists()
        assert sorted(p.name for p in user_bin.iterdir()) == ["uv"]
        assert sorted(p.name for p in bin_dir.iterdir()) == ["hermes"]

    @pytest.mark.platforms("posix")
    def test_posix_launcher_that_runs_the_trees_venv_is_recognized(self, home, running, tmp_path, monkeypatch):
        managed = _checkout(home / "hermes-agent")
        user_bin = tmp_path / "user-bin"
        user_bin.mkdir()
        launcher = user_bin / "hermes"
        launcher.write_text(f'#!/bin/sh\nexec {managed.resolve()}/venv/bin/hermes "$@"\n', encoding="utf-8")
        monkeypatch.setattr(installs, "_launcher_bin_dirs", lambda default_root: (user_bin,))

        plan = self._plan(self._install(managed), home, running.resolve(), windows=False)

        assert plan.launchers == (launcher,)

    def test_a_launcher_that_cannot_be_removed_makes_the_command_fail(self, home, running, tmp_path):
        managed = _checkout(home / "hermes-agent")
        stuck = tmp_path / "stuck-launcher"
        stuck.mkdir()
        plan = installs.RemovalPlan(self._install(managed), "tree", tree=managed.resolve(), launchers=(stuck,))

        assert installs.execute_removal(plan) == 1

        assert not managed.exists()
        assert stuck.exists()

    def test_managed_clone_removal_keeps_everything_the_installs_share(self, home, running):
        managed = _checkout(home / "hermes-agent")
        pack = managed / ".git" / "objects"
        pack.mkdir()
        readonly = pack / "pack.idx"
        readonly.write_text("x", encoding="utf-8")
        readonly.chmod(0o444)
        bin_dir = home / "bin"
        bin_dir.mkdir()
        for name in ("hermes.exe", "hermes-acp.cmd", "uv.exe"):
            (bin_dir / name).write_text("x", encoding="utf-8")
        (home / "config.yaml").write_text("model: x\n", encoding="utf-8")
        plan = self._plan(self._install(managed), home, running.resolve(), windows=True)

        assert installs.execute_removal(plan) == 0

        assert not managed.exists()
        assert sorted(p.name for p in bin_dir.iterdir()) == ["uv.exe"]
        assert (home / "config.yaml").exists()

    def test_unmanaged_checkout_removal_leaves_bin_alone(self, home, running, tmp_path):
        elsewhere = _checkout(tmp_path / "elsewhere" / "hermes-agent")
        bin_dir = home / "bin"
        bin_dir.mkdir()
        (bin_dir / "hermes.exe").write_text("x", encoding="utf-8")
        plan = self._plan(self._install(elsewhere), home, running.resolve(), windows=True)

        assert installs.execute_removal(plan) == 0

        assert not elsewhere.exists()
        assert (bin_dir / "hermes.exe").exists()

    def test_shortcuts_into_the_tree_are_removed_and_others_kept(self, home, running, tmp_path):
        managed = _checkout(home / "hermes-agent")
        mine = tmp_path / "mine.lnk"
        other = tmp_path / "other.lnk"
        mine.write_text("x", encoding="utf-8")
        other.write_text("x", encoding="utf-8")
        listing = [
            {"Path": str(mine), "Target": str(managed / "apps" / "desktop" / "Hermes.exe")},
            {"Path": str(other), "Target": str(tmp_path / "elsewhere" / "Hermes.exe")},
        ]
        plan = self._plan(
            self._install(managed), home, running.resolve(), windows=True,
            run=lambda argv, **k: _ok("" if argv[0] == "git" else json.dumps(listing)))

        assert plan.shortcuts == (mine,)
        installs.execute_removal(plan)

        assert not mine.exists()
        assert other.exists()


class TestCli:
    def _args(self, **fields):
        base = {"installs_command": "list", "json": False, "install_id": None, "yes": False, "dry_run": False,
                "all_users": False}
        return SimpleNamespace(**{**base, **fields})

    def test_list_json_carries_installs_and_launchers(self, home, running, capsys):
        managed = _checkout(home / "hermes-agent")

        rc = installs.run_cli(self._args(json=True), run=_quiet)

        data = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert data["current"] == installs.install_id(running)
        ids = {entry["id"]: entry for entry in data["installs"]}
        assert ids[installs.install_id(managed)]["removable"] is True
        assert ids[installs.install_id(running)]["removable"] is False
        assert isinstance(data["launchers"], list)

    def test_list_json_reports_the_notice_state(self, home, running, capsys):
        def notice():
            installs.run_cli(self._args(json=True), run=_quiet)
            return json.loads(capsys.readouterr().out)["notice"]

        assert _counts(notice()) == {"count": 0, "dismissed": False}

        _checkout(home / "hermes-agent")
        assert _counts(notice()) == {"count": 1, "dismissed": False}

        installs.dismiss()
        assert _counts(notice()) == {"count": 1, "dismissed": True}
        assert notice()["partial"] is installs._windows(None)

    def test_dry_run_changes_nothing(self, home, running, capsys):
        managed = _checkout(home / "hermes-agent")

        rc = installs.run_cli(self._args(
            installs_command="remove", install_id=installs.install_id(managed), dry_run=True), run=_quiet)

        assert rc == 0
        assert managed.exists()
        assert str(managed.resolve()) in capsys.readouterr().out

    def test_remove_without_yes_needs_a_terminal(self, home, running, monkeypatch, capsys):
        managed = _checkout(home / "hermes-agent")
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))

        rc = installs.run_cli(self._args(
            installs_command="remove", install_id=installs.install_id(managed)), run=_quiet)

        assert rc == 1
        assert managed.exists()

    def test_remove_with_yes_removes_the_tree(self, home, running):
        managed = _checkout(home / "hermes-agent")

        rc = installs.run_cli(self._args(
            installs_command="remove", install_id=installs.install_id(managed), yes=True), run=_quiet)

        assert rc == 0
        assert not managed.exists()

    def test_unknown_id_is_an_error(self, home, running, capsys):
        rc = installs.run_cli(
            self._args(installs_command="remove", install_id="nope", yes=True), run=_quiet)

        assert rc == 1
        assert "nope" in capsys.readouterr().err

    def test_all_users_applies_only_to_package_installs(self, home, running, capsys):
        managed = _checkout(home / "hermes-agent")

        rc = installs.run_cli(self._args(
            installs_command="remove", install_id=installs.install_id(managed), yes=True, all_users=True),
            run=_quiet)

        assert rc == 1
        assert managed.exists()
        assert "--all-users" in capsys.readouterr().err


def test_powershell_calls_read_utf8_output(tmp_path):
    seen = []

    def run(cmd, **kwargs):
        seen.append((list(cmd), kwargs))
        return _ok("[]")

    installs._appx_packages(run, True)
    installs._shortcuts_into(tmp_path, run)
    package = installs.Install(
        id="x", root=tmp_path, steward="desktop-app", sources=("appx",),
        package_full_name="Nous.Hermes_1_arm64__abc")
    plan = installs.RemovalPlan(
        package, "appx",
        command=("powershell", "-NoProfile", "-NonInteractive", "-Command", "Remove-AppxPackage -Package Nous.Hermes_1_arm64__abc"))
    installs.execute_removal(plan, run=run)

    assert len(seen) == 3
    for cmd, kwargs in seen:
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert cmd[-1].startswith("[Console]::OutputEncoding")
    assert plan.command[-1] == "Remove-AppxPackage -Package Nous.Hermes_1_arm64__abc"
