"""New profile targets must not reuse the default session namespace."""

import tarfile
import shutil
import os
import builtins
import io
import stat
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_cli import profiles
from hermes_cli import profile_distribution as distributions
from hermes_cli.profile_distribution import (
    DistributionError,
    DistributionManifest,
    install_distribution,
    update_distribution,
    write_manifest,
)
from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted, named_profile_is_deleted


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Lifecycle tests use real profile files, never host services or running backends.
    for name in (
        "_cleanup_gateway_service", "_maybe_register_gateway_service", "_maybe_unregister_gateway_service",
        "_stop_profile_backends", "_notify_multiplexer",
    ):
        monkeypatch.setattr(profiles, name, lambda *args, **kwargs: None)
    return home


@pytest.mark.parametrize("operation", [
    "create", "import", "rename", "distribution", "distribution_tombstone", "distribution_symlink",
    "stage_inside_source", "stage_inside_target",
    "stage_file_link", "stage_dir_link", "stage_disappears", "stage_permission", "stage_manifest", "stage_existing",
])
@pytest.mark.parametrize("name", ["main", "MAIN", " main "])
def test_new_profile_targets_cannot_claim_default_namespace(profile_home, tmp_path, monkeypatch, operation, name):
    source = profile_home / "profiles" / "source"
    source.mkdir(parents=True)
    config = "model:\n  provider: custom\n  default: local-model\n"
    (source / "config.yaml").write_text(config)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="source")
    write_manifest(source, DistributionManifest(name="source"))
    target = profile_home / "profiles" / "main"
    if operation == "stage_existing":
        target.mkdir()
        (target / "config.yaml").write_text(config)
        workdir = tmp_path / "staging"
        previous = workdir / "local"
        previous.mkdir(parents=True)
        (previous / "keep.md").write_text("Caller-owned tree")
        with pytest.raises(DistributionError):
            distributions.plan_install(str(source), workdir, override_name=name)
        assert list(previous.iterdir()) == [previous / "keep.md"]
        assert (previous / "keep.md").read_text() == "Caller-owned tree"
        assert (target / "config.yaml").read_text() == config
        return

    if operation in {"stage_file_link", "stage_dir_link", "stage_disappears", "stage_permission", "stage_manifest"}:
        (source / "SOUL.md").write_text("Planned source content")
        (source / "skills").mkdir()
        (source / "skills" / "guide.md").write_text("Planned skill content")
        target.mkdir()
        (target / "config.yaml").write_text(config)
        workdir = tmp_path / "staging"
        external = tmp_path / "external"
        root_stat = source.stat()
        scanned = False
        real_scandir = os.scandir

        @contextmanager
        def mutate_after_enumeration(path):
            nonlocal scanned
            info = os.fstat(path) if isinstance(path, int) else Path(path).stat()
            is_source = (info.st_dev, info.st_ino) == (root_stat.st_dev, root_stat.st_ino)
            with real_scandir(path) as entries:
                yield entries
            if not is_source or scanned:
                return
            scanned = True
            if operation == "stage_permission":
                raise PermissionError("source permission changed during staging")
            if operation == "stage_disappears":
                (source / "SOUL.md").unlink()
                return
            if operation == "stage_manifest":
                write_manifest(source, DistributionManifest(name="other", version="9.9.9"))
                return
            if operation == "stage_dir_link":
                shutil.rmtree(source / "skills")
                external.mkdir()
                (external / "guide.md").write_text("External private content")
                (source / "skills").symlink_to(external, target_is_directory=True)
            else:
                (source / "SOUL.md").unlink()
                external.write_text("External private content")
                (source / "SOUL.md").symlink_to(external)

        monkeypatch.setattr(os, "scandir", mutate_after_enumeration)
        with pytest.raises(DistributionError):
            distributions.plan_install(str(source), workdir, override_name=name)
        assert scanned
        assert not (workdir / "local").exists()
        assert (target / "config.yaml").read_text() == config
        assert not (target / "SOUL.md").exists()
        return

    if operation in {"stage_inside_source", "stage_inside_target"}:
        parent = source if operation == "stage_inside_source" else target / "skills"
        workdir = parent / "tmp"
        workdir.mkdir(parents=True)
        if operation == "stage_inside_source":
            def refuse_recursive_copy(*args, **kwargs):
                pytest.fail("staging tried to copy the source into itself")

            monkeypatch.setattr(distributions.shutil, "copytree", refuse_recursive_copy)
        else:
            (target / "config.yaml").write_text(config)
            (parent / "guide.md").write_text("Existing skill content")
            write_manifest(source, DistributionManifest(name=name))

        with pytest.raises(DistributionError, match="outside"):
            distributions.plan_install(str(source), workdir)
        assert (source / "config.yaml").read_text() == config
        if operation == "stage_inside_source":
            assert not list(workdir.iterdir())
            assert not target.exists()
        else:
            assert (target / "config.yaml").read_text() == config
            assert (parent / "guide.md").read_text() == "Existing skill content"
            assert not list(workdir.iterdir())
        return

    if operation == "distribution_symlink":
        external = tmp_path / "private.md"
        external.write_text("Not distribution content")
        (source / "SOUL.md").symlink_to(external)
        target.mkdir()
        (target / "config.yaml").write_text(config)
        workdir = tmp_path / "staging"
        with pytest.raises(DistributionError, match="symlink"):
            distributions.plan_install(str(source), workdir, override_name=name)
        assert not (workdir / "local").exists()
        assert external.read_text() == "Not distribution content"
        assert (target / "config.yaml").read_text() == config
        assert not (target / "SOUL.md").exists()
        return

    tombstoned = operation == "distribution_tombstone"
    if tombstoned:
        target.mkdir()
        mark_named_profile_deleted(target)
    actions = {
        "create": lambda: profiles.create_profile(name, no_skills=True),
        "import": lambda: profiles.import_profile(str(archive), name=name),
        "rename": lambda: profiles.rename_profile("source", name),
        "distribution": lambda: install_distribution(str(source), name=name, create_alias=True),
        "distribution_tombstone": lambda: install_distribution(
            str(source), name=name, force=True, create_alias=True,
        ),
    }

    with pytest.raises(ValueError, match="main"):
        actions[operation]()

    assert target.exists() == tombstoned
    if tombstoned:
        assert named_profile_is_deleted(target)
        assert not list(target.iterdir())
    assert (source / "config.yaml").read_text() == config
    assert not profiles.find_alias_for_profile("main")



def _assert_confirmation_snapshot(source, legacy, tmp_path, monkeypatch, change):
    from hermes_cli import profile_cmd

    if change == "confirm_git":
        repository = tmp_path / "source.git"
        shutil.copytree(source, repository)

        def git(*args):
            return subprocess.run(
                ["git", "-c", "user.name=Distribution fixture", "-c", "user.email=fixture@example.invalid",
                 "-C", str(repository), *args],
                check=True, capture_output=True, text=True,
            ).stdout.strip()

        git("init", "--initial-branch=main")
        git("add", ".")
        git("commit", "-m", "Approved distribution")
        approved_commit = git("rev-parse", "HEAD")
    else:
        repository = source
    approved = []
    render = profile_cmd._render_distribution_plan

    def record_preview(plan):
        approved.append(plan)
        assert (plan.staged_dir / "SOUL.md").read_text() == "Updated distribution content"
        assert not plan.has_cron
        render(plan)

    def confirm_and_change_source(prompt):
        assert len(approved) == 1
        (repository / "SOUL.md").write_text("Unapproved content")
        (repository / "cron").mkdir()
        (repository / "cron" / "jobs.json").write_text('{"jobs": [{"prompt": "Unapproved job"}]}')
        write_manifest(repository, DistributionManifest(name="main", version="9.9.9"))
        if change == "confirm_git":
            git("add", ".")
            git("commit", "-m", "Move source after approval")
            assert git("rev-parse", "HEAD") != approved_commit
        return True

    monkeypatch.setattr(profile_cmd, "_render_distribution_plan", record_preview)
    monkeypatch.setattr(profile_cmd, "_confirm", confirm_and_change_source)
    profile_cmd._profile_install(SimpleNamespace(source=str(repository), install_name="main", force=True))
    assert (legacy / "SOUL.md").read_text() == "Updated distribution content"
    assert not (legacy / "cron" / "jobs.json").exists()
    installed_manifest = distributions.read_manifest(legacy)
    assert installed_manifest.version == approved[0].manifest.version
    assert installed_manifest.source == str(repository)
    assert not approved[0].staged_dir.exists()


def _assert_directory_modes(source, legacy, monkeypatch, finish):
    (source / "skills" / "empty").mkdir()
    (source / "skills" / "readonly").mkdir()
    (source / "skills" / "readonly" / "guide.md").write_text("Readable in an immutable directory")
    expected_modes = {"skills": 0o755, "skills/empty": 0o750, "skills/readonly": 0o555}
    for relative, mode in expected_modes.items():
        (source / relative).chmod(mode)
    stages = []
    real_plan = distributions.plan_install

    def record_staged_modes(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        stages.append({relative: stat.S_IMODE((plan.staged_dir / relative).stat().st_mode)
                       for relative in expected_modes})
        return plan

    monkeypatch.setattr(distributions, "plan_install", record_staged_modes)
    if finish == "install":
        installed = install_distribution(str(source), name="main", force=True)
    else:
        installed = update_distribution("main")
    assert stages == [expected_modes]
    assert {relative: stat.S_IMODE((legacy / relative).stat().st_mode)
            for relative in expected_modes} == expected_modes
    assert not list((legacy / "skills" / "empty").iterdir())
    assert not installed.staged_dir.exists()


def _assert_windows_source_contract(tmp_path, monkeypatch, change):
    from hermes_cli.profile_distribution_source import open_source
    import win32file

    parent = tmp_path / "source-parent"
    source = parent / "distribution"
    source.mkdir(parents=True)
    write_manifest(source, DistributionManifest(name="native-source"))
    payload = b"payload\n" * 262144
    (source / "SOUL.md").write_bytes(payload)
    (source / "empty.md").touch()

    if change in {"windows_junction", "windows_sharing"}:
        if change == "windows_junction":
            external = tmp_path / "private"
            external.mkdir()
            (external / "guide.md").write_bytes(b"SECRET outside distribution")
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(source / "skills"), str(external)],
                check=True, capture_output=True,
            )
            observed = []
            read_file = win32file.ReadFile

            def observe_read(*args, **kwargs):
                result = read_file(*args, **kwargs)
                observed.append(bytes(result[1]))
                return result

            monkeypatch.setattr(win32file, "ReadFile", observe_read)
            try:
                with pytest.raises(DistributionError, match="symlink|reparse"):
                    distributions.plan_install(str(source), tmp_path / "work")
                assert not any(b"SECRET outside distribution" in data for data in observed)
            finally:
                os.rmdir(source / "skills")
        else:
            writer = win32file.CreateFile(
                str(source / "SOUL.md"), win32file.GENERIC_WRITE,
                win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE | win32file.FILE_SHARE_DELETE,
                None, win32file.OPEN_EXISTING, 0, None,
            )
            try:
                with pytest.raises(DistributionError):
                    distributions.plan_install(str(source), tmp_path / "work")
            finally:
                win32file.CloseHandle(writer)
        assert not (tmp_path / "work" / "local").exists()
        moved = tmp_path / "released-parent"
        parent.rename(moved)
        installed = install_distribution(str(moved / "distribution"))
        assert (installed.target_dir / "SOUL.md").read_bytes() == payload
        return

    if change == "windows_ancestry":
        with open_source(source) as captured:
            with pytest.raises(PermissionError):
                parent.rename(tmp_path / "moved-parent")
            with captured.child("SOUL.md") as entry:
                assert entry.read(7) == payload[:7]
        parent.rename(tmp_path / "moved-parent")
        assert (tmp_path / "moved-parent" / "distribution" / "SOUL.md").read_bytes() == payload
        return

    if change in {"windows_unc", "windows_extended_unc"} and not tmp_path.drive.startswith("\\\\"):
        pytest.skip("Run with --basetemp on a real writable SMB share for native UNC coverage")
    with tempfile.TemporaryDirectory(dir=tmp_path) as workspace:
        exported = Path(workspace) / "distribution"
        shutil.copytree(source, exported)
        raw = str(exported)
        if change in {"windows_unc", "windows_extended_unc"}:
            assert raw.startswith("\\\\")
        if change == "windows_extended_unc":
            raw = "\\\\?\\UNC\\" + raw[2:]
        elif change == "windows_extended_path":
            raw = "\\\\?\\" + raw
        elif change == "windows_short_path":
            import win32api

            short_path = win32api.GetShortPathName(raw)
            if os.path.normcase(short_path) == os.path.normcase(raw):
                pytest.skip("The source volume must provide 8.3 aliases for native short-path coverage")
            assert Path(short_path).samefile(exported)
            raw = short_path
        with open_source(Path(raw)) as captured:
            for name, expected in (("SOUL.md", payload), ("empty.md", b"")):
                with captured.child(name) as entry:
                    chunks = []
                    while chunk := entry.read(1024 * 1024):
                        chunks.append(chunk)
                    assert b"".join(chunks) == expected
                    assert entry.read(1024 * 1024) == b""


def _observe_private_reads(private_file, monkeypatch):
    expected = private_file.stat()
    observed = []

    def record(descriptor):
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == (expected.st_dev, expected.st_ino):
            observed.append("external file opened for reading")

    def observe_open(original):
        def tracked(*args, **kwargs):
            opened = original(*args, **kwargs)
            mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
            if "r" in mode or "+" in mode:
                record(opened.fileno())
            return opened
        return tracked

    original_open = os.open

    def tracked_descriptor(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags & (os.O_WRONLY | os.O_RDWR) != os.O_WRONLY:
            record(descriptor)
        return descriptor

    monkeypatch.setattr(builtins, "open", observe_open(builtins.open))
    monkeypatch.setattr(io, "open", observe_open(io.open))
    monkeypatch.setattr(os, "open", tracked_descriptor)
    if os.name == "nt":
        import win32file
        original_read = win32file.ReadFile

        def tracked_native_read(*args, **kwargs):
            result = original_read(*args, **kwargs)
            if b"SECRET outside distribution" in result[1]:
                observed.append("external bytes read through native handle")
            return result

        monkeypatch.setattr(win32file, "ReadFile", tracked_native_read)
    return observed


@pytest.mark.parametrize("finish,change", [("rename", None), ("delete", None), ("retry_delete", None),
    ("install", "confirm_local"), ("install", "confirm_git"),
] + [
    (operation, change)
    for operation in ("install", "update")
    for change in (
        "delete", "rename", "tombstone", "recreate", "replacement", "source_replacement",
        "source_file", "source_file_link", "source_dir_link",
        "publication_rename", "publication_delete",
        "staged_file_link", "staged_dir_link", "staged_file_replace", "staged_file_write",
    )
] + [
    pytest.param(operation, "directory_modes", marks=marker, id=f"{operation}-directory_modes-{host}")
    for operation in ("install", "update")
    for host, marker in (("linux", pytest.mark.linux_only), ("macos", pytest.mark.macos_only))
] + [
    pytest.param("install", change, marks=pytest.mark.windows_only)
    for change in (
        "windows_eof", "windows_extended_path", "windows_short_path", "windows_unc", "windows_extended_unc",
        "windows_junction", "windows_sharing", "windows_ancestry",
    )
])
def test_legacy_main_profile_remains_manageable(profile_home, tmp_path, monkeypatch, finish, change):
    if change is not None and change.startswith("windows_"):
        _assert_windows_source_contract(tmp_path, monkeypatch, change)
        return
    legacy = profile_home / "profiles" / "main"
    legacy.mkdir(parents=True)
    config = "model:\n  provider: custom\n  default: local-model\n"
    (legacy / "config.yaml").write_text(config)
    if finish == "rename":
        profiles.create_wrapper_script("main")
    profiles.set_active_profile("main")

    assert profiles.resolve_profile_env("main") == str(legacy)
    assert any(profile.name == "main" for profile in profiles.list_profiles())
    clone = profiles.create_profile("copy", clone_from="main", no_alias=True)
    assert (clone / "config.yaml").is_file()

    source = profile_home / "profiles" / "distribution"
    source.mkdir()
    (source / "SOUL.md").write_text("Updated distribution content")
    (source / "skills").mkdir()
    (source / "skills" / "guide.md").write_text("Planned skill content")
    manifest = DistributionManifest(name="main", source=str(source))
    write_manifest(source, manifest)
    write_manifest(legacy, manifest)
    updated = update_distribution("main")
    assert updated.target_dir == legacy
    assert (legacy / "SOUL.md").read_text() == (source / "SOUL.md").read_text()
    assert (legacy / "config.yaml").read_text() == config

    if change in {"confirm_local", "confirm_git"}:
        _assert_confirmation_snapshot(source, legacy, tmp_path, monkeypatch, change)
        return
    if change == "directory_modes":
        _assert_directory_modes(source, legacy, monkeypatch, finish)
        return

    if change is not None:
        target_name = "copy" if change == "replacement" else "main"
        target = profiles.get_profile_dir(target_name)
        write_manifest(target, manifest)
        planned, release = Event(), Event()
        real_plan = distributions.plan_install
        publish = {
            "install": lambda: install_distribution(str(source), name=target_name.upper(), force=True),
            "update": lambda: update_distribution(target_name),
        }

        if change.startswith("staged_"):
            plans = []
            private_reads = []
            revalidate = distributions._revalidate_plan

            def pause_after_revalidation(plan):
                revalidate(plan)
                plans.append(plan)
                planned.set()
                assert release.wait(10), "test did not release publication"

            monkeypatch.setattr(distributions, "_revalidate_plan", pause_after_revalidation)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(publish[finish])
                try:
                    assert planned.wait(10), "publication did not reach revalidation"
                    staged = plans[0].staged_dir
                    root_inode = staged.stat().st_ino
                    external = tmp_path / "private"
                    if change == "staged_dir_link":
                        external.mkdir()
                        (external / "guide.md").write_text("SECRET outside distribution")
                        shutil.rmtree(staged / "skills")
                        (staged / "skills").symlink_to(external, target_is_directory=True)
                    elif change == "staged_file_link":
                        external.write_text("SECRET outside distribution")
                        (staged / "SOUL.md").unlink()
                        (staged / "SOUL.md").symlink_to(external)
                    elif change == "staged_file_replace":
                        external.write_text("Unapproved replacement")
                        external.replace(staged / "SOUL.md")
                    else:
                        (staged / "SOUL.md").write_text("Unapproved overwrite")
                    assert staged.stat().st_ino == root_inode
                    if change in {"staged_file_link", "staged_dir_link"}:
                        private_file = external / "guide.md" if change == "staged_dir_link" else external
                        private_reads = _observe_private_reads(private_file, monkeypatch)
                finally:
                    release.set()
                try:
                    pending.result(timeout=10)
                except DistributionError:
                    pass
            assert not private_reads
            assert (target / "SOUL.md").read_text() == "Updated distribution content"
            assert (target / "skills" / "guide.md").read_text() == "Planned skill content"
            assert not plans[0].staged_dir.exists()
            return

        if change.startswith("publication_"):
            from hermes_cli import profiles_lifecycle

            (source / "new_entry.md").write_text("Published while deletion waits")
            entered, attempted = Event(), Event()
            real_copy = distributions._copy_dist_payload
            real_lock = profiles.profile_lifecycle_lock

            def pause_publication(*args, **kwargs):
                entered.set()
                assert release.wait(10)
                real_copy(*args, **kwargs)

            @contextmanager
            def observe_lifecycle_lock():
                # A nonblocking probe proves exclusion without sleep-based assertions.
                available = profiles_lifecycle._THREAD_LOCK.acquire(blocking=False)
                if available:
                    profiles_lifecycle._THREAD_LOCK.release()
                try:
                    assert not available, "profile mutation can interleave with payload publication"
                finally:
                    attempted.set()
                with real_lock():
                    yield

            mutation = {
                "publication_rename": lambda: profiles.rename_profile("main", "previous"),
                "publication_delete": lambda: profiles.delete_profile("main", yes=True),
            }[change]
            monkeypatch.setattr(distributions, "_copy_dist_payload", pause_publication)
            monkeypatch.setattr(profiles, "profile_lifecycle_lock", observe_lifecycle_lock)
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(publish[finish])
                try:
                    assert entered.wait(10)
                    changed = pool.submit(mutation)
                    assert attempted.wait(10)
                finally:
                    release.set()
                pending.result(timeout=10)
                changed.result(timeout=10)
            assert not target.exists()
            if change == "publication_rename":
                assert (profile_home / "profiles" / "previous" / "SOUL.md").read_text() == (source / "SOUL.md").read_text()
            return

        def pause_after_plan(*args, **kwargs):
            plan = real_plan(*args, **kwargs)
            planned.set()
            assert release.wait(10), "test did not release the planned installation"
            return plan

        def recreate():
            profiles.delete_profile(target_name, yes=True)
            target.mkdir()
            clear_named_profile_deleted(target)

        def replace_target():
            profiles.delete_profile(target_name, yes=True)
            profiles.create_profile(target_name, no_alias=True, no_skills=True)

        def replace_source():
            profiles.rename_profile("distribution", "previous_source")
            profiles.create_profile("distribution", no_alias=True, no_skills=True)
            write_manifest(source, manifest)
            (source / "SOUL.md").write_text("Unplanned source content")

        def replace_payload():
            replacement = tmp_path / "replacement.md"
            replacement.write_text("Unplanned source content")
            replacement.replace(source / "SOUL.md")
            (source / "skills" / "guide.md").write_text("Unplanned skill content")
            write_manifest(source, DistributionManifest(name="other", version="9.9.9"))

        def link_payload(entry):
            original = source / entry
            external = tmp_path / "private"
            if original.is_dir():
                shutil.rmtree(original)
                external.mkdir()
                (external / "guide.md").write_text("Private external content")
                original.symlink_to(external, target_is_directory=True)
            else:
                original.unlink()
                external.write_text("Private external content")
                original.symlink_to(external)

        mutations = {
            "delete": lambda: profiles.delete_profile(target_name, yes=True),
            "rename": lambda: profiles.rename_profile(target_name, "previous"),
            "tombstone": lambda: mark_named_profile_deleted(target),
            "recreate": recreate,
            "replacement": replace_target,
            "source_replacement": replace_source,
            "source_file": replace_payload,
            "source_file_link": lambda: link_payload("SOUL.md"),
            "source_dir_link": lambda: link_payload("skills"),
        }
        monkeypatch.setattr(distributions, "plan_install", pause_after_plan)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(publish[finish])
            try:
                assert planned.wait(10), "installation did not reach the planning barrier"
                source_inode = source.stat().st_ino
                mutations[change]()
                if change in {"source_file", "source_file_link", "source_dir_link"}:
                    assert source.stat().st_ino == source_inode
                existed = target.exists()
                contents = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            finally:
                release.set()
            if change.startswith("source_"):
                installed = pending.result(timeout=10)
                assert installed.provenance == installed.manifest.source == str(source)
                assert installed.manifest.version == manifest.version
                assert (target / "SOUL.md").read_text() == "Updated distribution content"
                assert (target / "skills" / "guide.md").read_text() == "Planned skill content"
                assert not installed.staged_dir.exists()
                return
            else:
                with pytest.raises(DistributionError, match="changed"):
                    pending.result(timeout=10)
        assert target.exists() == existed
        assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == contents
        return

    if finish == "rename":
        renamed = profiles.rename_profile("main", "renamed")
        assert profiles.resolve_profile_env("renamed") == str(renamed)
        assert (renamed / "config.yaml").read_text() == config
        assert profiles.find_alias_for_profile("main") is None
    else:
        if finish == "retry_delete":
            def fail_removal(*args):
                raise OSError("temporary removal failure")

            with monkeypatch.context() as failing:
                failing.setattr(profiles, "_rmtree_with_retry", fail_removal)
                with pytest.raises(RuntimeError, match="Could not remove profile directory"):
                    profiles.delete_profile("main", yes=True)
            assert legacy.exists() and named_profile_is_deleted(legacy)
        profiles.delete_profile("main", yes=True)

    assert not legacy.exists()
    assert (clone / "config.yaml").is_file()
