"""New profile targets must not reuse the default session namespace."""

import tarfile
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

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


@pytest.mark.parametrize("finish,change", [("rename", None), ("delete", None), ("retry_delete", None)] + [
    (operation, change)
    for operation in ("install", "update")
    for change in (
        "delete", "rename", "tombstone", "recreate", "replacement", "source_replacement",
        "source_file", "source_file_link", "source_dir_link",
        "publication_rename", "publication_delete",
    )
])
def test_legacy_main_profile_remains_manageable(profile_home, tmp_path, monkeypatch, finish, change):
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
