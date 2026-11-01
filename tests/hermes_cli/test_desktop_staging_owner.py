"""Desktop staging belongs to one live builder and one host architecture."""

import os
from pathlib import Path

import pytest

from hermes_cli import main_desktop


def test_staging_gc_keeps_another_live_owner(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    desktop.mkdir()
    live = desktop / ".staging-42-arm64-aaaaaaaaaaaa-1"
    dead = desktop / ".staging-43-arm64-bbbbbbbbbbbb-1"
    # Left by builds from before the arch/scope fields: same leading owner pid.
    legacy_live = desktop / ".staging-44-1758000000"
    legacy_dead = desktop / ".staging-4242-1758000000"
    unowned = desktop / ".staging-old"
    for candidate in (live, dead, legacy_live, legacy_dead, unowned):
        candidate.mkdir()

    monkeypatch.setattr(main_desktop.os, "getpid", lambda: 99)
    monkeypatch.setattr(main_desktop, "_desktop_staging_owner_alive", lambda pid: pid in {42, 44})
    monkeypatch.setattr(main_desktop, "_desktop_staging_arch", lambda: "arm64")

    created = main_desktop._desktop_staging_dir(desktop)

    assert live.exists() and legacy_live.exists()
    assert not dead.exists()
    assert not legacy_dead.exists(), "an old-layout tree of a dead builder must still be reclaimed"
    assert not unowned.exists()
    assert created.name.startswith(".staging-99-arm64-")


def test_staging_identity_binds_path_affecting_user_data_roots(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(main_desktop.os, "getpid", lambda: 99)
    monkeypatch.setattr(main_desktop, "_desktop_staging_arch", lambda: "x64")
    monkeypatch.setattr(main_desktop._time_mod, "time", lambda: 1)

    first = main_desktop._desktop_staging_dir(
        desktop, env={"HERMES_HOME": str(tmp_path / "one")}
    )
    second = main_desktop._desktop_staging_dir(
        desktop, env={"HERMES_HOME": str(tmp_path / "two")}
    )

    assert first.name != second.name
    assert first.name.endswith("-1")
    assert second.name.endswith("-1")


def _two_linux_trees(release: Path, *, newer: str) -> dict:
    trees = {}
    for arch, name in (("arm64", "linux-arm64-unpacked"), ("x64", "linux-unpacked")):
        exe = release / name / "hermes"
        exe.parent.mkdir(parents=True)
        exe.write_text(arch, encoding="utf-8")
        trees[arch] = exe
    older = trees["x64" if newer == "arm64" else "arm64"]
    os.utime(older, (1_000_000, 1_000_000))
    return trees


@pytest.mark.parametrize(
    "release_parts, host, newer",
    [
        (("release",), "arm64", "x64"),
        (("release",), "x64", "arm64"),
        # The host's own staging dir names its arch; every candidate shares that ancestor.
        ((".staging-9-arm64-0123456789ab-1",), "arm64", "x64"),
        ((".staging-9-x64-0123456789ab-1",), "x64", "arm64"),
        # A checkout whose path says arm64 on an x64 host (picked the newer arm64 tree before).
        (("linux-arm64-src", "apps", "desktop", "release"), "x64", "arm64"),
    ],
)
def test_packaged_executable_prefers_matching_architecture(tmp_path, monkeypatch, release_parts, host, newer):
    release = tmp_path.joinpath(*release_parts)
    trees = _two_linux_trees(release, newer=newer)

    monkeypatch.setattr(main_desktop.sys, "platform", "linux")
    monkeypatch.setattr(main_desktop, "_desktop_staging_arch", lambda: host)

    assert main_desktop._desktop_packaged_executable_in(release) == trees[host]


@pytest.mark.parametrize("host", ["arm64", "x64"])
def test_mac_bundle_arch_comes_from_the_output_dir(tmp_path, monkeypatch, host):
    release = tmp_path / ".staging-9-arm64-0123456789ab-1"
    exes = {}
    for arch, name in (("arm64", "mac-arm64"), ("x64", "mac")):
        exe = release / name / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
        exe.parent.mkdir(parents=True)
        exe.write_text(arch, encoding="utf-8")
        exes[arch] = exe
    os.utime(exes[host], (1_000_000, 1_000_000))  # the right one is the older tree

    monkeypatch.setattr(main_desktop.sys, "platform", "darwin")
    monkeypatch.setattr(main_desktop, "_desktop_staging_arch", lambda: host)

    assert main_desktop._desktop_packaged_executable_in(release) == exes[host]
