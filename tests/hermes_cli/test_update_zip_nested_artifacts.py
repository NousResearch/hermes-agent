"""Real ZIP stage/commit preserves nested artifacts, including rollback.

Credit eman717, #90495 comment5557504436, for the complete artifact set and
hardlink-first graft; #70337/#87331 supplied the earlier release-only graft.
"""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from hermes_cli import update_cmd, update_cmd_zip as uz


ARTIFACTS = (
    "apps/desktop/release/win-unpacked/Hermes.exe",
    "apps/desktop/dist/index.html",
    "apps/desktop/node_modules/electron/index.js",
    "hermes_cli/web_dist/index.html",
)


def seed(root, *, artifacts=True):
    for name in ("apps/desktop/source.js", "hermes_cli/source.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old", encoding="utf-8")
    if artifacts:
        for name in ARTIFACTS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")


def extracted(root):
    seed(root, artifacts=False)
    for path in root.rglob("source.*"):
        path.write_text("new", encoding="utf-8")


@pytest.mark.parametrize("rollback", [False, True])
def test_real_stage_commit_preserves_every_artifact(tmp_path, monkeypatch, rollback):
    live, new = tmp_path / "live", tmp_path / "extracted"
    seed(live)
    extracted(new)
    staged = uz._stage_entries(str(new), ["apps", "hermes_cli"], str(live))
    # Grafts share file identity to avoid copying Electron's entire tree.
    for name in ARTIFACTS:
        top, nested = name.split("/", 1)
        assert os.path.samefile(live / name, Path(dict((d, s) for s, d in staged)[str(live / top)]) / nested)
    if rollback:
        original = os.rename

        def rename(source, destination):
            if str(source).endswith("hermes_cli.hermes-update-staging"):
                raise PermissionError("injected second swap failure")
            return original(source, destination)

        monkeypatch.setattr(os, "rename", rename)
        with pytest.raises(PermissionError):
            uz._commit_staged_replacements(staged)
        uz._discard_staged(staged)
    else:
        uz._commit_staged_replacements(staged)
    for name in ARTIFACTS:
        assert (live / name).read_text(encoding="utf-8") == name
    assert (live / "apps/desktop/source.js").read_text() == ("old" if rollback else "new")
    assert not list(live.glob("*.hermes-update-*"))


def test_link_failure_falls_back_to_copy_without_losing_artifacts(tmp_path, monkeypatch):
    live, new = tmp_path / "live", tmp_path / "extracted"
    seed(live)
    extracted(new)
    monkeypatch.setattr(os, "link", lambda *a, **kw: (_ for _ in ()).throw(OSError("no hardlinks")))
    uz._commit_staged_replacements(uz._stage_entries(str(new), ["apps", "hermes_cli"], str(live)))
    for name in ARTIFACTS:
        assert (live / name).read_text(encoding="utf-8") == name


def test_full_zip_download_extract_stage_guard_and_swap(tmp_path, monkeypatch):
    live = tmp_path / "live"
    seed(live)
    ignores = "\n".join("/" + name.rsplit("/", 1)[0] + "/" for name in ARTIFACTS) + "\n"
    (live / ".gitignore").write_text(ignores, encoding="utf-8")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"]):
        subprocess.run(["git", *args], cwd=live, check=True, capture_output=True)
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("hermes-agent-main/apps/desktop/source.js", "new")
        archive.writestr("hermes-agent-main/hermes_cli/source.py", "new")
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=live, sys=sys))
    uz._download_and_swap_zip("main", source_zip.as_uri())
    for name in ARTIFACTS:
        assert (live / name).read_text(encoding="utf-8") == name
    assert (live / "hermes_cli/source.py").read_text() == "new"


@pytest.mark.parametrize("previously_built", [False, True])
def test_never_built_install_and_new_archive_artifact_are_respected(tmp_path, previously_built):
    live, new = tmp_path / "live", tmp_path / "extracted"
    seed(live, artifacts=previously_built)
    extracted(new)
    source_asset = new / "hermes_cli/web_dist/index.html"
    source_asset.parent.mkdir()
    source_asset.write_text("newly-shipped", encoding="utf-8")
    uz._commit_staged_replacements(uz._stage_entries(str(new), ["apps", "hermes_cli"], str(live)))
    assert (live / "hermes_cli/web_dist/index.html").read_text() == "newly-shipped"
    assert (live / "apps/desktop/release").exists() is previously_built


@pytest.mark.parametrize("line", [
    " M apps/desktop/dist/index.html", "?? apps/desktop/dist-notes/user.txt",
    "!! apps/desktop/dist-other/", " R apps/desktop/dist/x -> notes.txt",
])
def test_nested_preservation_does_not_admit_unrelated_or_modified_paths(line):
    assert not uz._is_zip_preserved_entry_status_line(line)


def test_failed_graft_discards_staging_and_keeps_live_install(tmp_path, monkeypatch):
    live, new = tmp_path / "live", tmp_path / "extracted"
    seed(live)
    extracted(new)

    def fail_copy(*args, **kwargs):
        raise OSError("artifact unavailable")

    monkeypatch.setattr(uz, "_link_or_copy_artifact", fail_copy)
    with pytest.raises(OSError):
        uz._stage_entries(str(new), ["apps", "hermes_cli"], str(live))
    for name in ARTIFACTS:
        assert (live / name).read_text(encoding="utf-8") == name
    assert (live / "apps/desktop/source.js").read_text() == "old"
    assert not list(live.glob("*.hermes-update-*"))


@pytest.mark.windows_only
def test_exclusively_open_artifact_can_be_grafted_without_read_copy(tmp_path):
    import ctypes
    from ctypes import wintypes

    live, new = tmp_path / "live", tmp_path / "extracted"
    seed(live)
    extracted(new)
    source = live / ARTIFACTS[0]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(source), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    try:
        staged = uz._stage_entries(str(new), ["apps", "hermes_cli"], str(live))
    finally:
        assert kernel.CloseHandle(handle)
    destination = Path(staged[0][0]) / ARTIFACTS[0].split("/", 1)[1]
    assert os.path.samefile(source, destination)
    uz._commit_staged_replacements(staged)
    assert source.read_text(encoding="utf-8") == ARTIFACTS[0]
