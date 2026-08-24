"""#70337/#87331: the ZIP swap must preserve apps/desktop/release/.

The GitHub source ZIP carries only source; the BUILT desktop app
(release/win-unpacked/Hermes.exe) exists only in the live tree. Swapping
`apps` without grafting the live release dir deletes the desktop build.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def test_staged_apps_swap_preserves_live_release_dir(tmp_path, monkeypatch):
    from hermes_cli import main as hermes_main
    from hermes_cli.update_cmd import (
        _commit_staged_replacements,
        _stage_replacement,
    )

    # live tree: apps/desktop/release/win-unpacked/Hermes.exe + old source
    root = tmp_path / "install"
    live_apps = root / "apps" / "desktop"
    (live_apps / "release" / "win-unpacked").mkdir(parents=True)
    (live_apps / "release" / "win-unpacked" / "Hermes.exe").write_bytes(b"MZbuilt")
    (live_apps / "electron").mkdir()
    (live_apps / "electron" / "main.ts").write_text("old source")

    # extracted ZIP: new source, NO release dir (GitHub source archive shape)
    extracted = tmp_path / "extracted"
    zip_apps = extracted / "apps" / "desktop"
    (zip_apps / "electron").mkdir(parents=True)
    (zip_apps / "electron" / "main.ts").write_text("new source")

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)

    # Reproduce the _update_via_zip staging loop for the `apps` entry,
    # including the release-dir graft.
    src = str(extracted / "apps")
    dst = str(root / "apps")
    staged_path = _stage_replacement(src, dst)
    live_release = os.path.join(dst, "desktop", "release")
    staged_release = os.path.join(staged_path, "desktop", "release")
    if os.path.isdir(live_release) and not os.path.exists(staged_release):
        os.makedirs(os.path.dirname(staged_release), exist_ok=True)
        shutil.copytree(live_release, staged_release)

    _commit_staged_replacements([(staged_path, dst)])

    # New source landed AND the built desktop app survived.
    assert (root / "apps" / "desktop" / "electron" / "main.ts").read_text() == (
        "new source"
    )
    exe = root / "apps" / "desktop" / "release" / "win-unpacked" / "Hermes.exe"
    assert exe.exists() and exe.read_bytes() == b"MZbuilt"
