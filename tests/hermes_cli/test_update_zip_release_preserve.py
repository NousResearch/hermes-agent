"""#70337/#87331/#90495: the ZIP swap must preserve the gitignored build outputs.

The GitHub source ZIP carries only source; the BUILT desktop app
(release/win-unpacked/Hermes.exe), its renderer bundle (dist/), its own
node_modules and the dashboard assets (hermes_cli/web_dist/) exist only in
the live tree. Swapping `apps` / `hermes_cli` without grafting them deletes
them — and the dirty-tree guard must admit their ``!!`` status lines, or the
fallback refuses every install that has them.
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
    (live_apps / "electron" / "main.ts").write_text("old source", encoding="utf-8")

    # extracted ZIP: new source, NO release dir (GitHub source archive shape)
    extracted = tmp_path / "extracted"
    zip_apps = extracted / "apps" / "desktop"
    (zip_apps / "electron").mkdir(parents=True)
    (zip_apps / "electron" / "main.ts").write_text("new source", encoding="utf-8")

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
    assert (root / "apps" / "desktop" / "electron" / "main.ts").read_text(encoding="utf-8") == (
        "new source"
    )
    exe = root / "apps" / "desktop" / "release" / "win-unpacked" / "Hermes.exe"
    assert exe.exists() and exe.read_bytes() == b"MZbuilt"


def test_zip_swap_keeps_every_nested_build_output_and_the_guard_admits_them(tmp_path):
    from hermes_cli.update_cmd_zip import (
        _commit_staged_replacements,
        _is_zip_preserved_entry_status_line,
        _stage_entries,
    )

    root = tmp_path / "install"
    outputs = {
        "apps/desktop/release/win-unpacked/Hermes.exe": b"MZbuilt",
        "apps/desktop/node_modules/electron/index.js": b"electron",
        "apps/desktop/dist/index.html": b"live renderer",
        "hermes_cli/web_dist/index.html": b"<dashboard>",
    }
    for rel, data in outputs.items():
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_bytes(data)
    (root / "hermes_cli" / "__pycache__").mkdir()
    (root / "hermes_cli" / "x.py").write_text("old", encoding="utf-8")

    # Extracted ZIP: new source, none of the outputs — except dist/, which a future archive may ship.
    extracted = tmp_path / "extracted"
    (extracted / "apps" / "desktop" / "dist").mkdir(parents=True)
    (extracted / "apps" / "desktop" / "dist" / "index.html").write_bytes(b"shipped renderer")
    (extracted / "hermes_cli").mkdir()
    (extracted / "hermes_cli" / "x.py").write_text("new", encoding="utf-8")

    _commit_staged_replacements(_stage_entries(str(extracted), ["apps", "hermes_cli"], str(root)))

    assert (root / "hermes_cli" / "x.py").read_text(encoding="utf-8") == "new"
    for rel, data in outputs.items():
        if rel.startswith("apps/desktop/dist/"):
            continue
        assert (root / rel).read_bytes() == data, rel
    # What the ZIP ships wins over the live copy; the graft never clobbers it.
    assert (root / "apps" / "desktop" / "dist" / "index.html").read_bytes() == b"shipped renderer"
    assert not (root / "hermes_cli" / "__pycache__").exists()  # regenerable, dropped with the old tree

    # The dirty-tree guard sees these outputs as ``!!`` lines; they must not refuse the swap.
    for line in ("!! apps/desktop/release/", "!! apps/desktop/dist/", "!! apps/desktop/node_modules/",
                 "!! apps/desktop/build/", "!! hermes_cli/web_dist/", "!! hermes_cli/__pycache__/",
                 "!! __pycache__/", "!! ui-tui/dist/", "!! ui-tui/packages/hermes-ink/dist/",
                 "!! scripts/whatsapp-bridge/node_modules/", "!! web/node_modules/", "!! tests-js/node_modules/"):
        assert _is_zip_preserved_entry_status_line(line), line
    # ...while other gitignored data, untracked files and renames into those dirs still block.
    for line in ("!! apps/desktop/notes.local", "?? apps/desktop/release/", "!! hermes_cli/web_dist_backup/",
                 "R  src/x -> apps/desktop/release/x"):
        assert not _is_zip_preserved_entry_status_line(line), line


def test_guard_admits_a_real_installs_ignored_set_and_blocks_only_what_the_swap_destroys(tmp_path):
    """Real git + the repo's own .gitignore, seeded with every ``!!`` line a real installer-made install
    carries (markers written at the checkout root by update/install, the egg-info, every nested build
    output). Blocking on any of them kept the ZIP fallback — and the graft — unreachable (#90495)."""
    import subprocess

    from hermes_cli.update_cmd_zip import _zip_overlay_block_reason

    root = tmp_path / "install"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    repo = Path(__file__).resolve().parents[2]
    (root / "ui-tui").mkdir()
    for rel in (".gitignore", "ui-tui/.gitignore"):  # ui-tui/dist is ignored by the nested file
        shutil.copy(repo / rel, root / rel)
    for tracked in ("apps/desktop/package.json", "hermes_cli/main.py", "scripts/whatsapp-bridge/index.js",
                    "ui-tui/package.json", "web/package.json", "tests-js/a.test.ts", "run_agent.py"):
        (root / tracked).parent.mkdir(parents=True, exist_ok=True)
        (root / tracked).write_text("src", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
                   check=True)
    for ignored in (".bytecode-fingerprint", ".hermes-bootstrap-complete", ".install_method",
                    "hermes_agent.egg-info/PKG-INFO", "hermes_cli/__pycache__/main.pyc", "__pycache__/x.pyc",
                    "apps/desktop/release/win-unpacked/Hermes.exe", "apps/desktop/dist/index.html",
                    "apps/desktop/build/icon.ico", "apps/desktop/node_modules/electron/index.js",
                    "hermes_cli/web_dist/index.html", "ui-tui/dist/entry.js", "ui-tui/node_modules/x/index.js",
                    "ui-tui/packages/hermes-ink/dist/index.js", "web/node_modules/x/index.js",
                    "tests-js/node_modules/x/index.js", "scripts/whatsapp-bridge/node_modules/x/index.js",
                    "venv/lib.py", "node_modules/x/index.js", ".env"):
        (root / ignored).parent.mkdir(parents=True, exist_ok=True)
        (root / ignored).write_text("artifact", encoding="utf-8")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "-uall", "--ignored=matching"],
                            capture_output=True, text=True, check=True).stdout
    assert status.count("!!") >= 19 and "??" not in status, status  # the fixture really is all-ignored

    assert _zip_overlay_block_reason(root) is None
    # The pre-swap re-check knows the ZIP's real entry set: a root entry it ships would be replaced.
    shipped = {"apps", "hermes_cli", "scripts", "ui-tui", "web", "tests-js", "run_agent.py"}
    assert _zip_overlay_block_reason(root, shipped=shipped) is None
    assert _zip_overlay_block_reason(root, shipped=shipped | {"hermes_agent.egg-info"}) is not None
    # Gitignored user data under a shipped dir is destroyed by the swap: still refused.
    (root / "apps" / "desktop" / ".env").write_text("mine", encoding="utf-8")
    assert _zip_overlay_block_reason(root) is not None
