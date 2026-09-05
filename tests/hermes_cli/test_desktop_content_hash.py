"""Desktop content-hash coverage for workspace build inputs."""

from pathlib import Path

from hermes_cli.main_desktop import _compute_desktop_content_hash


def _desktop_workspace(root: Path) -> None:
    (root / "apps" / "desktop" / "src").mkdir(parents=True)
    (root / "apps" / "shared" / "src").mkdir(parents=True)
    (root / "apps" / "desktop" / "src" / "app.tsx").write_text(
        "export const app = 1\n", encoding="utf-8"
    )
    (root / "apps" / "shared" / "src" / "shared.ts").write_text(
        "export const shared = 1\n", encoding="utf-8"
    )
    (root / "package.json").write_text("{}\n", encoding="utf-8")
    (root / ".gitignore").write_text("apps/desktop/dist/\n", encoding="utf-8")


def test_desktop_hash_changes_with_shared_workspace_source(tmp_path: Path) -> None:
    _desktop_workspace(tmp_path)
    before = _compute_desktop_content_hash(tmp_path)

    (tmp_path / "apps" / "shared" / "src" / "shared.ts").write_text(
        "export const shared = 2\n", encoding="utf-8"
    )

    assert _compute_desktop_content_hash(tmp_path) != before


def test_desktop_hash_still_ignores_build_output(tmp_path: Path) -> None:
    _desktop_workspace(tmp_path)
    before = _compute_desktop_content_hash(tmp_path)

    dist = tmp_path / "apps" / "desktop" / "dist"
    dist.mkdir()
    (dist / "bundle.js").write_text("generated\n", encoding="utf-8")

    assert _compute_desktop_content_hash(tmp_path) == before
