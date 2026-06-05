from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tui_finds_bundled_entry_js(tmp_path):
    """_find_bundled_tui finds entry.js bundled in the package."""
    tui_dist = tmp_path / "hermes_cli" / "tui_dist"
    tui_dist.mkdir(parents=True)
    entry = tui_dist / "entry.js"
    entry.write_text("// bundled TUI", encoding="utf-8")

    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is not None
    assert result.name == "entry.js"


def test_tui_returns_none_when_no_bundle(tmp_path):
    """_find_bundled_tui returns None when no bundle exists."""
    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is None


def test_pypi_workflow_bundles_tui_esm_package_json():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml"
    ).read_text(encoding="utf-8")

    assert (
        'printf \'{"type":"module"}\\n\' > hermes_cli/tui_dist/package.json'
        in workflow
    )
    assert "test -f hermes_cli/tui_dist/package.json" in workflow
