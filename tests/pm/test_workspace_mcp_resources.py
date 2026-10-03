"""MCP catalog discovery survives the PM's non-package resource snapshot."""
import pytest

from hermes_cli import mcp_catalog
from pm import workspace


def _source(tmp_path, include):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\n'
        f'[tool.setuptools.packages.find]\ninclude={include}\n', encoding="utf-8",
    )
    return source


def _manifest(root, name="fixture"):
    path = root / name / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        f'manifest_version: 1\nname: {name}\ndescription: Local catalog fixture\n'
        'transport:\n  type: http\n  url: https://example.invalid/mcp\n'
        'auth:\n  type: none\n', encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("include", ['["pm"]', '["*"]'])
def test_generated_workspace_keeps_mcp_catalog(tmp_path, monkeypatch, include):
    source = _source(tmp_path, include)
    manifest = _manifest(source / "optional-mcps")
    for name in (".git", "node_modules", "__pycache__"):
        folder = manifest.parent / name
        folder.mkdir()
        (folder / "ignored.txt").write_text("runtime artifact", encoding="utf-8")
    (manifest.parent / ".env").write_text("TEST_ONLY=fixture", encoding="utf-8")
    monkeypatch.delenv("HERMES_OPTIONAL_MCPS", raising=False)
    monkeypatch.setattr(mcp_catalog, "__file__", str(source / "hermes_cli/mcp_catalog.py"))
    expected = mcp_catalog.get_entry("fixture")
    assert expected is not None  # The same valid fixture works before the snapshot.
    destination = tmp_path / "workspace"
    workspace._generate_pyproject([], destination, source=source)
    monkeypatch.setattr(mcp_catalog, "__file__", str(destination / "hermes_cli/mcp_catalog.py"))

    entries = mcp_catalog.list_catalog()
    assert [entry.name for entry in entries] == [expected.name]
    entry = mcp_catalog.get_entry("official/fixture")
    assert entry.transport == expected.transport
    assert entry.auth == expected.auth
    assert entry.manifest_path == destination / manifest.relative_to(source)
    for name in (".git", "node_modules", "__pycache__", ".env"):
        assert not (entry.manifest_path.parent / name).exists()

    # An explicit packaged catalog remains authoritative over bundled data.
    override = tmp_path / "external"
    _manifest(override, "override")
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(override))
    assert [entry.name for entry in mcp_catalog.list_catalog()] == ["override"]


@pytest.mark.parametrize("present", [False, True])
def test_optional_mcp_catalog_absence_and_copy_failure(tmp_path, monkeypatch, present):
    source = _source(tmp_path, '["pm"]')
    destination = tmp_path / "workspace"
    if present:
        _manifest(source / "optional-mcps")

        def unreadable(*args, **kwargs):
            raise PermissionError("MCP catalog unreadable")

        monkeypatch.setattr(workspace.shutil, "copytree", unreadable)
        with pytest.raises(PermissionError, match="MCP catalog unreadable"):
            workspace._generate_pyproject([], destination, source=source)
    else:
        workspace._generate_pyproject([], destination, source=source)
        monkeypatch.delenv("HERMES_OPTIONAL_MCPS", raising=False)
        monkeypatch.setattr(mcp_catalog, "__file__", str(destination / "hermes_cli/mcp_catalog.py"))
        assert mcp_catalog.list_catalog() == []
