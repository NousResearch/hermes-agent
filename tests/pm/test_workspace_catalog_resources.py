"""A generated PM workspace retains its offline plugin catalog and removals."""
import pytest

from hermes_cli.plugin_catalog import load_catalog, load_removed_list
from pm import workspace


@pytest.mark.parametrize("include", ['["pm"]', '["*"]'])
def test_generated_workspace_keeps_offline_plugin_catalog(tmp_path, include):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\n'
        f'[tool.setuptools.packages.find]\ninclude={include}\n', encoding="utf-8",
    )
    catalog = source / "plugin-catalog"
    catalog.mkdir()
    (catalog / "example.yaml").write_text(
        'name: example\nrepo: https://github.com/example/plugin\n'
        f'sha: "{"a" * 40}"\ndescription: Offline fixture\nmaintainer: Example\n',
        encoding="utf-8",
    )
    (catalog / "removed.yaml").write_text(
        'removed:\n  - name: withdrawn\n    repo: https://github.com/example/withdrawn\n'
        '    reason: Withdrawn by maintainer\n', encoding="utf-8",
    )
    for ignored in (".git", "__pycache__", "node_modules"):
        (catalog / ignored).mkdir()
        (catalog / ignored / "private.txt").write_text("not a catalog input", encoding="utf-8")
    (catalog / ".env").write_text("NOT_A_REAL_SECRET=fixture", encoding="utf-8")
    destination = tmp_path / "workspace"

    workspace._generate_pyproject([], destination, source=source)

    entries = load_catalog(destination / "plugin-catalog")
    assert [entry.name for entry in entries] == ["example"]
    assert entries[0].sha == "a" * 40
    assert [entry.name for entry in load_removed_list(destination / "plugin-catalog")] == ["withdrawn"]
    for ignored in (".git", "__pycache__", "node_modules", ".env"):
        assert not (destination / "plugin-catalog" / ignored).exists()


def test_core_without_catalog_still_generates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\n'
        '[tool.setuptools.packages.find]\ninclude=["pm"]\n', encoding="utf-8",
    )
    destination = tmp_path / "workspace"
    workspace._generate_pyproject([], destination, source=source)
    assert load_catalog(destination / "plugin-catalog") == []
    assert load_removed_list(destination / "plugin-catalog") == []


def test_catalog_copy_failure_is_not_silently_dropped(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "plugin-catalog").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\n'
        '[tool.setuptools.packages.find]\ninclude=["pm"]\n', encoding="utf-8",
    )

    def unreadable(*args, **kwargs):
        raise PermissionError("catalog unreadable")

    monkeypatch.setattr(workspace.shutil, "copytree", unreadable)
    with pytest.raises(PermissionError, match="catalog unreadable"):
        workspace._generate_pyproject([], tmp_path / "workspace", source=source)
