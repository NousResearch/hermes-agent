"""The migration check prunes excluded trees before traversing them."""
from scripts import check_compat_pointers as checker


def test_inventory_does_not_descend_into_excluded_trees(tmp_path, monkeypatch):
    included = tmp_path / "hermes_cli"
    included.mkdir()
    (included / "entry.py").touch()
    nested = included / "node_modules"
    nested.mkdir()
    (nested / "not_source.py").touch()
    real_walk = checker.os.walk
    visited = []

    def walk(*args, **kwargs):
        for directory, dirs, files in real_walk(*args, **kwargs):
            visited.append(directory)
            yield directory, dirs, files

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker.os, "walk", walk)
    assert list(checker._py_files()) == [included / "entry.py"]
    assert str(nested) not in visited
