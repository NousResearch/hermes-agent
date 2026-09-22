"""One mode-000 / ACL-denied child under a plugin root must not abort the listing (#111804)."""

from __future__ import annotations

from pathlib import Path

from plugins import plugin_loader


def _deny(monkeypatch, denied: Path) -> None:
    """chmod 000 does not bite as root, so fail the stat of the denied child's ``__init__.py``."""
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.parent == denied:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_iter_plugin_dirs_skips_unreadable_child(tmp_path, monkeypatch, caplog):
    for name in ("denied", "good"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    _deny(monkeypatch, tmp_path / "denied")

    with caplog.at_level("WARNING", logger="plugins.plugin_loader"):
        assert plugin_loader.iter_plugin_dirs(tmp_path) == [tmp_path / "good"]
    assert "Skipping unreadable plugin directory" in caplog.text
