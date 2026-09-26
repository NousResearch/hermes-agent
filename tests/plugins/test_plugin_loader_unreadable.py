"""One mode-000 / ACL-denied child under a plugin root must not abort the listing (#111804)."""

from __future__ import annotations

from pathlib import Path

from plugins import plugin_loader


def _deny(monkeypatch, denied: Path) -> None:
    """chmod 000 does not bite as root, so fail the probes of the denied child's ``__init__.py``.

    ``Path.exists``/``is_dir`` route through ``os.path`` on 3.12+, which swallows
    ``PermissionError`` into False; the loader's own guard sees the error only when the
    pathlib probe raises, so fail both layers the way a real ACL denial can."""
    real_exists, real_is_dir = Path.exists, Path.is_dir

    def refuse(method):
        def probe(self, *args, **kwargs):
            if self.parent == denied or self == denied:
                raise PermissionError(13, "Permission denied", str(self))
            return method(self, *args, **kwargs)
        return probe

    monkeypatch.setattr(Path, "exists", refuse(real_exists))
    monkeypatch.setattr(Path, "is_dir", refuse(real_is_dir))


def test_iter_plugin_dirs_skips_unreadable_child(tmp_path, monkeypatch):
    for name in ("denied", "good"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    _deny(monkeypatch, tmp_path / "denied")

    assert plugin_loader.iter_plugin_dirs(tmp_path) == [tmp_path / "good"]
