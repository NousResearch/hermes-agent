"""#77432: purge site-packages dist-info that lack RECORD before uv reinstall."""

from __future__ import annotations

from pathlib import Path

import hermes_cli.update_cmd as update_cmd


def _make_dist(site: Path, name: str, version: str, *, with_record: bool) -> Path:
    dist_info = site / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    if with_record:
        (dist_info / "RECORD").write_text(f"{name}/__init__.py,,\n", encoding="utf-8")
    pkg = site / name.replace("-", "_")
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("# pkg\n", encoding="utf-8")
    return dist_info


def test_purge_removes_missing_record_and_keeps_healthy(tmp_path: Path):
    site = tmp_path / "site-packages"
    site.mkdir()
    _make_dist(site, "cryptography", "49.0.0", with_record=False)
    _make_dist(site, "healthy", "1.0.0", with_record=True)

    purged = update_cmd._purge_site_packages_missing_record(site)

    assert purged == ["cryptography"]
    assert not (site / "cryptography-49.0.0.dist-info").exists()
    assert not (site / "cryptography").exists()
    assert (site / "healthy-1.0.0.dist-info" / "RECORD").is_file()
    assert (site / "healthy" / "__init__.py").is_file()


def test_heal_scans_windows_and_posix_site_packages(tmp_path: Path, monkeypatch):
    root = tmp_path / "checkout"
    win_sp = root / "venv" / "Lib" / "site-packages"
    posix_sp = root / "venv" / "lib" / "python3.11" / "site-packages"
    win_sp.mkdir(parents=True)
    posix_sp.mkdir(parents=True)
    _make_dist(win_sp, "cryptography", "49.0.0", with_record=False)
    _make_dist(posix_sp, "broken_pkg", "2.0", with_record=False)

    fake_main = type("M", (), {"PROJECT_ROOT": root})()
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

    purged = update_cmd._heal_stale_distinfo_without_record(root)

    assert "cryptography" in purged
    assert "broken_pkg" in purged
    assert not (win_sp / "cryptography-49.0.0.dist-info").exists()
    assert not (posix_sp / "broken_pkg-2.0.dist-info").exists()
