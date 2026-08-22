"""Doctor must notice half-installed distributions.

`hermes doctor`'s package section import-probes a handful of hardcoded
module names. That catches "dependency absent", but not the failure mode
where a distribution's files are on disk and its ``.dist-info`` metadata
is not: pip can then neither use nor uninstall it, so the environment
cannot repair itself and every subsequent install of that package fails.

That is not hypothetical. A Windows install was left with `cryptography`
in exactly that state after an interrupted update -- Hermes would not
start, pip could not fix it, and doctor reported "All checks passed",
because `cryptography` is not one of the names it import-probes.

Adding more names to the hardcoded list only moves the goalpost, so the
check verifies the metadata of whatever is actually installed.
"""
from __future__ import annotations

import os
from pathlib import Path

from hermes_cli.doctor import _UNNAMED_DIST, _scan_installed_distributions


def _make_dist(site: Path, name: str, version: str = "1.0.0") -> Path:
    """Create a minimal, healthy wheel-style installed distribution."""
    dist_info = site / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(f"{name}/__init__.py,,\n", encoding="utf-8")
    pkg = site / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return dist_info


def test_healthy_distributions_are_not_flagged(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    _make_dist(site, "alpha")
    _make_dist(site, "beta")

    broken, total = _scan_installed_distributions([str(site)])

    assert total == 2
    assert broken == []


def test_missing_record_is_flagged(tmp_path: Path) -> None:
    """The exact shape of the real incident: package dir kept, RECORD gone."""
    site = tmp_path / "site-packages"
    site.mkdir()
    dist_info = _make_dist(site, "cryptography", "50.0.0")
    (dist_info / "RECORD").unlink()

    broken, total = _scan_installed_distributions([str(site)])

    assert total == 1
    assert len(broken) == 1
    name, reason = broken[0]
    assert name == "cryptography"
    assert "RECORD" in reason
    # The package itself is still importable-looking on disk -- which is
    # precisely why an import probe would have missed this.
    assert (site / "cryptography" / "__init__.py").exists()


def test_metadata_without_name_is_flagged(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    dist_info = _make_dist(site, "gamma")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nVersion: 1.0.0\n", encoding="utf-8"
    )

    broken, total = _scan_installed_distributions([str(site)])

    assert total == 1
    assert len(broken) == 1
    assert "Name" in broken[0][1]


def test_unnamed_distribution_never_reports_a_path_as_its_name(tmp_path: Path) -> None:
    """The name is interpolated into a pip command -- it must stay a name.

    A path in the name slot yields `pip --force-reinstall C:\\...\\foo.dist-info`,
    which cannot be run. The path belongs in the reason instead.
    """
    site = tmp_path / "site-packages"
    site.mkdir()
    dist_info = _make_dist(site, "delta")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nVersion: 1.0.0\n", encoding="utf-8"
    )

    broken, _ = _scan_installed_distributions([str(site)])

    name, reason = broken[0]
    assert name == _UNNAMED_DIST
    assert os.sep not in name and "/" not in name, f"path leaked into name: {name!r}"
    assert "delta" in reason, f"the path should be surfaced in the reason: {reason!r}"


def test_egg_info_without_record_is_not_flagged(tmp_path: Path) -> None:
    """Legacy/editable installs legitimately have no RECORD -- no false alarm."""
    site = tmp_path / "site-packages"
    site.mkdir()
    egg = site / "legacy-1.0.0.egg-info"
    egg.mkdir(parents=True)
    (egg / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: legacy\nVersion: 1.0.0\n", encoding="utf-8"
    )

    broken, total = _scan_installed_distributions([str(site)])

    assert total == 1, "the .egg-info distribution should still be discovered"
    assert broken == [], f"legacy install wrongly flagged: {broken}"


def test_scan_never_raises_on_a_broken_path(tmp_path: Path) -> None:
    """A diagnostics helper must not itself become the failure."""
    missing = tmp_path / "does-not-exist"
    broken, total = _scan_installed_distributions([str(missing)])
    assert broken == []
    assert total == 0


def test_returns_the_documented_shape(tmp_path: Path) -> None:
    """Contract check, kept hermetic.

    Deliberately not run against the live interpreter: that would make the
    suite depend on whatever happens to be installed on the runner, and it
    would fail on exactly the corrupted environment this check exists to
    diagnose.
    """
    site = tmp_path / "site-packages"
    site.mkdir()
    _make_dist(site, "epsilon")

    result = _scan_installed_distributions([str(site)])

    assert isinstance(result, tuple) and len(result) == 2
    broken, total = result
    assert isinstance(broken, list)
    assert isinstance(total, int)
    assert all(
        isinstance(item, tuple) and len(item) == 2 and all(isinstance(s, str) for s in item)
        for item in broken
    )
