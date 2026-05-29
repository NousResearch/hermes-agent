from pathlib import Path
import tomllib

from setuptools import find_packages


REPO_ROOT = Path(__file__).resolve().parents[1]


def _packages_find_include():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["setuptools"]["packages"]["find"]["include"]


def test_every_on_disk_subpackage_is_covered_by_packages_find():
    """Regression test for #34701 (and the bug class behind #34034 / #28149).

    ``[tool.setuptools.packages.find]`` ``include`` is hand-maintained. Every
    top-level package is listed twice — bare (``hermes_cli``) for the package
    itself and ``hermes_cli.*`` for its subpackages — EXCEPT when someone
    forgets the wildcard. v0.15.x listed ``hermes_cli`` without ``hermes_cli.*``,
    so the wheel shipped ``hermes_cli/*.py`` but dropped the ``dashboard_auth``
    and ``proxy`` subpackages. The dashboard then died on every install with
    ``ModuleNotFoundError: No module named 'hermes_cli.dashboard_auth'``.

    This drives setuptools' own discovery against the live tree: every package
    that exists on disk and would be found by a permissive ``<name>.*`` scan
    must also be found by the actual ``include`` list. A subpackage added under
    any listed package without the matching wildcard fails here instead of in a
    user's container.
    """
    include = _packages_find_include()

    # What the real include list actually selects.
    selected = set(find_packages(where=str(REPO_ROOT), include=include))

    # Top-level packages we ship (bare names in the include list, no wildcard).
    top_level = sorted({name for name in include if "." not in name})

    # For each shipped top-level package, every on-disk subpackage must be
    # covered by the include list.
    expected = set(
        find_packages(
            where=str(REPO_ROOT),
            include=[pattern for name in top_level for pattern in (name, f"{name}.*")],
        )
    )

    missing = sorted(expected - selected)
    assert not missing, (
        "These packages exist on disk but are dropped from the wheel because "
        "[tool.setuptools.packages.find] include is missing a wildcard. Add the "
        f"matching '<name>.*' entry in pyproject.toml: {missing}"
    )


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_bundled_plugin_manifests_ship_in_both_wheel_and_sdist():
    """Regression test for #34034 / #28149.

    Plugin discovery (hermes_cli/plugins.py) registers each bundled plugin by
    reading its ``plugin.yaml`` / ``plugin.yml`` manifest. Those manifests are
    data files, not Python modules, so they only reach installed packages when
    declared explicitly:

    - wheel  -> ``[tool.setuptools.package-data]`` ``plugins`` glob
    - sdist  -> ``MANIFEST.in`` (Homebrew and other downstream packagers build
                from the sdist)

    v0.15.0 declared neither, so the wheel shipped every adapter's Python code
    but none of its manifests, and *every* gateway platform failed with
    "No adapter available for <platform>". Both channels must cover manifests.
    """
    # There must actually be manifests on disk for the globs to match.
    on_disk = list((REPO_ROOT / "plugins").rglob("plugin.yaml")) + list(
        (REPO_ROOT / "plugins").rglob("plugin.yml")
    )
    assert on_disk, "expected bundled plugin manifests under plugins/"

    # Wheel channel: package-data must declare a glob that matches plugin
    # manifests anywhere under the plugins package.
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    plugins_pkg_data = data["tool"]["setuptools"]["package-data"].get("plugins", [])
    assert any(
        g.endswith("plugin.yaml") or g.endswith("plugin.yml")
        for g in plugins_pkg_data
    ), "pyproject package-data 'plugins' must ship plugin.yaml/plugin.yml (wheel)"

    # Sdist channel: MANIFEST.in must recursively include the manifests so
    # downstream packagers building from the sdist also get them.
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include plugins" in manifest and "plugin.yaml" in manifest, (
        "MANIFEST.in must recursive-include plugins plugin.yaml/plugin.yml (sdist)"
    )
