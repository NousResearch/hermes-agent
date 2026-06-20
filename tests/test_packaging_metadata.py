from pathlib import Path
import re
import tomllib

import pytest

# setuptools is declared in the [dev] extra and is the build backend, but
# guard the import so a runner without it skips these packaging checks
# instead of erroring out collection for the whole shard (it used to be
# picked up ambiently from the CI image; newer ubuntu-latest images don't
# ship it in the test venv).
find_packages = pytest.importorskip("setuptools", exc_type=ImportError).find_packages


REPO_ROOT = Path(__file__).resolve().parents[1]


def _distribution_name(requirement: str) -> str:
    """Extract the PEP 508 distribution name from a requirement string.

    Robust to markers (``; python_version < '3.12'``), direct references
    (``name @ https://...``), extras (``name[extra]``) and every version
    operator (``==``, ``>=``, ``<=``, ``~=``, ``!=``, ``<``, ``>``), so a
    future dep declared with any valid specifier shape doesn't silently
    mis-parse here.
    """
    spec = requirement.split(";", 1)[0]  # drop environment markers
    spec = spec.split("@", 1)[0]  # drop direct-reference URLs
    spec = spec.split("[", 1)[0]  # drop extras
    spec = re.split(r"[=<>!~]", spec, maxsplit=1)[0]  # drop any version operator
    return spec.strip().lower()


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


def test_packaging_declared_as_core_dependency():
    """Regression for #40503.

    ``packaging`` is imported directly on three production paths
    (plugins/memory/hindsight/__init__.py, tools/lazy_deps.py,
    hermes_cli/main.py) yet was undeclared, so it only reached users
    transitively. The slim Docker image shipped without it, silently
    disabling Hindsight append-mode and version-constraint checks. It must
    be a declared core dependency so it installs everywhere and the
    update-repair step (``_verify_core_dependencies_installed``) guards it.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = data["project"]["dependencies"]
    names = {_distribution_name(dep) for dep in core}
    assert "packaging" in names, (
        "packaging is imported on production paths (hindsight version compare, "
        "lazy_deps version constraints, requirement parsing) and must be a "
        "declared core dependency, not a transitive — see #40503"
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


# Minimum non-vulnerable Starlette: CVE-2026-48710 ("BadHost") was fixed in
# 1.0.1. Anything below that lets a malformed Host header desync
# ``request.url.path`` from the dispatched ASGI path, bypassing path-based
# authz in middleware/endpoints that gate on ``request.url``. Starlette is a
# transitive dep (fastapi in [web]; sse-starlette/mcp in [mcp]/[computer-use]/
# [dev]) so we pin it directly in every extra that exposes a server surface and
# enforce the floor in both pyproject and the committed lockfile.
_STARLETTE_CVE_FLOOR = (1, 0, 1)


def _version_tuple(spec: str) -> tuple[int, ...]:
    # "1.0.1" -> (1, 0, 1); tolerant of pre/post suffixes by truncating.
    head = spec.split("+", 1)[0]
    parts = []
    for chunk in head.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def test_starlette_pinned_above_cve_2026_48710_floor_in_pyproject():
    """Every extra that declares Starlette must pin a patched (>=1.0.1) version.

    Regression guard for #35067 / CVE-2026-48710. A future edit that drops the
    pin (re-exposing the unbounded transitive ``starlette>=0.27`` from mcp /
    ``>=0.40.0`` from fastapi) or pins a pre-1.0.1 version fails here instead of
    shipping a Host-header auth-bypass to dashboard / MCP-HTTP users.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    found = {}
    for extra, specs in extras.items():
        for spec in specs:
            name = spec.split("==", 1)[0].split(">", 1)[0].split("<", 1)[0].split("[", 1)[0].strip()
            if name.lower() == "starlette":
                assert "==" in spec, f"[{extra}] must exact-pin starlette, got {spec!r}"
                ver = spec.split("==", 1)[1].split(";", 1)[0].strip()
                found[extra] = ver

    # The four server-surface extras must each carry the direct pin.
    for extra in ("web", "mcp", "computer-use", "dev"):
        assert extra in found, (
            f"[{extra}] no longer pins starlette directly — CVE-2026-48710 "
            f"regression risk (mcp/fastapi pull it transitively with no upper bound)"
        )

    for extra, ver in found.items():
        assert _version_tuple(ver) >= _STARLETTE_CVE_FLOOR, (
            f"[{extra}] pins starlette=={ver}, below the CVE-2026-48710 fix "
            f"floor {'.'.join(map(str, _STARLETTE_CVE_FLOOR))}"
        )


def test_locked_starlette_is_not_vulnerable_to_cve_2026_48710():
    """The committed uv.lock must resolve starlette to a patched version.

    pyproject pins protect the declared extras, but the lockfile is what
    hash-verified installs (``uv sync --locked``) actually pull. Assert the
    resolved version is >= the CVE-2026-48710 fix floor so a stale-lock
    regression can't ship a vulnerable Starlette to users.
    """
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    versions = []
    in_starlette = False
    for line in lock.splitlines():
        if line.startswith("[[package]]"):
            in_starlette = False
        elif line.strip() == 'name = "starlette"':
            in_starlette = True
        elif in_starlette and line.startswith("version = "):
            versions.append(line.split("=", 1)[1].strip().strip('"'))
            in_starlette = False

    assert versions, "starlette not found in uv.lock"
    for ver in versions:
        assert _version_tuple(ver) >= _STARLETTE_CVE_FLOOR, (
            f"uv.lock resolves starlette=={ver}, below the CVE-2026-48710 fix "
            f"floor {'.'.join(map(str, _STARLETTE_CVE_FLOOR))} — regenerate the "
            f"lockfile after bumping the pin"
        )


def test_locale_catalogs_ship_in_both_wheel_and_sdist():
    """Regression test for #27632 / #35374 / #23943.

    locales/ is a bare data directory (no __init__.py), so it is invisible to
    packages.find and to package-data (which attaches to a package). It must be
    declared as setuptools data-files (wheel) AND grafted in MANIFEST.in
    (sdist). Without both, sealed installs drop the catalogs and gateway/CLI
    commands surface raw i18n keys like `gateway.reset.header_default`.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = data["tool"]["setuptools"].get("data-files", {})
    assert data_files.get("locales") == ["locales/*.yaml"], (
        "pyproject [tool.setuptools.data-files] must declare "
        'locales = ["locales/*.yaml"] so the wheel ships i18n catalogs'
    )

    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft locales" in manifest, (
        "MANIFEST.in must `graft locales` so the sdist ships i18n catalogs"
    )

    # Every on-disk catalog has the .yaml extension the globs above match.
    on_disk = list((REPO_ROOT / "locales").glob("*.yaml"))
    assert on_disk, "expected locales/*.yaml catalogs on disk"


def test_optional_mcps_catalog_entries_ship_in_wheel():
    """Every on-disk optional-mcps/<name> must ship in the wheel + sdist.

    optional-mcps/ is a bare data directory (no __init__.py), so each catalog
    entry is shipped ONLY via a hand-maintained [tool.setuptools.data-files]
    target (wheel) plus `graft optional-mcps` in MANIFEST.in (sdist). A new
    catalog manifest that lands without its data-files target ships in the
    sdist but is dropped from the wheel, so `hermes mcp catalog`/`install`
    silently omit it on pip/Docker/Nix installs. This guards that recurring
    gap (e.g. the unreal-engine entry added in 5ffbfed19 without a target).
    """
    on_disk = sorted(p.parent.name for p in (REPO_ROOT / "optional-mcps").glob("*/manifest.yaml"))
    assert on_disk, "expected optional-mcps/*/manifest.yaml entries on disk"

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = data["tool"]["setuptools"].get("data-files", {})

    missing = []
    for name in on_disk:
        key = f"optional-mcps/{name}"
        expected = [f"optional-mcps/{name}/manifest.yaml"]
        if data_files.get(key) != expected:
            missing.append(key)
    assert not missing, (
        "pyproject [tool.setuptools.data-files] is missing a wheel target for "
        f"each on-disk optional-mcps entry (so it would be dropped from the "
        f"wheel): {missing}"
    )

    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft optional-mcps" in manifest, (
        "MANIFEST.in must `graft optional-mcps` so the sdist ships the catalog"
    )

