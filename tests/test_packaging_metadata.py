import ast
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

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


def test_telegram_mini_app_assets_ship_in_both_wheel_and_sdist():
    """The first-party Telegram Mini App must survive every install path.

    The Mini App is package data beneath the Telegram plugin. A source checkout
    can serve these assets even when packaging metadata is wrong, so verify the
    wheel and sdist declarations independently and require the runtime assets.
    """
    static = REPO_ROOT / "plugins/platforms/telegram/mini_app/static"
    required = {
        "index.html",
        "app.css",
        "app.js",
        "assets/collapse-bold.woff2",
        "assets/collapse-regular.woff2",
        "assets/courier-prime.woff2",
        "assets/hermes-logo.webp",
        "assets/mondwest-regular.woff2",
        "assets/rules-expanded-bold.woff2",
        "assets/rules-expanded-regular.woff2",
    }
    actual = {
        str(path.relative_to(static)) for path in static.rglob("*") if path.is_file()
    }
    assert required <= actual
    assert not any(
        "dashboard" in path.lower() or "mock" in path.lower() or path.startswith("qa/")
        for path in actual
    )

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    plugin_data = data["tool"]["setuptools"]["package-data"].get("plugins", [])
    assert "*/telegram/mini_app/static/*" in plugin_data
    assert "*/telegram/mini_app/static/**/*" in plugin_data

    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include plugins/platforms/telegram/mini_app/static *" in manifest


def test_built_distributions_include_telegram_mini_app_runtime(tmp_path):
    """Build and inspect both distribution formats, not metadata strings alone."""
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / filename, source / filename)
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    for module in metadata["tool"]["setuptools"]["py-modules"]:
        shutil.copy2(REPO_ROOT / f"{module}.py", source / f"{module}.py")
    for directory in (
        "agent",
        "tools",
        "hermes_cli",
        "gateway",
        "tui_gateway",
        "cron",
        "acp_adapter",
        "plugins",
        "providers",
        "locales",
        "optional-mcps",
    ):
        shutil.copytree(
            REPO_ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    dist = tmp_path / "dist"
    uv = shutil.which("uv")
    assert uv, "uv is required to prove wheel/sdist packaging contracts"
    build = subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--sdist",
            "--no-build-isolation",
            "--offline",
            "--cache-dir",
            str(tmp_path / "uv-cache"),
            "--out-dir",
            str(dist),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, f"uv build failed:\n{build.stderr}"

    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = set(archive.namelist())
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_files = {
            member.name for member in archive.getmembers() if member.isfile()
        }

    required_payload = {
        "plugins/platforms/telegram/mini_app/__init__.py",
        "plugins/platforms/telegram/mini_app/app.py",
        "plugins/platforms/telegram/mini_app/auth.py",
        "plugins/platforms/telegram/mini_app/cli.py",
        "plugins/platforms/telegram/mini_app/projection.py",
        "plugins/platforms/telegram/mini_app/run.py",
        "plugins/platforms/telegram/mini_app/service.py",
        "plugins/platforms/telegram/mini_app/static/index.html",
        "plugins/platforms/telegram/mini_app/static/app.css",
        "plugins/platforms/telegram/mini_app/static/app.js",
        "plugins/platforms/telegram/mini_app/static/assets/hermes-logo.webp",
    }
    assert required_payload <= wheel_files
    assert all(
        any(name.endswith(path) for name in sdist_files) for path in required_payload
    )
    assert not any(
        "mini_app/static" in name
        and ("dashboard" in name.lower() or "mock" in name.lower() or "/qa/" in name)
        for name in wheel_files | sdist_files
    )


def test_telegram_mini_app_frontend_remains_read_only():
    """Reject accidental reintroduction of the raw Dashboard capability set."""
    static = REPO_ROOT / "plugins/platforms/telegram/mini_app/static"
    app = (static / "app.js").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")

    forbidden = (
        "/dashboard",
        "/api/pty",
        "/api/hermes/update",
        "data-delete-session",
        "method:'DELETE'",
        "method:'PUT'",
        "method:'PATCH'",
    )
    for capability in forbidden:
        assert capability not in app
        assert capability not in index
    assert not (static / "dashboard-bridge.js").exists()

    # Authentication exchange is the browser's only POST. All data reads go
    # through a fixed frontend allowlist and same-origin credentials.
    assert app.count("method:'POST'") == 1
    assert "fetch('/api/auth/session'," in app
    assert "const READ_ONLY_ENDPOINTS = [" in app
    assert "Capacity" not in app
    assert "accounts." not in app
    assert "accountTimer" not in app
    assert "accountLimit" not in app
    assert "api('/api/live-usage')" not in app
    assert "provider" not in app.lower()
    assert "typeof x.enabled==='boolean'" in app
    assert "label:'Catalog'" in app
    assert ".filter(x=>x.enabled)" not in app
    allowed_reads = {
        "/api/me",
        "/api/status",
        "/api/live-usage?include_accounts=false",
        "/api/swarm/board",
        "/api/memory",
        "/api/sessions?limit=30",
        "/api/tools/toolsets",
        "/api/skills",
    }
    literal_reads = set(re.findall(r"\bapi\('([^']+)'", app))
    assert literal_reads
    assert literal_reads <= allowed_reads

    # A cache revision can be changed deliberately, but CSS and JS must always
    # use the same deterministic value so a deploy cannot mix revisions.
    css_revision = re.search(r'/static/app\.css\?v=([^"\']+)', index)
    js_revision = re.search(r'/static/app\.js\?v=([^"\']+)', index)
    assert css_revision and js_revision
    assert css_revision.group(1) == js_revision.group(1)

    # Only tiny display preferences persist. Legacy snapshots contained
    # memory/session/usage data and must be actively purged.
    assert "localStorage.removeItem('hermesMiniSnapshot')" in app
    preference_write = next(
        line
        for line in app.splitlines()
        if "localStorage.setItem('hermesMiniPreferences'" in line
    )
    sensitive_state = (
        "memory:state",
        "sessions:state",
        "preview:state",
        "result:state",
        "live:state",
        "status:state",
        "tools:state",
        "skills:state",
    )
    for state_name in sensitive_state:
        assert state_name not in preference_write


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


# ---------------------------------------------------------------------------
# Dependency-pin consistency: pyproject extras <-> tools/lazy_deps.py
#
# The same package is exact-pinned in two hand-maintained places: the
# [project.optional-dependencies] extras in pyproject.toml and the LAZY_DEPS
# allowlist in tools/lazy_deps.py (the lazy-install path deliberately mirrors
# the extras — see the comments on LAZY_DEPS: "match the corresponding extra
# in pyproject.toml ... update both this map AND the corresponding extra").
#
# They have silently drifted more than once: the aiohttp Slack pin (3.13.3 in
# the extras vs 3.13.4 in lazy_deps) and the anthropic pin (0.86.0 vs 0.87.0).
# The version a user ends up with then depends on whether the backend was
# installed eagerly (extra) or lazily (lazy_deps) — and for a CVE bump applied
# to only one side, that divergence is a latent security regression. These two
# tests assert the documented contract: the two sources agree, in lockstep.
# ---------------------------------------------------------------------------

# Matches "name==version" and "name[extra]==version", ignoring any trailing
# environment marker / comment. Only exact pins are collected; ranged specs
# (">=", "<") can't be compared for equality and are skipped.
_PIN_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;,#]+)"
)


def _canonical(name: str) -> str:
    # PEP 503 normalization so e.g. discord.py / discord-py compare equal.
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins_from_specs(specs):
    """Map canonical package name -> set of exact-pinned versions seen."""
    pins: dict[str, set[str]] = {}
    for spec in specs:
        m = _PIN_RE.match(spec)
        if not m:
            continue
        pins.setdefault(_canonical(m.group(1)), set()).add(m.group(2))
    return pins


def _pyproject_pinned_specs():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    return specs


def _lazy_deps_pinned_specs():
    """Extract every string literal inside the LAZY_DEPS dict via AST.

    Parsing rather than importing keeps this test free of
    tools/lazy_deps.py's runtime imports and side effects.
    """
    src = (REPO_ROOT / "tools" / "lazy_deps.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    specs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "LAZY_DEPS" for t in targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                specs.append(sub.value)
    assert specs, "could not extract specs from LAZY_DEPS — the AST parser drifted"
    return specs


def test_pyproject_pins_are_internally_consistent():
    """No package may be exact-pinned to two different versions in pyproject.

    A package legitimately appearing in several extras (e.g. aiohttp in
    messaging/slack/homeassistant/sms) must use the SAME version everywhere.
    """
    pins = _pins_from_specs(_pyproject_pinned_specs())
    conflicts = {name: sorted(v) for name, v in pins.items() if len(v) > 1}
    assert not conflicts, (
        "pyproject.toml exact-pins the same package to different versions "
        "across [project.dependencies] / extras: " + str(conflicts)
    )


def test_pyproject_and_lazy_deps_pins_agree():
    """Every package pinned in BOTH places must use the same version.

    Regression guard for the aiohttp / anthropic extras-vs-lazy drift:
    tools/lazy_deps.py mirrors the pyproject extras, so a CVE bump applied to
    one and not the other leaves users on a vulnerable version depending on
    the install path. Bump both in lockstep.
    """
    py = _pins_from_specs(_pyproject_pinned_specs())
    lazy = _pins_from_specs(_lazy_deps_pinned_specs())

    mismatches = [
        f"{name}: pyproject={sorted(py[name])} lazy_deps={sorted(lazy[name])}"
        for name in sorted(set(py) & set(lazy))
        if py[name] != lazy[name]
    ]
    assert not mismatches, (
        "pyproject.toml extras and tools/lazy_deps.py disagree on the pinned "
        "version of the same package — bump both in lockstep:\n  "
        + "\n  ".join(mismatches)
    )


def _lazy_deps_by_feature():
    """Parse LAZY_DEPS into {feature_name: [spec, ...]} via AST.

    Same parse-don't-import rationale as _lazy_deps_pinned_specs, but keeps the
    feature -> specs grouping so per-feature coverage can be asserted.
    """
    src = (REPO_ROOT / "tools" / "lazy_deps.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        if not any(isinstance(t, ast.Name) and t.id == "LAZY_DEPS" for t in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        by_feature: dict[str, list[str]] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            by_feature[key.value] = [
                sub.value
                for sub in ast.walk(value)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            ]
        assert by_feature, "could not extract features from LAZY_DEPS — AST parser drifted"
        return by_feature
    raise AssertionError("LAZY_DEPS dict literal not found in tools/lazy_deps.py")


# Security-critical packages whose patched floor must be enforced on EVERY
# install path, eager and lazy. test_pyproject_and_lazy_deps_pins_agree only
# fires when a package is pinned in BOTH sources, so it cannot catch a lazy
# feature that omits the pin entirely — the exact gap that left platform.slack
# carrying aiohttp==3.14.0 while platform.discord (whose discord.py dep pulls
# aiohttp transitively as its HTTP backbone) shipped without it, so the lazy
# Discord path could keep an already-installed vulnerable aiohttp. A fully
# general "no mirrored feature drops a pin" check is impossible statically
# (it can't see transitive deps), so this is the explicit coverage contract:
# each security package -> the lazy features that bundle an SDK pulling it and
# must therefore carry the same pin as the pyproject extra.
_REQUIRED_SECURITY_PINS = {
    # Every lazy messaging feature whose SDK pulls aiohttp transitively must
    # carry the patched floor directly: discord.py (aiohttp<4), slack-bolt,
    # mautrix/aiohttp-socks (aiohttp<4 / >=3.10), and microsoft-teams-apps —
    # none of those upper/lower bounds excludes a vulnerable already-installed
    # aiohttp, so the lazy path would not upgrade it without an explicit pin.
    "aiohttp": {
        "platform.discord",
        "platform.slack",
        "platform.matrix",
        "platform.teams",
    },
}


def test_security_pins_present_in_mirrored_lazy_features():
    """Curated security pins must be present (not just version-consistent) in
    every lazy feature that bundles an SDK pulling that package transitively.
    """
    py = _pins_from_specs(_pyproject_pinned_specs())
    by_feature = _lazy_deps_by_feature()

    problems = []
    for pkg, features in _REQUIRED_SECURITY_PINS.items():
        canon = _canonical(pkg)
        expected = py.get(canon)
        assert expected, (
            f"{pkg} is listed in _REQUIRED_SECURITY_PINS but is not exact-pinned "
            f"in pyproject.toml — update the map or the pin."
        )
        for feature in sorted(features):
            specs = by_feature.get(feature)
            assert specs is not None, (
                f"lazy feature {feature!r} named in _REQUIRED_SECURITY_PINS no "
                f"longer exists in LAZY_DEPS — update the map."
            )
            got = _pins_from_specs(specs).get(canon)
            if got != expected:
                problems.append(
                    f"{feature}: {pkg}="
                    f"{sorted(got) if got else 'MISSING'}, expected {sorted(expected)}"
                )
    assert not problems, (
        "a lazy feature is missing a security pin it must mirror from the "
        "pyproject extras — the lazy install path would not enforce the "
        "CVE-patched floor:\n  " + "\n  ".join(problems)
    )
