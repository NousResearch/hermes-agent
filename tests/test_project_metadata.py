"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import sys
import tomllib

import pytest
from packaging.markers import Marker, default_environment

def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def _load_package_data():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        tool = tomllib.load(handle)["tool"]
    return tool["setuptools"]["package-data"]


def _load_uv_lock():
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    with lock_path.open("rb") as handle:
        return tomllib.load(handle)


def _locked_package(name):
    packages = [package for package in _load_uv_lock()["package"] if package["name"] == name]
    assert len(packages) == 1, f"expected one {name!r} package in uv.lock, got {packages}"
    return packages[0]


def _wake_marker_environment(version, *, sys_platform="linux", platform_machine="x86_64"):
    environment = default_environment()
    environment.update(
        {
            "python_version": f"{version[0]}.{version[1]}",
            "python_full_version": f"{version[0]}.{version[1]}.0",
            "extra": "wake",
            "sys_platform": sys_platform,
            "platform_system": "Darwin" if sys_platform == "darwin" else "Linux",
            "platform_machine": platform_machine,
        }
    )
    return environment


def test_matrix_extra_not_in_all():
    """The [matrix] extra pulls `mautrix[encryption]` -> `python-olm`,
    which has Linux-only wheels and no native build path on Windows or
    modern macOS (archived libolm, C++ errors with Clang 21+).

    With matrix in [all], `uv sync --locked` on Windows tried to build
    python-olm from sdist and failed on `make`. As of 2026-05-12 the
    [matrix] extra is excluded from [all] entirely and routed through
    `tools/lazy_deps.py` (LAZY_DEPS["platform.matrix"]) — installs at
    first use, where the user is expected to have a toolchain.
    """
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies, "[matrix] extra must still exist for `uv sync --extra matrix`"
    # Must NOT appear in [all] in any form — neither unconditional nor
    # platform-gated. Lazy-install handles it.
    matrix_in_all = [
        dep for dep in optional_dependencies["all"]
        if "matrix" in dep
    ]
    assert not matrix_in_all, (
        "matrix must not appear in [all] — it's lazy-installed via "
        "tools/lazy_deps.py LAZY_DEPS['platform.matrix']. Found: "
        f"{matrix_in_all}"
    )


def test_wake_openwakeword_has_explicit_python_platform_compatibility_boundary():
    """The Linux tflite path is limited to CPython 3.11, not non-Linux."""
    optional_dependencies = _load_optional_dependencies()
    wake = optional_dependencies["wake"]

    openwakeword_spec = next(spec for spec in wake if spec.startswith("openwakeword==0.6.0"))
    marker = Marker(openwakeword_spec.split(";", 1)[1].strip())
    assert marker.evaluate(_wake_marker_environment((3, 11), sys_platform="linux")) is True
    assert marker.evaluate(_wake_marker_environment((3, 12), sys_platform="linux")) is False
    assert marker.evaluate(
        _wake_marker_environment((3, 13), sys_platform="darwin", platform_machine="arm64")
    ) is True
    assert marker.evaluate(
        _wake_marker_environment((3, 12), sys_platform="win32", platform_machine="AMD64")
    ) is True


def test_wake_lock_markers_preserve_linux_boundary_and_darwin_bridge():
    """The lock follows the platform-aware project marker."""
    optional_dependencies = _load_optional_dependencies()
    project_spec = next(
        spec for spec in optional_dependencies["wake"] if spec.startswith("openwakeword==")
    )
    project_marker = Marker(project_spec.split(";", 1)[1].strip())

    locked_root = _locked_package("hermes-agent")
    locked_spec = next(
        requirement
        for requirement in locked_root["metadata"]["requires-dist"]
        if requirement["name"] == "openwakeword"
    )
    locked_marker = Marker(locked_spec["marker"])

    environments = {
        "linux-cp311": _wake_marker_environment((3, 11), sys_platform="linux"),
        "linux-cp312": _wake_marker_environment((3, 12), sys_platform="linux"),
        "linux-cp314": _wake_marker_environment((3, 14), sys_platform="linux"),
        "darwin-arm64-cp312": _wake_marker_environment(
            (3, 12), sys_platform="darwin", platform_machine="arm64"
        ),
        "darwin-arm64-cp313": _wake_marker_environment(
            (3, 13), sys_platform="darwin", platform_machine="arm64"
        ),
        "windows-amd64-cp312": _wake_marker_environment(
            (3, 12), sys_platform="win32", platform_machine="AMD64"
        ),
    }
    expected = {
        "linux-cp311": True,
        "linux-cp312": False,
        "linux-cp314": False,
        "darwin-arm64-cp312": True,
        "darwin-arm64-cp313": True,
        "windows-amd64-cp312": True,
    }
    for name, environment in environments.items():
        assert project_marker.evaluate(environment) is expected[name], name
        assert locked_marker.evaluate(environment) is expected[name], name

    assert locked_spec["specifier"] == "==0.6.0"
    assert "extra == 'wake'" in locked_spec["marker"]

    openwakeword = _locked_package("openwakeword")
    tflite_dependency = next(
        dependency
        for dependency in openwakeword["dependencies"]
        if dependency["name"] == "tflite-runtime"
    )
    assert tflite_dependency["marker"] == "sys_platform == 'linux'"

    ai_edge_spec = next(
        requirement
        for requirement in locked_root["metadata"]["requires-dist"]
        if requirement["name"] == "ai-edge-litert" and requirement["marker"]
    )
    assert ai_edge_spec["marker"] == "sys_platform == 'darwin' and extra == 'wake'"
    assert ai_edge_spec["specifier"] == "==2.1.6"


def test_tflite_lock_contains_supported_cp311_linux_architectures():
    """The lock retains all three published CPython 3.11 Linux wheels."""
    wheels = _locked_package("tflite-runtime")["wheels"]
    wheel_urls = [wheel["url"] for wheel in wheels]
    for architecture in ("x86_64", "aarch64", "armv7l"):
        assert any("cp311" in url and architecture in url for url in wheel_urls), architecture
    assert not any("cp312" in url or "cp313" in url or "cp314" in url for url in wheel_urls)


@pytest.mark.skipif(
    sys.platform != "linux"
    or sys.implementation.name != "cpython"
    or sys.version_info[:2] != (3, 11),
    reason="wake import smoke runs on CPython 3.11 Linux only",
)
def test_supported_wake_runtime_import_smoke():
    """Import the locked tflite/openWakeWord path when the wake extra is present.

    Lean test environments intentionally do not install the optional ``wake``
    extra. On a CPython 3.11 Linux job that does install it, both imports must
    succeed; otherwise this test reports an explicit, actionable skip.
    """
    pytest.importorskip(
        "tflite_runtime.interpreter",
        reason="install the optional wake extra to run the CPython 3.11 smoke",
    )
    pytest.importorskip(
        "openwakeword.model",
        reason="install the optional wake extra to run the CPython 3.11 smoke",
    )


def test_lazy_installable_extras_excluded_from_all():
    """Policy (2026-05-12): every extra that has a `LAZY_DEPS` entry
    in `tools/lazy_deps.py` must be excluded from [all].

    The lazy-install system exists so one quarantined PyPI release
    (e.g. mistralai 2.4.6) can't break every fresh install. Putting a
    backend in BOTH [all] and LAZY_DEPS defeats that — fresh installs
    eager-install it and inherit whatever's broken upstream.

    If you're tempted to add an opt-in backend to [all] for "convenience,"
    add it to `LAZY_DEPS` instead so it installs at first use.
    """
    optional_dependencies = _load_optional_dependencies()

    # Hard-coded mirror of the extras that are in LAZY_DEPS as of
    # 2026-05-12. This list intentionally duplicates rather than
    # imports tools/lazy_deps.py so the test stays a contract — if
    # someone adds a new lazy-install backend, they have to update
    # this list AND verify [all] doesn't contain it.
    lazy_covered_extras = {
        "anthropic", "bedrock",
        "exa", "firecrawl", "parallel-web",
        "fal",
        "edge-tts", "tts-premium",
        "voice",  # faster-whisper / sounddevice / numpy
        "modal", "daytona", "vercel",
        "messaging", "slack", "matrix", "dingtalk", "feishu",
        "honcho", "hindsight",
        "supermemory", "mem0",
        "mistral",  # mistralai — Voxtral STT/TTS, lazy-installed (stt.mistral / tts.mistral)
    }
    all_extra_specs = optional_dependencies["all"]
    for extra in lazy_covered_extras:
        offending = [
            spec for spec in all_extra_specs
            if f"hermes-agent[{extra}]" in spec
        ]
        assert not offending, (
            f"[{extra}] is in [all] but also in LAZY_DEPS. "
            f"Remove it from [all] in pyproject.toml — it lazy-installs "
            f"at first use. Found in [all]: {offending}"
        )


def _exact_pins(specs):
    pins = {}
    for spec in specs:
        requirement = spec.split(";", 1)[0].strip()
        if "==" not in requirement:
            continue
        package, version = requirement.split("==", 1)
        package = package.split("[", 1)[0].lower().replace("_", "-")
        pins[package] = version
    return pins




def test_pyproject_pins_match_lazy_deps_pins():
    """Generalize #31817 to the whole pin surface, not just aiohttp.

    Any package that is exact-pinned in BOTH a pyproject extra and a
    `tools/lazy_deps.py` LAZY_DEPS entry must use the SAME version in both
    places. When they drift, `hermes update` resolves the pyproject extra
    pin and downgrades the package to the older version, reopening whatever
    the lazy pin fixed (the aiohttp #31817 case, and the anthropic
    CVE-2026-34450/34452 case found alongside it) — only for the lazy
    refresh to re-upgrade it on next feature use. The lazy pin is the
    security-current source of truth; extras must track it.
    """
    from tools.lazy_deps import LAZY_DEPS

    optional_dependencies = _load_optional_dependencies()

    # package -> version, as pinned across all pyproject extras. If an
    # extra pins a package at a different version than another extra, that
    # is itself a bug (caught below); here we just collect the set.
    pyproject_pins: dict[str, set[str]] = {}
    for specs in optional_dependencies.values():
        for package, version in _exact_pins(specs).items():
            pyproject_pins.setdefault(package, set()).add(version)

    # package -> version, as pinned across all LAZY_DEPS entries.
    lazy_pins: dict[str, set[str]] = {}
    for specs in LAZY_DEPS.values():
        if isinstance(specs, str):
            specs = (specs,)
        for package, version in _exact_pins(specs).items():
            lazy_pins.setdefault(package, set()).add(version)

    shared = sorted(set(pyproject_pins) & set(lazy_pins))
    assert shared, "expected at least one package pinned in both pyproject and LAZY_DEPS"

    drift = {
        package: {
            "pyproject": sorted(pyproject_pins[package]),
            "lazy_deps": sorted(lazy_pins[package]),
        }
        for package in shared
        if pyproject_pins[package] != lazy_pins[package]
    }
    assert not drift, (
        "pyproject extras pins must match tools/lazy_deps.py LAZY_DEPS pins "
        "for every shared package — otherwise `hermes update` downgrades the "
        "package below the security-current lazy pin (see #31817). Drift: "
        f"{drift}"
    )






def test_dingtalk_extra_includes_qrcode_for_qr_auth():
    """DingTalk's QR-code device-flow auth (hermes_cli/dingtalk_auth.py)
    needs the qrcode package."""
    optional_dependencies = _load_optional_dependencies()

    dingtalk_extra = optional_dependencies["dingtalk"]
    assert any(dep.startswith("qrcode") for dep in dingtalk_extra)






def _uv_lock_version(package: str) -> str:
    """Resolved version of ``package`` in uv.lock, or fail loudly."""
    versions = _uv_lock_versions(package)
    assert versions, f"{package} not found in uv.lock"
    assert len(versions) == 1, f"{package} resolves to multiple versions in uv.lock: {versions}"
    return next(iter(versions))


def _uv_lock_versions(package: str) -> set[str]:
    """All resolved versions of ``package`` in uv.lock (normally 0 or 1)."""
    import re

    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    lock = lock_path.read_text(encoding="utf-8")
    return {
        m.group(1)
        for m in re.finditer(
            rf'\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"',
            lock,
        )
    }


def test_every_lazy_deps_exact_pin_matches_uv_lock():
    """Class invariant for #60783/#60685: one version per package, everywhere.

    Any package that is BOTH exact-pinned in ``tools/lazy_deps.py`` AND
    resolved in the committed uv.lock is a *shared* package: the core
    install ships the locked version, and the ``hermes update`` lazy-refresh
    pass re-asserts the LAZY_DEPS pin whenever the package is present
    (``active_features()``). If the two disagree, every update churns the
    package — and when the lazy pin is older, it force-DOWNGRADES a version
    another consumer needs (huggingface-hub==1.2.3 vs transformers'
    >=1.5.0 broke Hindsight local embeddings; stale aiohttp pins reopened
    patched CVEs in #31817). Contract: for every such package, pin ==
    locked version. When bumping a pin, regenerate the lock in the same
    commit (`uv lock --upgrade-package <name>`), and vice versa.
    """
    from tools.lazy_deps import LAZY_DEPS

    drift = {}
    seen = set()
    for feature, specs in LAZY_DEPS.items():
        for package, pin in _exact_pins(specs).items():
            if (package, pin) in seen:
                continue
            seen.add((package, pin))
            locked = _uv_lock_versions(package)
            if not locked:
                # Lazy-only package never resolved by the core lock — no
                # shared-version hazard.
                continue
            if pin not in locked:
                drift.setdefault(package, {})[feature] = {
                    "lazy_pin": pin,
                    "uv_lock": sorted(locked),
                }

    assert not drift, (
        "LAZY_DEPS exact pins must match the uv.lock resolved version for "
        "every package the core lock also ships — otherwise `hermes update` "
        "churns/downgrades the shared package out from under its other "
        "consumers (#60783, #31817). Bump the pin AND run "
        "`uv lock --upgrade-package <name>` in the same commit. Drift: "
        f"{drift}"
    )


def test_huggingface_hub_lazy_pin_matches_uv_lock():
    """The whole tree must converge on ONE huggingface-hub version (#60783).

    huggingface-hub is a shared dependency: the core lock resolves it (via
    faster-whisper/tokenizers, and transformers/sentence-transformers when
    local Hindsight embeddings are installed), and LAZY_DEPS
    ['tool.trace_upload'] exact-pins it. Because active_features() activates
    a feature from mere package presence, the `hermes update` lazy-refresh
    pass re-asserts the LAZY_DEPS pin on every install where hub is present.
    If that pin drifts from the lock's resolved version, every update churns
    the shared package — and a pin below transformers' floor (>=1.5.0)
    force-downgrades it and breaks the Hindsight local daemon on startup.
    """
    from tools.lazy_deps import LAZY_DEPS

    lazy_pin = _exact_pins(LAZY_DEPS["tool.trace_upload"]).get("huggingface-hub")
    assert lazy_pin, "tool.trace_upload must exact-pin huggingface-hub"

    locked = _uv_lock_version("huggingface-hub")
    assert lazy_pin == locked, (
        "LAZY_DEPS['tool.trace_upload'] pins huggingface-hub=="
        f"{lazy_pin} but uv.lock resolves {locked}. These must move in "
        "lockstep (bump the pin AND run `uv lock --upgrade-package "
        "huggingface-hub`), or `hermes update` will churn/downgrade the "
        "shared package and break Hindsight local embeddings (#60783)."
    )


def test_huggingface_hub_lazy_pin_inside_transformers_window():
    """The hub pin must stay in transformers' accepted range (#60783).

    transformers (pulled by sentence-transformers for Hindsight
    local/local_embedded embeddings) requires huggingface-hub>=1.5.0,<2.
    An exact pin outside that window makes the lazy-refresh downgrade the
    shared package below what the embedding stack imports, and the
    Hindsight daemon fails on startup. Contract, not a snapshot: any
    future exact pin is fine as long as it stays inside the window.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from tools.lazy_deps import LAZY_DEPS

    pin = _exact_pins(LAZY_DEPS["tool.trace_upload"]).get("huggingface-hub")
    assert pin, "tool.trace_upload must exact-pin huggingface-hub"
    transformers_window = SpecifierSet(">=1.5.0,<2")
    assert Version(pin) in transformers_window, (
        f"huggingface-hub=={pin} falls outside transformers' accepted "
        "range (>=1.5.0,<2). The lazy refresh would downgrade the shared "
        "package and break Hindsight local embeddings (#60783)."
    )
