"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib

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


def test_lazy_installable_extras_excluded_from_all():
    """Policy (2026-05-12): every lazy-installable extra must stay out of [all].

    The lazy-install system exists so one quarantined PyPI release (e.g.
    mistralai 2.4.6) can't break every fresh install. Putting a backend in BOTH
    [all] and the lazy map defeats that — fresh installs eager-install it and
    inherit whatever's broken upstream.

    If you're tempted to add an opt-in backend to [all] for "convenience," map
    it in ``LAZY_DEPS`` instead so it installs at first use.

    This test READS the set of lazy extras from ``LAZY_DEPS``. Do not
    write that set out again here. A list written by hand needs its own
    update each time the map changes, so a new backend can enter both the
    map and [all] and this test still passes. ``[acp]`` and ``[google]``
    each reached that state.

    A small set of extras is deliberately in both, per the [all] policy comment
    in pyproject.toml — "things needed before the agent loop is alive" and
    "skill deps that dev environments need". Those are enumerated here with a
    reason each; anything else overlapping is a bug.
    """
    from tools.lazy_deps import LAZY_DEPS

    optional_dependencies = _load_optional_dependencies()

    # extra -> why it is intentionally BOTH eager (in [all]) and lazy-mapped.
    intentional_overlap = {
        # `uv sync --extra google` must give dev environments and packagers the
        # Workspace SDKs without a runtime pip path (which fails on Nix-managed
        # Pythons). The lazy entry covers lean installs that skipped [all].
        "google": "dev/packager convenience — see the [google] extra comment",
        # The dashboard is reachable via `hermes dashboard` before any agent
        # loop exists to lazy-install it.
        "web": "needed before the agent loop is alive",
        # Skill dep that dev environments are expected to have preinstalled.
        "youtube": "skill dep dev environments need",
    }

    lazy_covered_extras = set(LAZY_DEPS.values()) - set(intentional_overlap)
    all_extra_specs = optional_dependencies["all"]
    for extra in sorted(lazy_covered_extras):
        offending = [
            spec for spec in all_extra_specs
            if f"hermes-agent[{extra}]" in spec
        ]
        assert not offending, (
            f"[{extra}] is in [all] but is also lazy-installable. "
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
    """Lazy installs must resolve the same pins pyproject declares.

    Guard for #31817. That issue compared two lists of pins written by hand:
    ``tools/lazy_deps.py`` kept its own copy of every spec, and drift between
    the copies meant ``hermes update`` could downgrade a package below the
    security-current lazy pin. ``lazy_deps`` now READS the pyproject extras
    instead of duplicating them, so that specific drift is structurally
    impossible.

    What remains testable — and what actually protects the invariant — is that
    the read-through produces exactly the extras' pins. A bug in the extra
    resolver (bad composition, a dropped marker, a stale cache) would silently
    reintroduce the same class of failure.
    """
    from tools.lazy_deps import LAZY_DEPS, feature_specs

    optional_dependencies = _load_optional_dependencies()

    pyproject_pins: dict[str, set[str]] = {}
    for specs in optional_dependencies.values():
        for package, version in _exact_pins(specs).items():
            pyproject_pins.setdefault(package, set()).add(version)

    lazy_pins: dict[str, set[str]] = {}
    for feature in LAZY_DEPS:
        for package, version in _exact_pins(feature_specs(feature)).items():
            lazy_pins.setdefault(package, set()).add(version)

    shared = sorted(set(pyproject_pins) & set(lazy_pins))
    assert shared, "expected lazy features to resolve pins declared in pyproject"

    drift = {
        package: {
            "pyproject": sorted(pyproject_pins[package]),
            "lazy_deps": sorted(lazy_pins[package]),
        }
        for package in shared
        if not lazy_pins[package] <= pyproject_pins[package]
    }
    assert not drift, (
        "every pin a lazy feature resolves must come from a pyproject extra — "
        "a version appearing only on the lazy side means the extra resolver "
        f"invented it. Drift: {drift}"
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
    from tools.lazy_deps import LAZY_DEPS, feature_specs

    drift = {}
    seen = set()
    for feature in LAZY_DEPS:
        for package, pin in _exact_pins(feature_specs(feature)).items():
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
        "lazy-resolved exact pins must match the uv.lock resolved version for "
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
    from tools.lazy_deps import feature_specs

    lazy_pin = _exact_pins(feature_specs("tool.trace_upload")).get("huggingface-hub")
    assert lazy_pin, "tool.trace_upload must exact-pin huggingface-hub"

    locked = _uv_lock_version("huggingface-hub")
    assert lazy_pin == locked, (
        "the [trace-upload] extra pins huggingface-hub=="
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

    from tools.lazy_deps import feature_specs

    pin = _exact_pins(feature_specs("tool.trace_upload")).get("huggingface-hub")
    assert pin, "tool.trace_upload must exact-pin huggingface-hub"
    transformers_window = SpecifierSet(">=1.5.0,<2")
    assert Version(pin) in transformers_window, (
        f"huggingface-hub=={pin} falls outside transformers' accepted "
        "range (>=1.5.0,<2). The lazy refresh would downgrade the shared "
        "package and break Hindsight local embeddings (#60783)."
    )
