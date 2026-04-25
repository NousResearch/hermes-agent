"""Regression guard for #15336 — WhatsApp bridge.js must ship in installed wheels.

Before this fix, ``scripts/whatsapp-bridge/bridge.js`` lived outside
any Python package, so ``pip install`` (and downstream Nix /
Docker / Homebrew installs) simply didn't include it in the wheel.
Users who started Hermes from a packaged install hit:

    ✗ Bridge script not found at /nix/store/.../site-packages/scripts/whatsapp-bridge/bridge.js

The bridge files were moved into ``gateway/whatsapp_bridge/`` (a real
sub-package of ``gateway``) and registered as setuptools
``package-data`` so they end up at ``site-packages/gateway/
whatsapp_bridge/`` on every wheel-based install path.

These tests pin two invariants:

1. The bridge directory is *findable* from the ``gateway`` package's
   ``__file__`` — the resolution path that
   ``WhatsAppAdapter._DEFAULT_BRIDGE_DIR`` and the doctor / setup
   commands now use.
2. The expected files (``bridge.js``, ``allowlist.js``, ``package.json``)
   exist in that directory.  If a future refactor removes one of them
   without updating callers, this test fails before the next release
   ships.
"""
from pathlib import Path


def test_bridge_dir_resolves_from_gateway_package():
    """``WhatsAppAdapter._DEFAULT_BRIDGE_DIR`` must compute to a real
    directory that exists in the source tree (and therefore — given
    the ``[tool.setuptools.package-data]`` entry pinned in
    ``pyproject.toml`` — also exists at ``site-packages/gateway/
    whatsapp_bridge/`` after a wheel install)."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    bridge_dir = WhatsAppAdapter._DEFAULT_BRIDGE_DIR
    assert isinstance(bridge_dir, Path)
    assert bridge_dir.exists(), (
        f"_DEFAULT_BRIDGE_DIR resolved to {bridge_dir} which does not "
        f"exist on disk — the bridge files were likely moved without "
        f"updating the resolver, or the move regressed (#15336)"
    )
    assert bridge_dir.is_dir()


def test_bridge_dir_lives_inside_gateway_package():
    """The bridge directory must be ``gateway/whatsapp_bridge/`` so the
    ``gateway.whatsapp_bridge`` package-data entry in ``pyproject.toml``
    actually targets it.  If a future refactor moves the directory
    elsewhere without updating ``pyproject.toml``, the wheel will silently
    stop including the bridge files again — same #15336 regression we
    just fixed.
    """
    import gateway as _gateway_pkg
    from gateway.platforms.whatsapp import WhatsAppAdapter

    bridge_dir = WhatsAppAdapter._DEFAULT_BRIDGE_DIR
    expected_parent = Path(_gateway_pkg.__file__).resolve().parent
    assert bridge_dir.parent == expected_parent, (
        f"bridge dir parent is {bridge_dir.parent}, expected "
        f"{expected_parent}.  If you moved the bridge, update "
        f"pyproject.toml ``[tool.setuptools.package-data]`` to match."
    )
    assert bridge_dir.name == "whatsapp_bridge"


def test_bridge_dir_contains_required_files():
    """Pin the file list the consumers depend on — ``bridge.js``
    (the entry-point), ``allowlist.js`` (loaded by bridge.js at
    runtime), and ``package.json`` (so ``npm install`` works in
    the ``hermes setup`` flow)."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    bridge_dir = WhatsAppAdapter._DEFAULT_BRIDGE_DIR
    for required in ("bridge.js", "allowlist.js", "package.json"):
        target = bridge_dir / required
        assert target.exists(), (
            f"required bridge file missing: {target}.  Either the file "
            f"was deleted (and callers need updating) or the move "
            f"regressed (#15336)"
        )


def test_pyproject_package_data_covers_bridge_files():
    """``pyproject.toml`` must declare ``gateway.whatsapp_bridge`` as
    a package and include the bridge file globs as package-data;
    otherwise the wheel-build path leaves the directory empty even
    though the source tree has it.

    Parses ``pyproject.toml`` directly so this test catches the
    regression even when running outside an installed wheel.
    """
    try:
        import tomllib  # py3.11+
    except ImportError:  # pragma: no cover — older Pythons
        import tomli as tomllib

    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / "pyproject.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    package_data = (
        cfg.get("tool", {})
        .get("setuptools", {})
        .get("package-data", {})
    )
    bridge_globs = package_data.get("gateway.whatsapp_bridge")
    assert bridge_globs, (
        "pyproject.toml is missing the "
        "``[tool.setuptools.package-data] \"gateway.whatsapp_bridge\"`` "
        "entry — wheel installs will not contain bridge.js (#15336)"
    )
    # Pin the patterns we rely on so a future trim doesn't silently
    # drop the package.json or the JS sources.
    pattern_str = " ".join(bridge_globs)
    for needle in ("*.js", "package.json"):
        assert needle in pattern_str, (
            f"package-data globs for gateway.whatsapp_bridge are missing "
            f"{needle!r}: {bridge_globs!r}"
        )


def test_bridge_init_marker_present():
    """The ``__init__.py`` marker is what makes
    ``gateway/whatsapp_bridge/`` a regular setuptools package (rather
    than a PEP 420 namespace package, which has spottier package-data
    support across setuptools versions).  Without it, ``find_packages``
    skips the directory and the wheel ships nothing."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    init = WhatsAppAdapter._DEFAULT_BRIDGE_DIR / "__init__.py"
    assert init.exists(), (
        f"missing {init} — without it the directory isn't a real "
        f"setuptools package and package-data is skipped on some "
        f"setuptools versions (#15336)"
    )
