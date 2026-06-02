from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPCRAFT = REPO_ROOT / "snap" / "snapcraft.yaml"
LAUNCHER = REPO_ROOT / "packaging" / "snap" / "hermes-snap"
NOTES = REPO_ROOT / "packaging" / "snap" / "README.md"


def _load_manifest():
    assert SNAPCRAFT.exists(), f"missing snapcraft manifest: {SNAPCRAFT}"
    return yaml.safe_load(SNAPCRAFT.read_text(encoding="utf-8"))


def test_snapcraft_manifest_is_strict_and_ubuntu_cli_scoped():
    """The manifest must be strict-confined, CLI/gateway-scoped, and GUI-free."""
    manifest = _load_manifest()

    assert manifest["name"] == "hermes-agent"
    assert manifest["confinement"] == "strict"
    assert "apps" in manifest
    assert {"hermes", "hermes-agent", "hermes-acp", "gateway"} <= set(manifest["apps"])
    assert manifest["apps"]["gateway"]["daemon"] == "simple"
    # No desktop/GUI app is packaged in the snap.
    assert "desktop" not in manifest["apps"]
    assert "apps/desktop" not in SNAPCRAFT.read_text(encoding="utf-8")


def test_snapcraft_apps_declare_network_and_home_plugs():
    """Every app needs home + network access to be usable under confinement."""
    manifest = _load_manifest()

    for name, app in manifest["apps"].items():
        plugs = set(app.get("plugs", []))
        assert {"home", "network", "network-bind"} <= plugs, (
            f"app {name!r} is missing required plugs; has {sorted(plugs)}"
        )

    # The gateway daemon must recover from crashes.
    assert manifest["apps"]["gateway"]["restart-condition"] == "on-failure"


def test_snapcraft_version_is_derived_not_literal_git():
    """Version must be adopted from the build (pyproject), not the literal 'git'."""
    manifest = _load_manifest()

    # `version: git` is not a real snapcraft auto-version directive — it would
    # ship a snap literally versioned "git". We adopt the version instead.
    assert manifest.get("version") != "git"
    assert manifest.get("adopt-info") == "hermes"
    assert "craftctl set version" in SNAPCRAFT.read_text(encoding="utf-8")


def test_snapcraft_bundles_modern_node_runtime_for_tui():
    """The TUI needs node >=20; the snap must build+stage its own Node runtime.

    Guards against regressing to a snap that bundles an unrunnable TUI (no Node)
    or relies on core24's apt nodejs (v18, too old).
    """
    manifest = _load_manifest()
    text = SNAPCRAFT.read_text(encoding="utf-8")

    assert "tui" in manifest["parts"], "missing dedicated `tui` part"
    tui = manifest["parts"]["tui"]
    assert tui["source"] == "ui-tui"
    # A Node 20.x runtime is fetched and the bundle is built + staged.
    assert "NODE_VERSION=20" in text
    assert "nodejs.org/dist" in text
    assert "npm run build" in text
    assert "bin/node" in text
    # package.json must be staged beside dist/ so Node treats the bundle as ESM
    # ("type": "module"); without it `node dist/entry.js` parses as CommonJS.
    assert "tui/package.json" in text


def test_snapcraft_builds_and_stages_dashboard_web_assets():
    """The dashboard's static bundle is a build artifact, so the snap must build
    and stage it (web_dist) rather than relying on a pre-built checkout."""
    manifest = _load_manifest()
    text = SNAPCRAFT.read_text(encoding="utf-8")

    assert "web" in manifest["parts"], "missing dedicated `web` part"
    web = manifest["parts"]["web"]
    assert web["source"] == "web"
    assert "vite build" in text
    assert "web_dist" in text


def test_snap_launcher_sets_snap_safe_runtime_environment():
    """The launcher must redirect mutable state and expose bundled assets."""
    assert LAUNCHER.exists(), f"missing launcher: {LAUNCHER}"
    launcher = LAUNCHER.read_text(encoding="utf-8")

    # State is redirected into snap-managed writable storage.
    assert 'SNAP_COMMON_DIR="${SNAP_USER_COMMON:-' in launcher
    assert 'HERMES_HOME="${HERMES_HOME:-${HERMES_STATE_DIR}}"' in launcher
    # The install is flagged as snap-managed.
    assert 'HERMES_MANAGED="${HERMES_MANAGED:-snap}"' in launcher
    # Confinement-incompatible auto-installs are disabled.
    assert 'HERMES_DISABLE_LAZY_INSTALLS="${HERMES_DISABLE_LAZY_INSTALLS:-1}"' in launcher
    assert 'HERMES_SKIP_NODE_BOOTSTRAP="${HERMES_SKIP_NODE_BOOTSTRAP:-1}"' in launcher
    # Bundled assets and the staged Node runtime are wired up.
    assert "HERMES_BUNDLED_SKILLS" in launcher
    assert "HERMES_OPTIONAL_SKILLS" in launcher
    assert 'HERMES_NODE="${HERMES_NODE:-${SNAP}/bin/node}"' in launcher


def test_snap_launcher_routes_commands_to_hermes():
    """The launcher must dispatch entrypoints / subcommands to the CLI."""
    launcher = LAUNCHER.read_text(encoding="utf-8")

    # Direct entrypoints exec themselves; everything else routes through `hermes`.
    assert 'exec "${command_name}" "$@"' in launcher
    assert 'exec hermes "${command_name}" "$@"' in launcher


def test_snap_packaging_notes_cover_command_alternatives():
    """The notes must document the snap-specific command rewrites."""
    assert NOTES.exists(), f"missing packaging notes: {NOTES}"
    notes = NOTES.read_text(encoding="utf-8")

    assert "update" in notes
    assert "snap refresh hermes-agent" in notes
    assert "uninstall" in notes
    assert "snap remove hermes-agent" in notes
    assert "gateway start|stop|restart|status" in notes
    assert "computer-use" in notes
