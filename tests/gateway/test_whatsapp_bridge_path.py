"""Regression test for the WhatsApp default bridge path resolution.

The bug: ``WhatsAppAdapter._DEFAULT_BRIDGE_DIR`` used ``parents[2]`` from
``plugins/platforms/whatsapp/adapter.py``, which resolves to
``<repo>/plugins/scripts/whatsapp-bridge`` — a directory that does not
exist. The installer (``hermes_cli/main.py``) writes the bridge to
``<repo>/scripts/whatsapp-bridge``. The mismatch made every gateway start
log ``Bridge script not found`` and fail to connect WhatsApp until a manual
symlink papered over it. The line was correct when the adapter lived at
``gateway/platforms/whatsapp.py`` (one directory shallower); relocating it
into ``plugins/platforms/whatsapp/`` deepened it by one without bumping the
index.

These tests pin the invariant so a future relocation that breaks the index
again fails loudly in CI instead of silently in production.
"""

from pathlib import Path


# Repo root: this file is at <repo>/tests/gateway/test_whatsapp_bridge_path.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _installer_bridge_dir() -> Path:
    """Mirror the path the installer writes the bridge to.

    See ``hermes_cli/main.py``::

        project_root = Path(__file__).resolve().parents[1]
        bridge_dir = project_root / "scripts" / "whatsapp-bridge"
    """
    return _REPO_ROOT / "scripts" / "whatsapp-bridge"


def test_default_bridge_dir_matches_installer_path():
    """The adapter's default bridge dir must equal the installer's bridge dir.

    This is the load-bearing invariant: if the adapter looks somewhere the
    installer never writes, the gateway can never find the bridge after a
    clean ``hermes update``.
    """
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    assert WhatsAppAdapter._DEFAULT_BRIDGE_DIR == _installer_bridge_dir()


def test_default_bridge_js_exists_on_disk():
    """The resolved default bridge.js must actually exist in the repo tree.

    Guards against the path drifting away from the committed bridge source.
    """
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    bridge_js = WhatsAppAdapter._DEFAULT_BRIDGE_DIR / "bridge.js"
    assert bridge_js.exists(), f"bridge.js missing at {bridge_js}"


def test_default_bridge_dir_is_not_under_plugins():
    """Negative guard: the old ``parents[2]`` bug pointed under plugins/.

    The bridge lives at the repo root's ``scripts/``, never inside
    ``plugins/``. If this ever resolves under ``plugins/`` again, the
    off-by-one has regressed.
    """
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    parts = WhatsAppAdapter._DEFAULT_BRIDGE_DIR.parts
    assert "plugins" not in parts, (
        f"bridge dir wrongly resolved under plugins/: "
        f"{WhatsAppAdapter._DEFAULT_BRIDGE_DIR}"
    )
