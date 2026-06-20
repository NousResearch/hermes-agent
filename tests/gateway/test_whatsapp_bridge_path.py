"""Regression test for WhatsApp _DEFAULT_BRIDGE_DIR path resolution.

Issue #49831: After the adapter was relocated into
``plugins/platforms/whatsapp/``, the ``parents[2]`` index pointed at
``plugins/`` instead of the repo root, so the default bridge path
resolved to ``plugins/scripts/whatsapp-bridge/`` (non-existent) instead
of ``scripts/whatsapp-bridge/``.

The fix changes ``parents[2]`` → ``parents[3]``.
"""

from pathlib import Path


def test_default_bridge_dir_resolves_to_repo_root_scripts():
    """_DEFAULT_BRIDGE_DIR must point at <repo>/scripts/whatsapp-bridge."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    bridge_dir = WhatsAppAdapter._DEFAULT_BRIDGE_DIR
    assert bridge_dir.name == "whatsapp-bridge"
    assert bridge_dir.parent.name == "scripts"

    # The scripts/ directory must be at repo root (sibling of run_agent.py
    # or .git), not inside plugins/.
    repo_root = bridge_dir.parent.parent
    assert (repo_root / ".git").exists() or (repo_root / "run_agent.py").exists(), (
        f"_DEFAULT_BRIDGE_DIR parent chain does not reach repo root: {bridge_dir}"
    )


def test_default_bridge_dir_not_under_plugins():
    """_DEFAULT_BRIDGE_DIR must NOT resolve under plugins/ (the old bug)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    bridge_dir = WhatsAppAdapter._DEFAULT_BRIDGE_DIR
    parts = bridge_dir.parts
    # "plugins" should NOT appear between the repo root and "scripts"
    # Find where "scripts" is and check the part before it
    try:
        scripts_idx = parts.index("scripts")
    except ValueError:
        raise AssertionError(f"'scripts' not in path: {bridge_dir}")

    if scripts_idx > 0:
        preceding = parts[scripts_idx - 1]
        assert preceding != "plugins", (
            f"_DEFAULT_BRIDGE_DIR resolves under plugins/: {bridge_dir}"
        )
