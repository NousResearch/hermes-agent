"""Tests for docker/entrypoint.sh — UID remap chown behavior (#27221).

The entrypoint shell script cannot be unit-tested directly without Docker,
so these tests parse the script to verify that the required chown logic
is present for the install directories.
"""

import os
import re

ENTRYPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "docker", "entrypoint.sh",
)


def _read_entrypoint():
    with open(ENTRYPOINT) as f:
        return f.read()


class TestEntrypointUidRemapChown:
    """Verify that docker/entrypoint.sh chowns install dirs on UID remap."""

    def test_ui_tui_chown_present(self):
        """entrypoint.sh must chown ui-tui/ when HERMES_UID is remapped."""
        script = _read_entrypoint()
        assert "ui-tui" in script, "ui-tui directory not referenced in entrypoint"
        # Verify it's in the chown block (near needs_chown)
        block = script[script.find("needs_chown"):script.find("needs_chown") + 2000]
        assert "ui-tui" in block, "ui-tui chown not in needs_chown block"

    def test_gateway_chown_present(self):
        """entrypoint.sh must chown gateway/ when HERMES_UID is remapped."""
        script = _read_entrypoint()
        assert "gateway" in script, "gateway directory not referenced in entrypoint"
        block = script[script.find("needs_chown"):script.find("needs_chown") + 2000]
        assert "gateway" in block, "gateway chown not in needs_chown block"

    def test_chown_uses_hermes_user(self):
        """chown must use 'hermes:hermes' as the target ownership."""
        script = _read_entrypoint()
        # Find the install-dir chown loop
        for_dir_loop = re.search(
            r'for\s+_dir\s+in\s+.*\$INSTALL_DIR/ui-tui.*\$INSTALL_DIR/gateway',
            script,
            re.DOTALL,
        )
        assert for_dir_loop is not None, (
            "No loop chown-ing both ui-tui and gateway found"
        )
        # Verify the loop body uses hermes:hermes
        loop_text = for_dir_loop.group(0)
        # The chown call should appear after the for-in line
        assert "hermes:hermes" in script[script.find("ui-tui"):script.find("ui-tui") + 500]

    def test_chown_guarded_by_directory_check(self):
        """chown should only run if the directory exists."""
        script = _read_entrypoint()
        # Look for the -d guard
        assert '-d "$_dir"' in script or '-d "$INSTALL_DIR/ui-tui"' in script, (
            "Directory existence guard missing"
        )

    def test_chown_guarded_by_uid_check(self):
        """chown should only run if directory UID differs from target."""
        script = _read_entrypoint()
        # Look for the stat -c %u check
        assert "stat -c %u" in script, (
            "UID comparison guard missing"
        )
