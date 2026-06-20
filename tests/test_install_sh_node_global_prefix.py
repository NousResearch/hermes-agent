"""Regression tests for Hermes-managed Node PATH isolation.

When the installer falls back to a bundled Node under ``$HERMES_HOME/node``,
Hermes must keep node/npm/npx private. Older installers linked them into the
command link dir (usually ``~/.local/bin``), which could shadow nvm/fnm/Volta.
The managed npm prefix stays inside ``$HERMES_HOME/node`` so npm global bins
land in ``$HERMES_HOME/node/bin``: private to Hermes, but still visible to
Hermes subprocess PATH construction and dependency detection.
"""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_install_sh_keeps_bundled_npm_global_prefix_inside_hermes_home() -> None:
    text = INSTALL_SH.read_text()

    # prefix=$HERMES_HOME/node keeps global bins in $HERMES_HOME/node/bin,
    # which Hermes can add to subprocess PATH without polluting ~/.local/bin.
    assert "configure_managed_node_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"' in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$link_dir")" > "$HERMES_HOME/node/etc/npmrc"' not in text


def test_install_sh_repairs_existing_managed_node_on_rerun() -> None:
    """The redirect must run on every install (not just fresh Node installs),
    so re-running the installer repairs pre-existing managed installs whose
    Node is already up to date and would otherwise skip install_node."""
    text = INSTALL_SH.read_text()

    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]
    assert "configure_managed_node_npm_prefix" in check_node_body
    assert "remove_managed_node_path_symlinks" in check_node_body

    # No-op guard so it's safe to call when there is no managed Node.
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text
    assert "node_link_points_into_hermes_node" in text


def test_node_bootstrap_keeps_bundled_npm_global_prefix_inside_hermes_home() -> None:
    text = NODE_BOOTSTRAP.read_text()

    assert "_nb_configure_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$HERMES_HOME/node" > "$HERMES_HOME/node/etc/npmrc"' in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"' not in text

    # Runs at the top of ensure_node so existing managed installs are repaired
    # even when a modern Node is already present (early return path).
    ensure_node_body = text.split("ensure_node()", 1)[1]
    assert "_nb_configure_npm_prefix" in ensure_node_body
    assert "_nb_remove_legacy_node_links" in ensure_node_body
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text
    assert "_nb_link_points_into_hermes_node" in text


def test_installers_never_link_managed_node_tools_into_command_dir() -> None:
    install_text = INSTALL_SH.read_text()
    bootstrap_text = NODE_BOOTSTRAP.read_text()

    forbidden = (
        'ln -sf "$HERMES_HOME/node/bin/node"',
        'ln -sf "$HERMES_HOME/node/bin/npm"',
        'ln -sf "$HERMES_HOME/node/bin/npx"',
    )
    for snippet in forbidden:
        assert snippet not in install_text
        assert snippet not in bootstrap_text


def test_node_bootstrap_keeps_version_manager_and_bundled_cascade() -> None:
    text = NODE_BOOTSTRAP.read_text()
    ensure_node_body = text.split("ensure_node()", 1)[1]

    assert "_nb_try_fnm" in ensure_node_body
    assert "_nb_try_proto" in ensure_node_body
    assert "_nb_try_nvm" in ensure_node_body
    assert "_nb_try_termux_pkg" in ensure_node_body
    assert "_nb_try_brew" in ensure_node_body
    assert "_nb_install_bundled_node" in ensure_node_body


def test_node_bootstrap_version_floor_matches_desktop_build_floor(tmp_path) -> None:
    cases = [
        ("v20.18.9", False),
        ("v20.19.0", True),
        ("v22.11.0", False),
        ("v22.12.0", True),
        ("v24.0.0", True),
    ]

    for version, expected in cases:
        bin_dir = tmp_path / version / "bin"
        _write_executable(bin_dir / "node", f"#!/bin/sh\nprintf '{version}\\n'\n")
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(tmp_path / "home"),
            "HERMES_HOME": str(tmp_path / "home" / ".hermes"),
        }

        result = subprocess.run(
            ["bash", "-c", f'source "{NODE_BOOTSTRAP}"; _nb_have_modern_node'],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert (result.returncode == 0) is expected, version
