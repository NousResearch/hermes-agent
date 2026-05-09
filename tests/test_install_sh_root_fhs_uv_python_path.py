"""Regression test for install.sh root-mode uv Python install path.

When installing as root with the FHS layout (INSTALL_DIR=/usr/local/lib/...),
``uv python install`` must place the managed Python under a world-readable
location, otherwise the venv interpreter ends up at ``/root/.local/share/uv/...``
and the shared ``/usr/local/bin/hermes`` wrapper fails for non-root users with
"bad interpreter: Permission denied".  See #21457.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_root_fhs_layout_exports_world_readable_uv_python_dirs() -> None:
    text = INSTALL_SH.read_text()

    assert 'export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/usr/local/share/uv/python}"' in text
    assert 'export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-/usr/local/share/uv/bin}"' in text


def test_root_fhs_uv_python_export_is_inside_root_branch() -> None:
    """The export must live in the root-FHS branch of resolve_install_layout,
    above its `return 0`, so non-root and Termux installs are unaffected."""
    text = INSTALL_SH.read_text()

    marker = 'ROOT_FHS_LAYOUT=true'
    assert marker in text
    after_marker = text.split(marker, 1)[1]
    return_idx = after_marker.find('return 0')
    export_idx = after_marker.find('UV_PYTHON_INSTALL_DIR')
    assert export_idx != -1
    assert export_idx < return_idx
