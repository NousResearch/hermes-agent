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


def _resolve_install_layout_body() -> str:
    """Return just the body of resolve_install_layout(), bounded by its
    opening signature and the next top-level ``}`` close brace.

    Using the function body (not "first ``return 0`` after a marker") guards
    the tests below against future refactors that hoist the export above
    another conditional with its own early-return, or that insert an early-
    return between the marker and the export — both of which would leave the
    export unreachable while a less-strict assertion still passed.
    """
    text = INSTALL_SH.read_text(encoding="utf-8")
    head, _, rest = text.partition("resolve_install_layout() {\n")
    assert rest, "Could not find resolve_install_layout() in scripts/install.sh"
    body, _, _ = rest.partition("\n}\n")
    assert body, "Could not find resolve_install_layout() closing brace"
    return body


def test_root_fhs_layout_exports_world_readable_uv_python_dirs() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert 'export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/usr/local/share/uv/python}"' in text
    assert 'export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-/usr/local/share/uv/bin}"' in text


def test_root_fhs_uv_python_export_is_inside_root_branch() -> None:
    """The export must live in the root-FHS branch of resolve_install_layout,
    after ``ROOT_FHS_LAYOUT=true`` and before the branch's ``return 0``, so
    non-root and Termux installs are unaffected. Bound the slice by the
    function body (not "next return 0" in the whole file) so the assertion
    can't accept an unreachable export."""
    body = _resolve_install_layout_body()

    marker = 'ROOT_FHS_LAYOUT=true'
    assert marker in body
    after_marker = body.split(marker, 1)[1]
    return_idx = after_marker.find('return 0')
    export_idx = after_marker.find('UV_PYTHON_INSTALL_DIR')
    assert export_idx != -1, "UV_PYTHON_INSTALL_DIR export missing from root-FHS branch"
    assert return_idx != -1, "root-FHS branch must end with `return 0`"
    assert export_idx < return_idx, (
        "Export must precede the branch's `return 0` — otherwise unreachable"
    )
