"""Regression tests for root FHS uv-managed Python placement in install.sh.

When install.sh runs as root on Linux, it prefers the FHS code layout under
``/usr/local/lib/hermes-agent`` and exposes the launcher at
``/usr/local/bin/hermes``. Without overriding uv's managed-Python directories,
``uv python install`` still uses ``/root/.local/share/uv/python`` and the venv
interpreter can end up rooted in the root home. Non-root users then fail to
execute the shared launcher because its interpreter path is not world-readable.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_root_fhs_installs_export_shared_uv_python_dirs() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert "configure_uv_python_dirs()" in text
    assert 'if [ "$ROOT_FHS_LAYOUT" != true ]; then' in text
    assert 'export UV_PYTHON_INSTALL_DIR="/usr/local/share/uv/python"' in text
    assert 'export UV_PYTHON_BIN_DIR="/usr/local/share/uv/bin"' in text


def test_root_uv_python_dir_override_runs_before_uv_bootstrap() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    match = re.search(r"main\(\) \{(?P<body>.*?)^\}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "Could not locate main() in scripts/install.sh"
    body = match["body"]

    resolve_idx = body.find("resolve_install_layout")
    configure_idx = body.find("configure_uv_python_dirs")
    install_uv_idx = body.find("install_uv")
    check_python_idx = body.find("check_python")

    assert -1 not in (resolve_idx, configure_idx, install_uv_idx, check_python_idx)
    assert resolve_idx < configure_idx < install_uv_idx < check_python_idx, (
        "main() must configure shared uv Python dirs immediately after "
        "resolve_install_layout(), before uv install/Python discovery."
    )
