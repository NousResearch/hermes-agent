"""Regression for #31383.

`install.sh --no-venv` historically fell into `setup_path()`'s else branch
which only ran `which hermes` to locate the entry point. But `install_deps()`
calls `uv sync --extra all --locked` with `UV_PROJECT_ENVIRONMENT=$INSTALL_DIR/venv`,
which creates `$INSTALL_DIR/venv/bin/hermes` regardless of `USE_VENV`. The result
was that `--no-venv` users got a working entry point at `$INSTALL_DIR/venv/bin/hermes`
but no `~/.local/bin/hermes` shim, so `which hermes` returned empty after install.

`setup_path()` must fall back to `$INSTALL_DIR/venv/bin/hermes` in the
`--no-venv` branch so the shim is still created.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_setup_path_function(text: str) -> str:
    """Return the body of the `setup_path()` shell function."""
    match = re.search(
        r"^setup_path\(\)\s*\{\n(?P<body>.*?)^\}\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, "setup_path() function not found in install.sh"
    return match.group("body")


def test_setup_path_no_venv_branch_falls_back_to_install_dir_venv() -> None:
    """The else branch (USE_VENV=false) must check $INSTALL_DIR/venv/bin/hermes
    before giving up — because `uv sync` creates that venv anyway."""
    body = _extract_setup_path_function(INSTALL_SH.read_text())

    # The fallback path must appear inside the function.
    assert '$INSTALL_DIR/venv/bin/hermes' in body, (
        "setup_path() --no-venv branch must fall back to "
        "$INSTALL_DIR/venv/bin/hermes (uv sync creates it regardless of USE_VENV). "
        "See #31383."
    )

    # The fallback must be guarded so a real system-wide hermes (e.g. user
    # has hermes installed via pipx) still wins.
    assert 'command -v hermes' in body or 'which hermes' in body, (
        "setup_path() --no-venv branch should still try a system PATH lookup "
        "before falling back to the install-dir venv."
    )


def test_setup_path_only_reaches_not_found_warning_after_both_lookups() -> None:
    """The `hermes not found on PATH after install` warning must be reachable
    only after both the PATH probe AND the $INSTALL_DIR/venv/bin/hermes
    fallback fail. Regression for the original #31383 behavior where the
    else branch returned early after only `which hermes`.
    """
    body = _extract_setup_path_function(INSTALL_SH.read_text())

    # Locate the "hermes not found on PATH after install" warning.
    not_found_idx = body.find('hermes not found on PATH after install')
    assert not_found_idx >= 0

    # The $INSTALL_DIR/venv/bin/hermes fallback must occur BEFORE that warning.
    fallback_idx = body.find('$INSTALL_DIR/venv/bin/hermes')
    assert fallback_idx >= 0
    assert fallback_idx < not_found_idx, (
        "The $INSTALL_DIR/venv/bin/hermes fallback must run before the "
        "'hermes not found' warning, otherwise --no-venv silently skips "
        "shim creation even when uv created the venv."
    )
