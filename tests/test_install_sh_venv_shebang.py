"""Regression tests for install.sh venv entrypoint shebang fix (#47822).

When HERMES_HOME contains spaces (e.g. ``/Volumes/External Disk/Users/…/.hermes``),
pip-generated entrypoint scripts in ``venv/bin/`` have absolute shebangs like::

    #!/Volumes/External Disk/Users/…/.hermes/hermes-agent/venv/bin/python3

The kernel splits on the first space, so the script fails to execute.
The ``fix_venv_entrypoint_shebangs()`` function rewrites these to::

    #!/usr/bin/env python3
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_fix_venv_entrypoint_shebangs_defined() -> None:
    """The shebang-fix function must exist in install.sh."""
    text = INSTALL_SH.read_text()
    assert "fix_venv_entrypoint_shebangs()" in text


def test_fix_venv_entrypoint_shebangs_called_in_setup_path() -> None:
    """setup_path() must call fix_venv_entrypoint_shebangs early."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"setup_path\(\)\s*\{(.+?)^(?:\w[^\(]*\(\)\s*\{|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "setup_path() function not found"
    body = match.group(1)
    assert "fix_venv_entrypoint_shebangs" in body, (
        "setup_path() must call fix_venv_entrypoint_shebangs"
    )


def test_shebang_fix_only_triggers_on_spaces() -> None:
    """The function early-returns when the venv path has no spaces."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"fix_venv_entrypoint_shebangs\(\)\s*\{(.+?)^\}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "fix_venv_entrypoint_shebangs() body not found"
    body = match.group(1)
    # Must check for spaces in the path before doing any work.
    assert "case" in body and " " in body, (
        "Function must check for spaces in the path before rewriting shebangs"
    )


def test_shebang_fix_uses_env_python() -> None:
    """The rewrite must use #!/usr/bin/env, not a hardcoded path."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"fix_venv_entrypoint_shebangs\(\)\s*\{(.+?)^\}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match
    body = match.group(1)
    assert "/usr/bin/env" in body, (
        "Shebang rewrite must use #!/usr/bin/env for space safety"
    )


def test_shebang_fix_reads_first_line() -> None:
    """The function must read the shebang line to detect python entrypoints."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"fix_venv_entrypoint_shebangs\(\)\s*\{(.+?)^\}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match
    body = match.group(1)
    assert "read" in body and "first_line" in body, (
        "Function must read the first line of each script to check for python shebangs"
    )
