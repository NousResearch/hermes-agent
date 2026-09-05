"""Regression tests for install.sh path quoting (issue #40820).

When the user's ``$HOME`` (and therefore ``$HERMES_HOME``) lives under a path
containing spaces — e.g. an external drive mounted at
``/Volumes/External Disk/Users/<name>`` — bare command-position expansions of
the managed tool paths (``$UV_CMD`` = ``$HERMES_HOME/bin/uv``,
``$HERMES_CMD`` = the installed hermes launcher) word-split and the installer
fails at the ``python-deps``/``venv`` stage with
``/Volumes/External: No such file or directory``.

Every command-position use of these variables must be double-quoted.
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
# Both Unix installers resolve the same managed uv path, so both have to quote
# it. test_install_sh_uv_lock_config.py already pins their run_locked_uv_sync()
# bodies to be byte-identical, which only holds if they are fixed together.
SETUP_SH = REPO_ROOT / "setup-hermes.sh"


def _unquoted_expansions(text: str, var: str) -> list[str]:
    """Return source lines where ``$<var>`` appears without an immediately
    preceding double quote (i.e. ``$VAR`` / ``${VAR}`` rather than ``"$VAR"``)."""
    offending = []
    pattern = re.compile(r"\$\{?" + re.escape(var) + r"\b")
    for line in text.splitlines():
        for m in pattern.finditer(line):
            start = m.start()
            # A properly quoted use is preceded by a double quote: "$VAR ...".
            # Assignments (VAR=...) never produce a "$VAR" match because they
            # have no leading '$'.
            if start == 0 or line[start - 1] != '"':
                offending.append(line.strip())
    return offending


@pytest.mark.parametrize("script", (INSTALL_SH, SETUP_SH), ids=lambda p: p.name)
def test_uv_cmd_always_quoted(script: Path) -> None:
    offending = _unquoted_expansions(script.read_text(), "UV_CMD")
    assert not offending, (
        f"Bare $UV_CMD in {script.name} must be double-quoted (breaks when "
        f"$HERMES_HOME has spaces, issue #40820):\n  " + "\n  ".join(offending)
    )


def test_hermes_cmd_always_quoted() -> None:
    text = INSTALL_SH.read_text()
    offending = _unquoted_expansions(text, "HERMES_CMD")
    assert not offending, (
        "Bare $HERMES_CMD must be double-quoted (breaks when $HERMES_HOME has "
        f"spaces, issue #40820):\n  " + "\n  ".join(offending)
    )
