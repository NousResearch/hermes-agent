"""Stop a pytest run that is not using this checkout's test interpreter, before collection.

The suite runs under the isolated test environment PM builds beside the checkout's install state
(``pm.testenv``; ``source ./activate`` exports it as ``$__HERMES_TEST_PYTHON``, and
``scripts/run_tests.sh`` uses it). Another interpreter on ``PATH`` — a conda base, Homebrew or
system Python, a shell alias that outranks the activated ``PATH`` — gets as far as the first
missing dependency or the wrong dependency set, and the failure then looks like a Hermes bug.

The decision is by identity, not by name or version: when this checkout HAS a selected test
environment, the running interpreter's ``sys.prefix`` must be that environment. A checkout without
one (CI lanes that install their own interpreter, Nix devShells, a fresh clone) is not judged.
``HERMES_ALLOW_FOREIGN_TEST_PYTHON=1`` opts out explicitly.

Stdlib only at import, and nothing here writes state: it runs before the conftest sandboxes
``HERMES_HOME``, so it sees the same install state activation used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OPT_OUT_ENV = "HERMES_ALLOW_FOREIGN_TEST_PYTHON"


def _expected_test_python(project_root: Path):
    """The checkout's selected test interpreter, None when it has none. Raises when this
    interpreter cannot even import the package manager that selects it."""
    from pm.testenv import testenv_python  # noqa: PLC0415 — imported after the opt-out

    try:
        python = testenv_python(project_root)
    except Exception:  # a damaged selection is PM's to report, not this guard's
        return None
    return None if python is None else Path(python)


def foreign_interpreter_message(project_root: Path, environ=None) -> str | None:
    environ = os.environ if environ is None else environ
    if environ.get(OPT_OUT_ENV) == "1":
        return None
    current = Path(sys.prefix)
    try:
        expected = _expected_test_python(project_root)
    except Exception as exc:
        expected, reason = None, f"it cannot import the Hermes package manager ({type(exc).__name__}: {exc})"
    else:
        if expected is None:
            return None
        try:
            # <venv>/bin/python or <venv>\\Scripts\\python.exe: the venv is its sys.prefix.
            if current.resolve() == expected.parent.parent.resolve():
                return None
        except OSError:
            pass
        reason = "it is not this checkout's test environment"
    return (
        "ERROR:\n"
        "Hermes tests must run inside the activated source environment.\n\n"
        "Run:\n\n"
        f"    cd {project_root}\n"
        "    source ./activate\n"
        "    scripts/run_tests.sh <files>      (or: \"$__HERMES_TEST_PYTHON\" -m pytest <files>)\n\n"
        f"Current interpreter:\n    {sys.executable} (Python {sys.version.split()[0]}; {reason})\n"
        f"Expected:\n    {expected or 'the PM-managed test environment (pm.testenv)'}\n\n"
        "A shell alias or function named python outranks the activated PATH.\n"
        f"Set {OPT_OUT_ENV}=1 to run under another interpreter deliberately."
    )
