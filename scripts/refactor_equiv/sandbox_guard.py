"""Refuse to run FS-writing golden runners against a real HERMES_HOME.

The state_ext golden runner writes ``config.yaml`` and seeds ``*.db`` files
under ``$HERMES_HOME``. Under the ``Determinism()`` context that is a
throwaway temp dir — but if a runner is ever invoked OUTSIDE the context
(direct CLI experiment, a mutation harness misconfiguration, a worker
following the wrong invocation), ``HERMES_HOME`` resolves to the operator's
real ``~/.hermes`` and the runner **clobbers the live config**. This
happened on 2026-07-16 (live ``~/.hermes/config.yaml`` reduced to the
runner's 2-line dashboard flag; live gateway lost toolsets/approvals).

Any golden runner that writes to ``$HERMES_HOME`` MUST call
``require_sandboxed_home()`` before its first write.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_EXTRA_TMP_ROOTS: tuple[str, ...] = (
    "/tmp",
    "/private/tmp",
    "/private/var/folders",
    "/var/folders",
)


class UnsafeHomeError(RuntimeError):
    """Raised when a golden runner would write into a real HERMES_HOME."""


def require_sandboxed_home() -> Path:
    """Return $HERMES_HOME as a Path iff it is provably a sandbox.

    A sandbox home is one that is NOT the default ``~/.hermes`` (or a path
    inside it) and that lives under the system temp dir, OR one explicitly
    whitelisted via a ``.refactor-equiv-sandbox`` marker file (for bespoke
    fixture homes checked into test data).
    """
    raw = os.environ.get("HERMES_HOME")
    if not raw:
        raise UnsafeHomeError(
            "HERMES_HOME is unset; golden runners that write files must run "
            "under scripts.refactor_equiv.determinism.Determinism() or an "
            "explicit temp home."
        )
    home = Path(raw).resolve()
    real_home = (Path.home() / ".hermes").resolve()
    if home == real_home or real_home in home.parents:
        raise UnsafeHomeError(
            f"refusing to write into the real HERMES_HOME ({home}); run under "
            "Determinism() or point HERMES_HOME at a temp directory."
        )
    if (home / ".refactor-equiv-sandbox").exists():
        return home
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if tmp_root == home or tmp_root in home.parents:
        return home
    # Also accept pytest tmp_path-style locations under /private/var or TMPDIR
    # variants that resolve() may canonicalize differently on macOS.
    # Module-level so tests can narrow it to prove the refusal branch.
    for candidate in _EXTRA_TMP_ROOTS:
        croot = Path(candidate).resolve()
        if croot == home or croot in home.parents:
            return home
    raise UnsafeHomeError(
        f"HERMES_HOME ({home}) is not a recognized sandbox (not under the "
        "system temp dir and no .refactor-equiv-sandbox marker); refusing "
        "to write."
    )
