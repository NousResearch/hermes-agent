"""Dependency-light console entry for ``hermes``.

Runs stdlib-only update-marker recovery *before* importing ``hermes_cli.main``,
so a wiped ``python-dotenv`` (or other probed package) cannot prevent recovery
from starting (#57828 / #58004 review).
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> object:
    if argv is not None:
        # Tests may pass argv; keep sys.argv consistent for downstream CLI.
        sys.argv = [sys.argv[0], *argv]

    # UTF-8 stdio on Windows — same guard as every other Hermes entry point.
    try:
        import hermes_bootstrap  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        try:
            hermes_bootstrap.harden_import_path()
        except Exception:
            pass

    # Marker recovery must not import hermes_cli.main / env_loader / dotenv.
    from hermes_update_recovery import maybe_recover

    maybe_recover()

    from hermes_cli.main import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
