"""Lightweight process entry point for the standalone Hermes agent.

Keep auth-residence validation here so the installed ``hermes-agent`` command
can fail before importing ``run_agent``, whose library module intentionally
initializes runtime integrations at import time.
"""

# Keep Windows UTF-8 setup first, matching every Hermes process entry point.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

import sys

from hermes_constants import HermesAuthHomeError, validate_hermes_auth_home


def main():
    """Validate launcher state, then preserve the existing agent behavior."""
    try:
        validate_hermes_auth_home()
    except HermesAuthHomeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    from run_agent import main as run_agent_main

    return run_agent_main()


if __name__ == "__main__":
    raise SystemExit(main())
