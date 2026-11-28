"""The ``hermes pm`` entrypoint: dotenv first, then dispatch.

``hermes_cli/main.py`` dispatches the ``pm`` subcommand very early — before the
application's dependency imports — so that ``pm`` can run in a stripped
environment. That early position had a consequence nobody intended: the module
level ``load_hermes_dotenv()`` call in ``main.py`` sits BELOW the dispatch, so
``hermes pm`` started with an environment that never saw the home dotenv.

Every other entrypoint (the CLI itself, the desktop update probe) loaded
``<HERMES_HOME>/.env`` first. ``pm`` did not, and it is invisible until the one
thing ``pm`` reads from the environment is a credential.

``pm.update._github_headers()`` lifts the api.github.com quota from the
anonymous 60 requests/hour **per exit IP** to 5000 requests/hour **per
account**, using ``GITHUB_TOKEN`` / ``GH_TOKEN``. Behind a shared egress (office
NAT, a Clash/VPN node) the anonymous bucket is spent by every other client
behind the same address, so every GitHub-sourced package failed to resolve:

    ffmpeg   resolve failed: HTTP Error 403: rate limit exceeded
    gh       resolve failed: HTTP Error 403: rate limit exceeded
    git      resolve failed: HTTP Error 403: rate limit exceeded
    ripgrep  resolve failed: HTTP Error 403: rate limit exceeded
    uv       resolve failed: HTTP Error 403: rate limit exceeded

The token was present the whole time in the home dotenv. It simply had not been
loaded yet when ``pm`` started.

This module owns the ordering so ``main.py`` keeps only the dispatch itself.
"""

from __future__ import annotations

import sys
from pathlib import Path


def dispatch_pm(argv: list[str], project_root: Path) -> int:
    """Run ``hermes pm`` with the home dotenv loaded, and return its exit code.

    The dotenv load happens HERE, before ``pm.cli`` is imported, because
    ``pm`` reads its GitHub credential from ``os.environ`` at call time
    (``pm.update._github_headers()``). Loading it after the dispatch would
    leave the process on the anonymous per-IP quota.
    """
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(project_env=project_root / ".env")

    from pm.cli import main as pm_main

    return pm_main(argv)


def maybe_dispatch_pm(argument_vector: list[str], project_root: Path) -> int | None:
    """Dispatch ``hermes pm`` when it is the requested subcommand.

    Returns the pm exit code when pm ran, or ``None`` when the caller should
    continue with the normal entrypoint. A pm failure still propagates: the
    caller raises ``SystemExit`` with the returned code.
    """
    if argument_vector[1:2] != ["pm"]:
        return None
    return dispatch_pm(argument_vector[2:], project_root)


__all__ = ["dispatch_pm", "maybe_dispatch_pm"]
