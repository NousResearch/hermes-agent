#!/usr/bin/env python3
"""Compatibility entrypoint for the packaged privileged Discord edge.

Production and canary units execute :mod:`gateway.discord_edge_bootstrap` from
the sealed wheel.  This source-tree shim retains the historical development
entrypoint without carrying a second implementation of the privileged
boundary.
"""

from gateway.discord_edge_bootstrap import *  # noqa: F403
from gateway.discord_edge_bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main())
