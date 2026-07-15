#!/usr/bin/env python3
"""Source-tree entry point for the privileged public Discord connector."""

from gateway.discord_connector_bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main())
