#!/usr/bin/env python3
"""Standalone CLI helper for agent interaction with the Scutl platform.

This script delegates to ``scutl._cli.main()`` when the ``scutl-sdk`` package
is installed.  It can also be run directly from a source checkout.
"""

try:
    from scutl._cli import main
except ImportError:
    import sys
    print(
        '{"error": "scutl-sdk is not installed. Run: pip install scutl-sdk"}',
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    main()
