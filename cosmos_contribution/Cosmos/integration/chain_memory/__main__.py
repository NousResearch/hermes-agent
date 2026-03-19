#!/usr/bin/env python3
"""
Chain Memory CLI

Usage:
    python -m cosmos.integration.chain_memory setup
    python -m cosmos.integration.chain_memory push
    python -m cosmos.integration.chain_memory pull
    python -m cosmos.integration.chain_memory list
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "setup":
    # Run setup wizard
    from .setup import main
    main()
else:
    # Run main CLI
    from .startup import main
    main()
