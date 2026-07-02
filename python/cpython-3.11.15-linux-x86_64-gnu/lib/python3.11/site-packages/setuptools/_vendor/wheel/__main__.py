"""
Wheel command line tool (enables the ``python -m wheel`` syntax)
"""

from __future__ import annotations

import sys
from typing import NoReturn


def main() -> NoReturn:  # needed for console script
    if __package__ == "":
        # To be able to run 'python wheel-0.9.whl/wheel':
        import os.path

        path = os.path.dirname(os.path.dirname(__file__))
        sys.path[0:0] = [path]

    from ._commands import main as cli_main

    sys.exit(cli_main())


if __name__ == "__main__":
    main()
