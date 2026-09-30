"""CLI subcommand parser builders for ``hermes <subcommand>``.

Each group owns a ``build_<group>_parser(subparsers, ...)`` in its own module; ``main()``
calls them. ``cmd_*`` handlers are dependency-injected so no module imports ``main``
(cycle avoidance). Shared parser helpers live in ``_shared.py``.
"""

from __future__ import annotations
