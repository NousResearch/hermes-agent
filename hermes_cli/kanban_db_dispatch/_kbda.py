"""Late-bound access to ``hermes_cli.kanban_db_default_assignee``.

This package is imported by that module, so a direct import here would run the origin while
this package is only half built. The origin instance is captured at import time (the cycle
partner is already in ``sys.modules``), exactly as the single-module version bound it at its
own bottom; only a first use from outside that cycle imports it on demand.
"""

import sys


_origin = sys.modules.get("hermes_cli.kanban_db_default_assignee")


def __getattr__(name):
    global _origin
    if _origin is None:
        import importlib

        _origin = importlib.import_module("hermes_cli.kanban_db_default_assignee")
    return getattr(_origin, name)
