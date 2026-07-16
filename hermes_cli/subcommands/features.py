"""``hermes features`` subcommand — manage the data-dir activation ledger.

Provides ``list`` and ``disable <name>`` for the persistent feature ledger
stored at ``$HERMES_HOME/state/features.json`` (see
``tools.lazy_deps``).  ``enable`` is implicit — using a feature (via
``ensure()``) records it automatically.
"""

from __future__ import annotations

import sys
from typing import Callable


def build_features_parser(subparsers, *, cmd_features: Callable) -> None:
    """Attach the ``features`` subcommand to ``subparsers``."""
    features_parser = subparsers.add_parser(
        "features",
        help="Manage lazy-feature activation ledger",
        description=(
            "List or disable lazily-installed backend features tracked in the "
            "activation ledger ($HERMES_HOME/state/features.json).\n\n"
            "Features are recorded automatically when ensure() installs them. "
            "Use 'hermes features list' to see what's activated and "
            "'hermes features disable <name>' to remove one."
        ),
    )
    features_sub = features_parser.add_subparsers(dest="features_action")

    # hermes features list
    features_sub.add_parser(
        "list",
        help="Show activated features from the ledger",
    )

    # hermes features disable <name>
    disable_p = features_sub.add_parser(
        "disable",
        help="Remove a feature from the activation ledger",
    )
    disable_p.add_argument(
        "name",
        help="Feature name to remove (e.g. memory.honcho)",
    )

    # hermes features apply-ledger [--json]
    apply_p = features_sub.add_parser(
        "apply-ledger",
        help="Re-install all ledger features against the current venv",
        description=(
            "Runs ensure() for every feature in the ledger. Used by the "
            "updater post-flip to restore lazy backends into a new venv. "
            "Failures are reported as warnings, never blocking."
        ),
    )
    apply_p.add_argument(
        "--venv-python",
        default=None,
        help="Path to the venv's Python executable (default: current interpreter)",
    )
    apply_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON instead of human-readable text",
    )

    features_parser.set_defaults(func=cmd_features)
