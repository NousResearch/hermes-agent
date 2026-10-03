"""living-subsystems plugin — independent JSON-backed subsystems exposed as ``hermes subsystems``.

The classes are importable for direct use; the CLI gives cron a stable entry point
(``hermes subsystems run governance``) without any change to the core agent loop.
"""

from __future__ import annotations

import json
from typing import Dict, Type

from .base import Subsystem
from .fitness_builder import FitnessBuilder
from .governance import Governance
from .reflective_evolution import ReflectiveEvolution
from .science_loop import ScienceLoop

SUBSYSTEMS: Dict[str, Type[Subsystem]] = {
    "governance": Governance,
    "science-loop": ScienceLoop,
    "reflective-evolution": ReflectiveEvolution,
    "fitness-builder": FitnessBuilder,
}

__all__ = ["Subsystem", "SUBSYSTEMS", "Governance", "ScienceLoop", "ReflectiveEvolution", "FitnessBuilder", "register"]


def _setup_cli(subparser) -> None:
    sub = subparser.add_subparsers(dest="subsystems_action")
    for action, help_text in (("status", "Show subsystem status"), ("run", "Run subsystem(s)")):
        p = sub.add_parser(action, help=help_text)
        p.add_argument("name", nargs="?", choices=sorted(SUBSYSTEMS), help="Subsystem (default: all)")


def _handle_cli(args) -> int:
    action = getattr(args, "subsystems_action", None)
    if action not in ("status", "run"):
        print("usage: hermes subsystems {status,run} [name]")
        return 2
    names = [args.name] if args.name else sorted(SUBSYSTEMS)
    out = {n: getattr(SUBSYSTEMS[n](), action)() for n in names}
    print(json.dumps(out, indent=2, default=str))
    return 0 if all(r["ok"] for r in out.values()) else 1


def register(ctx) -> None:
    ctx.register_cli_command(
        name="subsystems", help="Run or inspect living subsystems (governance, science loop, ...)",
        setup_fn=_setup_cli, handler_fn=_handle_cli)
