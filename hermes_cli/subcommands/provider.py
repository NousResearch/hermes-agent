from __future__ import annotations
from typing import Callable

def build_provider_parser(subparsers, *, cmd_provider_diagnose: Callable) -> None:
    """Adds the provider command group and its subcommands."""
    provider_parser = subparsers.add_parser("provider", help="Provider management and diagnostics")
    subparsers_provider = provider_parser.add_subparsers()
    
    # diagnose subcommand
    diagnose_parser = subparsers_provider.add_parser("diagnose", help="Test provider connectivity")
    diagnose_parser.add_argument("name", help="The name of the provider to diagnose")
    diagnose_parser.set_defaults(func=cmd_provider_diagnose)
    