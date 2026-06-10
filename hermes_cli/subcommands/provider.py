from __future__ import annotations
from typing import Callable

def build_provider_parser(subparsers, *, cmd_provider_diagnose: Callable) -> None:
    """provider komut grubunu ve alt komutlarını ekler."""
    provider_parser = subparsers.add_parser("provider", help="Provider yönetimi ve teşhisi")
    subparsers_provider = provider_parser.add_subparsers()
    
    # diagnose alt komutu
    diagnose_parser = subparsers_provider.add_parser("diagnose", help="Provider bağlantısını test et")
    diagnose_parser.add_argument("name", help="Teşhis edilecek provider ismi")
    diagnose_parser.set_defaults(func=cmd_provider_diagnose)