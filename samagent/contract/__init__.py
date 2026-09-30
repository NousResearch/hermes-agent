"""SamAgent contract management: freeze, ownership guard, and Contract Change Requests (CCR)."""
from __future__ import annotations

from samagent.contract.freeze import (
    ContractChangeRequest,
    ContractVersion,
    OwnershipVerdict,
    apply_ccr,
    check_git_diff_ownership,
    check_path_ownership,
    freeze_contract,
    load_ownership_map,
)

__all__ = [
    "ContractChangeRequest",
    "ContractVersion",
    "OwnershipVerdict",
    "apply_ccr",
    "check_git_diff_ownership",
    "check_path_ownership",
    "freeze_contract",
    "load_ownership_map",
]
