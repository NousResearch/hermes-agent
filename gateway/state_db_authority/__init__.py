"""Proof-bearing writer admission for every gateway ``SessionDB`` path.

Import is passive. The first writable handle for a canonical state.db path runs
Hermes' full health probe, opens a kernel-identity anchor, and receives a proof
whose lifetime ends with the last live admitted handle. Replacing the pathname
while that proof is live is refused as split-brain.
"""

from ._authority import AUTHORITY, GatewayStateDBAuthority
from ._install import install_gateway_state_db_authority
from ._model import (
    IntegrityVerdict,
    StateDBAdmissionBusyError,
    StateDBAdmissionError,
    StateDBAdmissionProof,
    StateDBFileIdentity,
    StateDBGenerationConflictError,
    StateDBIntegrityError,
    StateDBIntegrityReport,
    canonical_state_db_path,
    sqlite_read_only_uri,
)
from ._verify import verify_state_db_integrity


def gateway_state_db_authority_snapshot():
    return AUTHORITY.snapshot()


__all__ = [
    "GatewayStateDBAuthority",
    "IntegrityVerdict",
    "StateDBAdmissionBusyError",
    "StateDBAdmissionError",
    "StateDBAdmissionProof",
    "StateDBFileIdentity",
    "StateDBGenerationConflictError",
    "StateDBIntegrityError",
    "StateDBIntegrityReport",
    "canonical_state_db_path",
    "gateway_state_db_authority_snapshot",
    "install_gateway_state_db_authority",
    "sqlite_read_only_uri",
    "verify_state_db_integrity",
]
