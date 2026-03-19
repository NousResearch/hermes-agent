"""
Protected Chain Memory Core

This module contains the core upload/download logic for on-chain memory storage.
The source is compiled to a binary for distribution - source should NOT be public.

COSMOS TOKEN REQUIRED: 100,000+ COSMOS on Solana to use this feature.
Token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

To compile for distribution:
    python scripts/build_protected.py --method cython

Copyright (c) 2026 Cosmos AI. All rights reserved.
"""

from .core import (
    ChainUploader,
    ChainDownloader,
    verify_installation,
    get_fingerprint,
    verify_cosmos_for_operation,
    MIN_COSMOS_REQUIRED,
    COSMOS_TOKEN_MINT,
)

__all__ = [
    "ChainUploader",
    "ChainDownloader",
    "verify_installation",
    "get_fingerprint",
    "verify_cosmos_for_operation",
    "MIN_COSMOS_REQUIRED",
    "COSMOS_TOKEN_MINT",
]

# Note: Verification happens on each upload/download operation now
# No longer verify on import to avoid blocking non-chain-memory usage
