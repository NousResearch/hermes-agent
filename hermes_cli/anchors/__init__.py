"""Static reference anchors for TEE-attested inference providers.

Until hermes wires direct on-chain readers (Base RPC against DstackKms/DstackApp
for NEAR), we pin known-good app_id / compose_hash / os_image_hash / KMS pubkey
values captured from the production attestation endpoint. _verify_*_attestation
fails closed in strict mode if the live attestation does not match the anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_ANCHORS_DIR = Path(__file__).resolve().parent


def load_nearai_anchor() -> Dict[str, Any]:
    return json.loads((_ANCHORS_DIR / "nearai_mainnet.json").read_text())


def expected_for_model(anchor: Dict[str, Any], model: str) -> Dict[str, Any] | None:
    return anchor.get("models", {}).get(model)
