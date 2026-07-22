"""Mandate file loading and HMAC signing for live execution."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from hermes_trader.config import TRADER_HOME_SUBDIR

MANDATE_VERSION = 1
MANDATE_FILENAME = "mandate.json"


@dataclass(frozen=True)
class Mandate:
    version: int
    wallet_address: str
    signed_at: str
    expires_at: Optional[str]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "wallet_address": self.wallet_address,
            "signed_at": self.signed_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Mandate":
        return cls(
            version=int(data.get("version", 0)),
            wallet_address=str(data.get("wallet_address", "")).strip().lower(),
            signed_at=str(data.get("signed_at", "")).strip(),
            expires_at=_optional_str(data.get("expires_at")),
            signature=str(data.get("signature", "")).strip().lower(),
        )


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_mandate_path() -> Path:
    env_path = os.environ.get("HERMES_TRADER_MANDATE", "").strip()
    if env_path:
        return Path(env_path)
    return _hermes_home() / TRADER_HOME_SUBDIR / MANDATE_FILENAME


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signing_key() -> bytes:
    """Derive mandate HMAC key from env (never log or persist)."""
    secret = os.environ.get("HERMES_TRADER_MANDATE_SECRET", "").strip()
    if secret:
        return secret.encode("utf-8")
    private_key = os.environ.get("USER_PRIVATE_KEY", "").strip()
    if private_key:
        return hashlib.sha256(private_key.encode("utf-8")).digest()
    return b""


def _compute_signature(payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canonical_bytes(payload), hashlib.sha256).hexdigest()


def _payload_for_signing(mandate: Mandate) -> dict[str, Any]:
    data = mandate.to_dict()
    data.pop("signature", None)
    return data


def sign_mandate(
    wallet_address: str,
    *,
    expires_at: Optional[str] = None,
    signed_at: Optional[str] = None,
    signing_key: Optional[bytes] = None,
) -> Mandate:
    """Create a signed mandate for live trading."""
    wallet = wallet_address.strip().lower()
    if not wallet:
        raise ValueError("wallet_address is required")
    key = signing_key if signing_key is not None else _signing_key()
    if not key:
        raise ValueError(
            "Cannot sign mandate: set HERMES_TRADER_MANDATE_SECRET or USER_PRIVATE_KEY"
        )
    unsigned = Mandate(
        version=MANDATE_VERSION,
        wallet_address=wallet,
        signed_at=signed_at or _utc_now_iso(),
        expires_at=expires_at,
        signature="",
    )
    signature = _compute_signature(_payload_for_signing(unsigned), key)
    return Mandate(
        version=unsigned.version,
        wallet_address=unsigned.wallet_address,
        signed_at=unsigned.signed_at,
        expires_at=unsigned.expires_at,
        signature=signature,
    )


def validate_mandate(
    mandate: Mandate,
    *,
    expected_wallet: Optional[str] = None,
    signing_key: Optional[bytes] = None,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Return (ok, message). Message is empty when ok."""
    if mandate.version != MANDATE_VERSION:
        return False, f"unsupported mandate version {mandate.version}"
    if not mandate.wallet_address:
        return False, "mandate missing wallet_address"
    if not mandate.signed_at:
        return False, "mandate missing signed_at"
    if not mandate.signature:
        return False, "mandate missing signature"

    key = signing_key if signing_key is not None else _signing_key()
    if not key:
        return False, "no signing key available (set USER_PRIVATE_KEY or HERMES_TRADER_MANDATE_SECRET)"

    expected_sig = _compute_signature(_payload_for_signing(mandate), key)
    if not hmac.compare_digest(expected_sig, mandate.signature):
        return False, "mandate signature invalid"

    wallet = (expected_wallet or os.environ.get("USER_ADDRESS", "")).strip().lower()
    if wallet and mandate.wallet_address != wallet:
        return False, "mandate wallet_address does not match USER_ADDRESS"

    if mandate.expires_at:
        try:
            expires = datetime.fromisoformat(mandate.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False, "mandate expires_at is not valid ISO-8601"
        current = now or datetime.now(timezone.utc)
        if current >= expires:
            return False, "mandate expired"

    return True, ""


def load_mandate(path: Optional[Path | str] = None) -> Optional[Mandate]:
    target = Path(path) if path is not None else default_mandate_path()
    if not target.is_file():
        return None
    with open(target, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{target}: mandate root must be a JSON object")
    return Mandate.from_mapping(data)


def save_mandate(mandate: Mandate, path: Optional[Path | str] = None) -> Path:
    target = Path(path) if path is not None else default_mandate_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(mandate.to_dict(), handle, indent=2)
        handle.write("\n")
    return target