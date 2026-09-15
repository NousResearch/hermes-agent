"""Signed local mandate for Hermes Agentic Trader live execution."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

MANDATE_VERSION = 2
MANDATE_FILENAME = "mandate.json"


@dataclass(frozen=True)
class Mandate:
    version: int
    wallet_address: str
    signed_at: str
    expires_at: str
    allowed_chain_ids: tuple[int, ...]
    max_trade_usd: float
    max_slippage_bps: int
    max_writes_per_hour: int
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "wallet_address": self.wallet_address,
            "signed_at": self.signed_at,
            "expires_at": self.expires_at,
            "allowed_chain_ids": list(self.allowed_chain_ids),
            "max_trade_usd": self.max_trade_usd,
            "max_slippage_bps": self.max_slippage_bps,
            "max_writes_per_hour": self.max_writes_per_hour,
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Mandate":
        chains = data.get("allowed_chain_ids") or []
        if not isinstance(chains, (list, tuple)):
            raise ValueError("mandate allowed_chain_ids must be an array")
        return cls(
            version=int(data.get("version", 0)),
            wallet_address=str(data.get("wallet_address", "")).strip().lower(),
            signed_at=str(data.get("signed_at", "")).strip(),
            expires_at=str(data.get("expires_at", "")).strip(),
            allowed_chain_ids=tuple(int(v) for v in chains),
            max_trade_usd=float(data.get("max_trade_usd", 0)),
            max_slippage_bps=int(data.get("max_slippage_bps", 0)),
            max_writes_per_hour=int(data.get("max_writes_per_hour", 0)),
            signature=str(data.get("signature", "")).strip().lower(),
        )


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def default_mandate_path() -> Path:
    override = os.environ.get("HERMES_TRADER_MANDATE", "").strip()
    if override:
        return Path(override)
    return _hermes_home() / "trader" / MANDATE_FILENAME


def _secret(signing_key: Optional[bytes] = None) -> bytes:
    if signing_key is not None:
        return signing_key
    secret = os.environ.get("HERMES_TRADER_MANDATE_SECRET", "")
    return secret.encode("utf-8") if secret else b""


def _canonical_payload(mandate: Mandate) -> dict[str, Any]:
    payload = mandate.to_dict()
    payload.pop("signature", None)
    return payload


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _signature(payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canonical_bytes(payload), hashlib.sha256).hexdigest()


def sign_mandate(
    wallet_address: str,
    *,
    allowed_chain_ids: tuple[int, ...] = (8453,),
    max_trade_usd: float = 50.0,
    max_slippage_bps: int = 100,
    max_writes_per_hour: int = 10,
    expires_at: Optional[str] = None,
    lifetime_minutes: int = 60,
    signed_at: Optional[str] = None,
    signing_key: Optional[bytes] = None,
) -> Mandate:
    wallet = wallet_address.strip().lower()
    if not wallet:
        raise ValueError("wallet_address is required")
    if not allowed_chain_ids:
        raise ValueError("allowed_chain_ids cannot be empty")
    if max_trade_usd <= 0:
        raise ValueError("max_trade_usd must be positive")
    if max_slippage_bps <= 0:
        raise ValueError("max_slippage_bps must be positive")
    if max_writes_per_hour <= 0:
        raise ValueError("max_writes_per_hour must be positive")

    key = _secret(signing_key)
    if not key:
        raise ValueError("HERMES_TRADER_MANDATE_SECRET is required")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    signed_text = signed_at or now.isoformat()
    expiry_text = expires_at or (
        now + timedelta(minutes=max(1, int(lifetime_minutes)))
    ).isoformat()

    unsigned = Mandate(
        version=MANDATE_VERSION,
        wallet_address=wallet,
        signed_at=signed_text,
        expires_at=expiry_text,
        allowed_chain_ids=tuple(int(v) for v in allowed_chain_ids),
        max_trade_usd=float(max_trade_usd),
        max_slippage_bps=int(max_slippage_bps),
        max_writes_per_hour=int(max_writes_per_hour),
        signature="",
    )
    return Mandate(
        **{
            **unsigned.__dict__,
            "signature": _signature(_canonical_payload(unsigned), key),
        }
    )


def validate_mandate(
    mandate: Mandate,
    *,
    expected_wallet: Optional[str] = None,
    signing_key: Optional[bytes] = None,
    now: Optional[datetime] = None,
) -> tuple[bool, str]:
    if mandate.version != MANDATE_VERSION:
        return False, f"unsupported mandate version {mandate.version}"
    if not mandate.wallet_address:
        return False, "mandate missing wallet_address"
    if not mandate.signed_at:
        return False, "mandate missing signed_at"
    if not mandate.expires_at:
        return False, "mandate missing expires_at"
    if not mandate.allowed_chain_ids:
        return False, "mandate has no allowed chains"
    if mandate.max_trade_usd <= 0:
        return False, "mandate max_trade_usd must be positive"
    if mandate.max_slippage_bps <= 0:
        return False, "mandate max_slippage_bps must be positive"
    if mandate.max_writes_per_hour <= 0:
        return False, "mandate max_writes_per_hour must be positive"
    if not mandate.signature:
        return False, "mandate missing signature"

    key = _secret(signing_key)
    if not key:
        return False, "HERMES_TRADER_MANDATE_SECRET is unavailable"

    expected = _signature(_canonical_payload(mandate), key)
    if not hmac.compare_digest(expected, mandate.signature):
        return False, "mandate signature invalid"

    wallet = (
        expected_wallet
        if expected_wallet is not None
        else os.environ.get("USER_ADDRESS", "")
    )
    wallet = str(wallet).strip().lower()
    if not wallet:
        return False, "USER_ADDRESS is unavailable"
    if wallet != mandate.wallet_address:
        return False, "mandate wallet does not match USER_ADDRESS"

    try:
        signed = datetime.fromisoformat(mandate.signed_at.replace("Z", "+00:00"))
        expires = datetime.fromisoformat(mandate.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False, "mandate timestamps must be ISO-8601"
    if signed.tzinfo is None or expires.tzinfo is None:
        return False, "mandate timestamps must include timezone"

    current = now or datetime.now(timezone.utc)
    if signed > current + timedelta(minutes=5):
        return False, "mandate signed_at is in the future"
    if expires <= signed:
        return False, "mandate expires_at must be after signed_at"
    if current >= expires:
        return False, "mandate expired"

    return True, ""


def load_mandate(path: Optional[Path | str] = None) -> Optional[Mandate]:
    target = Path(path) if path is not None else default_mandate_path()
    if not target.is_file():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mandate root must be an object")
    return Mandate.from_mapping(data)


def save_mandate(
    mandate: Mandate,
    path: Optional[Path | str] = None,
) -> Path:
    target = Path(path) if path is not None else default_mandate_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(mandate.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target
