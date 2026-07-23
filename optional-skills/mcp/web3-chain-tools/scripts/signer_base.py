"""Signing backends: dev-only env key vs hardware stub."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChainSigner(Protocol):
    """Sign EVM dict txs and opaque Solana message bytes."""

    def sign_evm_transaction(self, tx: dict[str, Any]) -> str:  # returns hex signed raw
        ...

    def sign_solana_bytes(self, message: bytes) -> bytes:
        ...


class HardwareWalletSigner:
    """Placeholder for Ledger / keystore hooks — not implemented in v1."""

    def sign_evm_transaction(self, tx: dict[str, Any]) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "HardwareWalletSigner: integrate via vendor SDK or remote signer; "
            "see SKILL.md extension section."
        )

    def sign_solana_bytes(self, message: bytes) -> bytes:  # noqa: ARG002
        raise NotImplementedError("HardwareWalletSigner: Solana path not implemented.")


class EnvPrivateKeySigner(ABC):
    """Load hex key from env var name in WEB3_EVM_PRIVATE_KEY_ENV (never from tool args)."""

    @abstractmethod
    def sign_evm_transaction(self, tx: dict[str, Any]) -> str:
        raise NotImplementedError


class EnvPrivateKeySignerEVM(EnvPrivateKeySigner):
    def __init__(self, env_name: str) -> None:
        if os.environ.get("WEB3_ALLOW_INSECURE_ENV_SIGNER", "").strip() not in ("1", "true", "yes"):
            raise RuntimeError(
                "Refusing env private key signer: set WEB3_ALLOW_INSECURE_ENV_SIGNER=1 (dev only)"
            )
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            raise RuntimeError(f"Missing {env_name} for env signer")
        self._key_bytes = bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)

    def sign_evm_transaction(self, tx: dict[str, Any]) -> str:
        from eth_account import Account

        acct = Account.from_key(self._key_bytes)
        signed = acct.sign_transaction(tx)  # type: ignore[arg-type]
        return signed.rawTransaction.hex()  # type: ignore[attr-defined]
