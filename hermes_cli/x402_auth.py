"""x402 upto-auth support for Hermes.

Matches the Opencode loader pattern:
- fetch router config once
- create a capped ERC-2612 permit
- cache the resulting payment header
- invalidate and refresh on 401/402 when the router says the session/cap is exhausted
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


USDC_ADDRESSES = {
    "eip155:8453": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "eip155:84532": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "eip155:1": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
}
DEFAULT_RPC_URLS = {
    "eip155:8453": "https://mainnet.base.org",
    "eip155:84532": "https://sepolia.base.org",
    "eip155:1": "https://cloudflare-eth.com",
}
DEFAULT_PERMIT_CAP_USDC = "2"
DEFAULT_VALIDITY_SECONDS = 3600
PRE_INVALIDATE_WINDOW_SECONDS = 60
PAYMENT_REQUIRED_HEADER = "PAYMENT-REQUIRED"
PAYMENT_SIGNATURE_HEADER = "PAYMENT-SIGNATURE"
PERMIT_TYPES = {
    "Permit": [
        {"name": "owner", "type": "address"},
        {"name": "spender", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "deadline", "type": "uint256"},
    ]
}
ERC2612_NONCES_ABI = [
    {
        "type": "function",
        "name": "nonces",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "nonce", "type": "uint256"}],
    }
]


@dataclass
class RouterConfig:
    network: str
    asset: str
    pay_to: str
    facilitator_signer: str
    token_name: str
    token_version: str
    payment_header: str = PAYMENT_SIGNATURE_HEADER


@dataclass
class CachedPermit:
    payment_sig: str
    deadline: int
    max_value: str
    nonce: str
    network: str
    asset: str
    pay_to: str


class PermitCache:
    def __init__(self) -> None:
        self._cache: dict[str, CachedPermit] = {}

    def get(self, network: str, asset: str, pay_to: str) -> Optional[CachedPermit]:
        key = self._build_key(network, asset, pay_to)
        permit = self._cache.get(key)
        if not permit:
            return None
        if self._is_near_deadline(permit.deadline):
            self._cache.pop(key, None)
            return None
        return permit

    def set(self, permit: CachedPermit) -> None:
        self._cache[self._build_key(permit.network, permit.asset, permit.pay_to)] = permit

    def invalidate(self, network: str, asset: str, pay_to: str) -> None:
        self._cache.pop(self._build_key(network, asset, pay_to), None)

    def _build_key(self, network: str, asset: str, pay_to: str) -> str:
        return f"{network.lower()}:{asset.lower()}:{pay_to.lower()}"

    def _is_near_deadline(self, deadline: int) -> bool:
        return deadline - int(time.time()) <= PRE_INVALIDATE_WINDOW_SECONDS


_cached_router_config: Optional[RouterConfig] = None
_cached_router_url: Optional[str] = None


def clear_router_config_cache() -> None:
    global _cached_router_config, _cached_router_url
    _cached_router_config = None
    _cached_router_url = None


def normalize_private_key(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.startswith("0X"):
        cleaned = "0x" + cleaned[2:]
    if not cleaned.startswith("0x"):
        cleaned = "0x" + cleaned
    if len(cleaned) != 66:
        return None
    try:
        int(cleaned[2:], 16)
    except ValueError:
        return None
    return cleaned


def normalize_token_name(token_name: str, asset: str) -> str:
    if token_name.strip() == "USDC" and asset.lower() in {a.lower() for a in USDC_ADDRESSES.values()}:
        return "USD Coin"
    return token_name.strip()


def usdc_to_units(amount: str) -> str:
    return str(int(float(amount) * 1_000_000))


def decode_payment_required_header(value: str) -> Optional[dict]:
    try:
        return json.loads(base64.b64decode(value).decode("utf-8"))
    except Exception:
        return None


def get_requirement_pay_to(requirement: Optional[dict]) -> Optional[str]:
    if not requirement:
        return None
    return requirement.get("payTo") or requirement.get("pay_to")


def get_requirement_max_amount_required(requirement: Optional[dict]) -> Optional[str]:
    if not requirement:
        return None
    extra = requirement.get("extra") or {}
    return (
        extra.get("maxAmountRequired")
        or extra.get("max_amount_required")
        or extra.get("maxAmount")
        or extra.get("max_amount")
    )


def apply_payment_requirement(config: RouterConfig, requirement: Optional[dict]) -> RouterConfig:
    if not requirement:
        return config
    pay_to = get_requirement_pay_to(requirement) or config.pay_to
    extra = requirement.get("extra") or {}
    asset = requirement.get("asset") or config.asset
    token_name = normalize_token_name(str(extra.get("name") or config.token_name), asset)
    return RouterConfig(
        network=str(requirement.get("network") or config.network),
        asset=asset,
        pay_to=pay_to,
        facilitator_signer=pay_to or config.facilitator_signer,
        token_name=token_name,
        token_version=str(extra.get("version") or config.token_version),
        payment_header=config.payment_header,
    )


def parse_error_response(data: Any) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("code"), str) or isinstance(data.get("error"), str):
        return {
            "code": data.get("code") if isinstance(data.get("code"), str) else None,
            "error": data.get("error") if isinstance(data.get("error"), str) else data.get("message"),
            "message": data.get("message") if isinstance(data.get("message"), str) else None,
        }
    nested = data.get("error")
    if isinstance(nested, dict):
        return {
            "code": nested.get("code") if isinstance(nested.get("code"), str) else nested.get("type"),
            "error": nested.get("message") if isinstance(nested.get("message"), str) else nested.get("error"),
            "message": nested.get("message") if isinstance(nested.get("message"), str) else None,
        }
    return None


def should_invalidate_permit(error: Optional[dict]) -> bool:
    if not error:
        return False
    text = f"{error.get('error', '')} {error.get('message', '')}".lower()
    return error.get("code") in {"cap_exhausted", "session_closed"} or "cap exhausted" in text or "session closed" in text


def parse_router_config_response(data: dict) -> RouterConfig:
    network = (data.get("networks") or [{}])[0]
    eip712 = data.get("eip712_config") or {}
    asset = ((network.get("asset") or {}).get("address")) or USDC_ADDRESSES["eip155:8453"]
    token_name = normalize_token_name(str(eip712.get("domain_name") or "USD Coin"), asset)
    return RouterConfig(
        network=str(network.get("network_id") or "eip155:8453"),
        asset=asset,
        pay_to=str(network.get("pay_to") or ""),
        facilitator_signer=str(network.get("pay_to") or ""),
        token_name=token_name,
        token_version=str(eip712.get("domain_version") or "2"),
    )


def fetch_router_config(router_url: str) -> RouterConfig:
    global _cached_router_config, _cached_router_url
    if _cached_router_config and _cached_router_url == router_url:
        return _cached_router_config

    import requests

    response = requests.get(f"{router_url}/v1/config", timeout=10)
    response.raise_for_status()
    config = parse_router_config_response(response.json())
    _cached_router_config = config
    _cached_router_url = router_url
    return config


def get_default_config(network: str) -> RouterConfig:
    return RouterConfig(
        network=network,
        asset=USDC_ADDRESSES.get(network, USDC_ADDRESSES["eip155:8453"]),
        pay_to="",
        facilitator_signer="",
        token_name="USD Coin",
        token_version="2",
    )


def _router_url_from_base(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[:-3].rstrip("/")
    return cleaned


def _get_web3(network: str, rpc_url: Optional[str]) -> Any:
    from web3 import Web3

    effective_rpc = (rpc_url or "").strip() or os.getenv("X402_RPC_URL", "").strip() or DEFAULT_RPC_URLS.get(network)
    if not effective_rpc:
        raise RuntimeError(f"x402 network {network} requires X402_RPC_URL.")
    return Web3(Web3.HTTPProvider(effective_rpc))


def _fetch_permit_nonce(network: str, token: str, owner: str, rpc_url: Optional[str]) -> int:
    web3 = _get_web3(network, rpc_url)
    contract = web3.eth.contract(address=web3.to_checksum_address(token), abi=ERC2612_NONCES_ABI)
    return int(contract.functions.nonces(web3.to_checksum_address(owner)).call())


def sign_permit(account: Any, config: RouterConfig, permit_cap: str, rpc_url: Optional[str]) -> dict:
    chain_id = int(config.network.split(":")[1])
    deadline = int(time.time()) + DEFAULT_VALIDITY_SECONDS
    nonce = str(_fetch_permit_nonce(config.network, config.asset, account.address, rpc_url))
    message = {
        "owner": account.address,
        "spender": config.facilitator_signer,
        "value": int(permit_cap),
        "nonce": int(nonce),
        "deadline": int(deadline),
    }
    signed = account.sign_typed_data(
        domain_data={
            "name": config.token_name,
            "version": config.token_version,
            "chainId": chain_id,
            "verifyingContract": config.asset,
        },
        message_types=PERMIT_TYPES,
        message_data=message,
    )
    return {
        "signature": "0x" + signed.signature.hex(),
        "nonce": nonce,
        "deadline": str(deadline),
    }


def create_cached_permit(account: Any, config: RouterConfig, permit_cap: str, rpc_url: Optional[str]) -> CachedPermit:
    signed = sign_permit(account, config, permit_cap, rpc_url)
    payload = {
        "x402Version": 2,
        "accepted": {
            "scheme": "upto",
            "network": config.network,
            "asset": config.asset,
            "payTo": config.pay_to,
            "extra": {
                "name": config.token_name,
                "version": config.token_version,
            },
        },
        "payload": {
            "authorization": {
                "from": account.address,
                "to": config.facilitator_signer,
                "value": permit_cap,
                "validBefore": signed["deadline"],
                "nonce": signed["nonce"],
            },
            "signature": signed["signature"],
        },
    }
    payment_sig = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("utf-8")
    return CachedPermit(
        payment_sig=payment_sig,
        deadline=int(signed["deadline"]),
        max_value=permit_cap,
        nonce=signed["nonce"],
        network=config.network,
        asset=config.asset,
        pay_to=config.pay_to,
    )


def create_x402_request_headers_resolver(
    *,
    private_key: str,
    base_url: str,
    preferred_network: str | None = None,
    rpc_url: str | None = None,
    permit_cap_units: str | None = None,
):
    from eth_account import Account

    normalized_private_key = normalize_private_key(private_key)
    if not normalized_private_key:
        raise RuntimeError("Invalid x402 private key. Expected 32-byte hex, with or without 0x prefix.")

    account = Account.from_key(normalized_private_key)
    router_url = _router_url_from_base(base_url)
    default_network = preferred_network or os.getenv("X402_NETWORK", "").strip() or "eip155:8453"
    default_permit_cap = permit_cap_units or usdc_to_units(os.getenv("X402_PERMIT_CAP_USDC", "").strip() or DEFAULT_PERMIT_CAP_USDC)
    permit_cache = PermitCache()
    router_config: Optional[RouterConfig] = None
    last_permit: Optional[CachedPermit] = None

    def _get_or_create_permit(cap_override: Optional[str] = None) -> CachedPermit:
        nonlocal last_permit, router_config
        if router_config is None:
            raise RuntimeError("x402 router config is not initialized.")
        cached = permit_cache.get(router_config.network, router_config.asset, router_config.pay_to)
        if cached and (not cap_override or cached.max_value == cap_override):
            last_permit = cached
            return cached
        if cached:
            permit_cache.invalidate(router_config.network, router_config.asset, router_config.pay_to)
        fresh = create_cached_permit(account, router_config, cap_override or default_permit_cap, rpc_url)
        permit_cache.set(fresh)
        last_permit = fresh
        return fresh

    def _resolver(
        *,
        force_refresh: bool = False,
        api_kwargs: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, str]:
        nonlocal router_config
        if router_config is None:
            try:
                router_config = fetch_router_config(router_url)
            except Exception:
                router_config = get_default_config(default_network)

        if error is not None:
            response = getattr(error, "response", None)
            status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
            if status_code not in {401, 402}:
                return {}

            error_payload = None
            if response is not None:
                try:
                    error_payload = parse_error_response(response.json())
                except Exception:
                    error_payload = None

            requirement = None
            if response is not None and getattr(response, "headers", None):
                payment_required = response.headers.get(PAYMENT_REQUIRED_HEADER)
                decoded = decode_payment_required_header(payment_required) if payment_required else None
                if isinstance(decoded, dict):
                    accepts = decoded.get("accepts") or []
                    if accepts:
                        requirement = accepts[0]

            previous_config = router_config
            if requirement:
                router_config = apply_payment_requirement(router_config, requirement)

            if error_payload and should_invalidate_permit(error_payload):
                permit_cache.invalidate(previous_config.network, previous_config.asset, previous_config.pay_to)
                if (
                    previous_config.network != router_config.network
                    or previous_config.asset != router_config.asset
                    or previous_config.pay_to != router_config.pay_to
                ):
                    permit_cache.invalidate(router_config.network, router_config.asset, router_config.pay_to)
                refreshed = _get_or_create_permit(get_requirement_max_amount_required(requirement))
                return {router_config.payment_header: refreshed.payment_sig}

            return {}

        if force_refresh:
            permit_cache.invalidate(router_config.network, router_config.asset, router_config.pay_to)

        permit = _get_or_create_permit()
        return {router_config.payment_header: permit.payment_sig}

    return _resolver
