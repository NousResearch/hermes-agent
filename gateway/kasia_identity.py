from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from gateway.kasia_config import load_kasia_settings, normalized_kasia_address_variants

DEFAULT_MAINNET_KNS_URL = "https://api.knsdomains.org/mainnet/api/v1"
DEFAULT_TESTNET_KNS_URL = "https://api.knsdomains.org/tn10/api/v1"

_KASIA_ADDRESS_PREFIXES = ("kaspa:", "kaspatest:", "kaspasim:")
_KNS_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\.kas$")
_BARE_ADDRESS_RE = re.compile(r"^[qp][a-z0-9]{5,}$")


def default_kasia_kns_url(network: str = "mainnet") -> str:
    normalized_network = str(network or "").strip().lower()
    return (
        DEFAULT_MAINNET_KNS_URL
        if normalized_network.startswith("mainnet")
        else DEFAULT_TESTNET_KNS_URL
    )


def normalize_kasia_kns_name(value: Any) -> Optional[str]:
    trimmed = str(value or "").strip().lower()
    if not trimmed:
        return None
    if "." not in trimmed and (
        trimmed.startswith(_KASIA_ADDRESS_PREFIXES)
        or _BARE_ADDRESS_RE.fullmatch(trimmed)
    ):
        return None
    full_name = trimmed if trimmed.endswith(".kas") else f"{trimmed}.kas"
    return full_name if _KNS_NAME_RE.fullmatch(full_name) else None


def _default_address_prefix(env: Optional[Mapping[str, str]] = None) -> str:
    settings = load_kasia_settings(env=env)
    normalized_network = str(settings.network or "").strip().lower()
    if normalized_network.startswith("mainnet"):
        return "kaspa:"
    if normalized_network.startswith("test") or normalized_network.startswith("tn"):
        return "kaspatest:"
    return "kaspasim:"


def canonicalize_kasia_address(
    value: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    trimmed = str(value or "").strip().lower()
    if not trimmed:
        return None
    if trimmed.startswith(_KASIA_ADDRESS_PREFIXES):
        return trimmed
    if _BARE_ADDRESS_RE.fullmatch(trimmed):
        return f"{_default_address_prefix(env)}{trimmed}"
    return None


def _fetch_json_default(url: str, *, timeout: float = 5.0) -> dict:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "HermesAgent/1.0",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _resolve_kns_base_url(env: Optional[Mapping[str, str]] = None) -> str:
    settings = load_kasia_settings(env=env)
    return str(settings.kns_url or default_kasia_kns_url(settings.network)).strip()


def resolve_kasia_kns_name(
    target: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[Callable[..., dict]] = None,
) -> Optional[str]:
    normalized_name = normalize_kasia_kns_name(target)
    if not normalized_name:
        return None

    base_url = _resolve_kns_base_url(env)
    if not base_url:
        return None

    fetch = fetch_json or _fetch_json_default
    try:
        response = fetch(
            f"{base_url.rstrip('/')}/{quote(normalized_name, safe='')}/owner"
        )
        owner = canonicalize_kasia_address(response.get("data", {}).get("owner"), env=env)
        asset = str(response.get("data", {}).get("asset") or "").strip().lower()
        if response.get("success") and owner and asset == normalized_name:
            return owner
    except (OSError, URLError, ValueError, TypeError, KeyError):
        return None
    return None


def lookup_kasia_primary_name(
    address: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[Callable[..., dict]] = None,
) -> Optional[str]:
    canonical_address = canonicalize_kasia_address(address, env=env)
    if not canonical_address:
        return None

    base_url = _resolve_kns_base_url(env)
    if not base_url:
        return None

    fetch = fetch_json or _fetch_json_default
    try:
        response = fetch(
            f"{base_url.rstrip('/')}/primary-name/{quote(canonical_address, safe='')}"
        )
        domain = response.get("data", {}).get("domain", {}).get("name")
        if response.get("success"):
            return normalize_kasia_kns_name(domain)
    except (OSError, URLError, ValueError, TypeError, KeyError):
        return None
    return None


def kasia_target_matches(
    target: Any,
    allowed_target: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    display_name: Optional[str] = None,
) -> bool:
    allowed_kns = normalize_kasia_kns_name(allowed_target)
    if allowed_kns:
        display_kns = normalize_kasia_kns_name(display_name)
        target_kns = normalize_kasia_kns_name(target)
        if allowed_kns and allowed_kns in {display_kns, target_kns}:
            return True

    if allowed_kns:
        resolved_address = resolve_kasia_kns_name(allowed_kns, env=env)
        if not resolved_address:
            return False
        allowed_variants = normalized_kasia_address_variants(resolved_address)
    else:
        allowed_variants = normalized_kasia_address_variants(
            canonicalize_kasia_address(allowed_target, env=env) or allowed_target
        )

    target_variants = normalized_kasia_address_variants(
        canonicalize_kasia_address(target, env=env) or target
    )
    return bool(target_variants and allowed_variants and target_variants & allowed_variants)


@dataclass(frozen=True, slots=True)
class KasiaIdentity:
    original_target: str
    canonical_address: Optional[str]
    kns_name: Optional[str]
    display_name: str
    identity_source: str

    def to_record(self) -> dict[str, Any]:
        return {
            "user_id": self.canonical_address or self.original_target,
            "user_name": self.display_name,
            "display_name": self.display_name,
            "canonical_address": self.canonical_address,
            "kns_name": self.kns_name,
            "identity_source": self.identity_source,
            "original_target": self.original_target,
        }


def resolve_kasia_identity(
    target: Any,
    *,
    display_name: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[Callable[..., dict]] = None,
) -> KasiaIdentity:
    original_target = str(target or "").strip()
    canonical_address = canonicalize_kasia_address(original_target, env=env)

    display_kns = normalize_kasia_kns_name(display_name)
    target_kns = normalize_kasia_kns_name(original_target)
    kns_name = display_kns or target_kns

    if target_kns:
        canonical_address = resolve_kasia_kns_name(
            target_kns,
            env=env,
            fetch_json=fetch_json,
        )
        kns_name = target_kns
        identity_source = "kns"
    else:
        identity_source = "address" if canonical_address else "unknown"
        if canonical_address and not kns_name:
            kns_name = lookup_kasia_primary_name(
                canonical_address,
                env=env,
                fetch_json=fetch_json,
            )

    resolved_display_name = (
        kns_name
        or str(display_name or "").strip()
        or canonical_address
        or original_target
    )
    if str(display_name or "").strip() and not normalize_kasia_kns_name(display_name):
        identity_source = "nickname"

    return KasiaIdentity(
        original_target=original_target,
        canonical_address=canonical_address,
        kns_name=kns_name,
        display_name=resolved_display_name,
        identity_source=identity_source,
    )
