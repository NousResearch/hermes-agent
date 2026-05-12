#!/usr/bin/env python3
"""
Web3 Permission Audit CLI for Hermes Agent.

Read-only cross-chain wallet permission checks for:
- Solana SPL token delegates, close authorities, frozen accounts, and mint/freeze authorities
- XRPL trust line controls, rippling flags, freeze flags, and issuer exposure
- Base/EVM ERC-20 allowances for explicit owner/spender pairs

The tool never asks for private keys, never signs transactions, and never submits
state-changing RPC calls. It is intentionally conservative about claims: where a
chain cannot enumerate a permission class via plain RPC, the report says so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

VERSION = "0.1.0"
USER_AGENT = "HermesAgent-Web3PermissionAudit/0.1"

DEFAULT_BASE_RPC_URL = os.environ.get(
    "WEB3_AUDIT_BASE_RPC_URL",
    os.environ.get("BASE_RPC_URL", "https://mainnet.base.org"),
)
DEFAULT_SOLANA_RPC_URL = os.environ.get(
    "WEB3_AUDIT_SOLANA_RPC_URL",
    os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
)
DEFAULT_XRPL_RPC_URL = os.environ.get(
    "WEB3_AUDIT_XRPL_RPC_URL",
    os.environ.get("XRPL_RPC_URL", "https://s1.ripple.com:51234/"),
)

ALLOW_PRIVATE_RPC = os.environ.get("WEB3_AUDIT_ALLOW_PRIVATE_RPC", "").lower() in {
    "1",
    "true",
    "yes",
}

SOLANA_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SOLANA_TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
LAMPORTS_PER_SOL = 1_000_000_000
WEI_PER_ETH = 10**18
UINT256_MAX = 2**256 - 1

EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
SOLANA_ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
XRPL_CLASSIC_ADDRESS_RE = re.compile(r"^r[1-9A-HJ-NP-Za-km-z]{24,34}$")

# ERC-20 function selectors.
SEL_ALLOWANCE = "dd62ed3e"
SEL_BALANCE_OF = "70a08231"
SEL_SYMBOL = "95d89b41"
SEL_DECIMALS = "313ce567"

# Known Base token contracts. This is not discovery; it is a bounded default
# token set so users can audit common assets without an indexer.
BASE_KNOWN_TOKENS: dict[str, tuple[str, int]] = {
    "0x4200000000000000000000000000000000000006": ("WETH", 18),
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC", 6),
    "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22": ("cbETH", 18),
    "0x940181a94a35a4569e4529a3cdfb74e38fd98631": ("AERO", 18),
    "0x4ed4e862860bed51a9570b96d89af5e1b0efefed": ("DEGEN", 18),
    "0xac1bd2486aaf3b5c0fc3fd868558b082a531b2b4": ("TOSHI", 18),
    "0x532f27101965dd16442e59d40670faf5ebb142e4": ("BRETT", 18),
    "0xa88594d404727625a9437c3f886c7643872296ae": ("WELL", 18),
    "0xc1cba3fcea344f92d9239c08c0568f6f2f0ee452": ("wstETH", 18),
    "0xb6fe221fe9eef5aba221c348ba20a1bf5e73624c": ("rETH", 18),
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": ("cbBTC", 8),
}

SPENDER_ALIASES: dict[str, str] = {
    # Uniswap Permit2 is intentionally the only built-in spender alias because
    # it is widely reused across EVM chains. Other dapp router addresses change
    # more often and should be provided explicitly by the user.
    "permit2": "0x000000000022d473030f116ddee9f6b43ac78ba3",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False))


def normalize_evm_address(address: str) -> str:
    if not EVM_ADDRESS_RE.match(address):
        raise ValueError(f"Invalid EVM address: {address}")
    return address.lower()


def validate_solana_address(address: str) -> None:
    if not SOLANA_ADDRESS_RE.match(address):
        raise ValueError(f"Invalid Solana address: {address}")


def validate_xrpl_address(address: str) -> None:
    if not XRPL_CLASSIC_ADDRESS_RE.match(address):
        raise ValueError(f"Invalid XRPL classic address: {address}")


def validate_rpc_url(url: str, allow_private: bool = False) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("RPC URL must use http or https")
    if not parsed.hostname:
        raise ValueError("RPC URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("RPC URL must not include credentials")

    host = parsed.hostname.lower()
    if not allow_private:
        blocked = host in {"localhost"} or host.endswith(".local")
        try:
            ip = ipaddress.ip_address(host)
            blocked = blocked or ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            pass
        if blocked:
            raise ValueError(
                "RPC URL points to a local/private host. Use --allow-private-rpc "
                "or WEB3_AUDIT_ALLOW_PRIVATE_RPC=1 only when you intentionally "
                "want to query a trusted private node."
            )
    return url


def http_post_json(url: str, payload: Any, timeout: int = 20, retries: int = 2) -> Any:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"RPC HTTP error {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RPC connection error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("RPC returned invalid JSON") from exc
    raise RuntimeError("RPC request failed after retries")


def json_rpc(url: str, method: str, params: list[Any] | dict[str, Any] | None = None) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
    body = http_post_json(url, payload)
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"RPC error from {method}: {body['error']}")
    return body.get("result") if isinstance(body, dict) else None


def make_finding(
    *,
    severity: str,
    chain: str,
    category: str,
    title: str,
    subject: str,
    grantee: str | None,
    evidence: dict[str, Any],
    impact: str,
    action: str,
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "chain": chain,
        "category": category,
        "title": title,
        "subject": subject,
        "grantee": grantee,
        "evidence": evidence,
        "impact": impact,
        "suggested_action": action,
        "confidence": confidence,
    }


def severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.get("severity", "info")] = counts.get(finding.get("severity", "info"), 0) + 1
    return counts


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda item: (
            SEVERITY_ORDER.get(item.get("severity", "info"), 99),
            item.get("chain", ""),
            item.get("category", ""),
            item.get("subject", ""),
        ),
    )


def sample_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        return items
    if limit == 0:
        return []
    return items[:limit]


def build_report(
    *,
    command: str,
    chain_reports: dict[str, Any],
    findings: list[dict[str, Any]],
    limitations: list[str],
) -> dict[str, Any]:
    ordered_findings = sort_findings(findings)
    counts = severity_counts(ordered_findings)
    actionable = sum(counts[sev] for sev in ("critical", "high", "medium"))
    return {
        "tool": "web3-permission-audit",
        "version": VERSION,
        "generated_at": utc_now(),
        "command": command,
        "summary": {
            "finding_count": len(ordered_findings),
            "actionable_findings": actionable,
            "severity_counts": counts,
        },
        "findings": ordered_findings,
        "chain_reports": chain_reports,
        "limitations": limitations,
        "safety_model": {
            "read_only": True,
            "requires_private_keys": False,
            "signs_transactions": False,
            "submits_transactions": False,
        },
    }


def decode_uint(hex_data: str | None) -> int:
    if not hex_data or hex_data == "0x":
        return 0
    return int(hex_data[2:] if hex_data.startswith("0x") else hex_data, 16)


def encode_evm_address(address: str) -> str:
    return normalize_evm_address(address).replace("0x", "").zfill(64)


def decode_abi_string(hex_data: str | None) -> str:
    if not hex_data or hex_data == "0x":
        return ""
    data = hex_data[2:] if hex_data.startswith("0x") else hex_data
    try:
        if len(data) == 64:
            raw = bytes.fromhex(data).rstrip(b"\x00")
            return raw.decode("utf-8", errors="ignore").strip()
        if len(data) < 128:
            return ""
        length = int(data[64:128], 16)
        if length <= 0 or length > 256:
            return ""
        raw = bytes.fromhex(data[128 : 128 + length * 2])
        return raw.decode("utf-8", errors="ignore").strip()
    except ValueError:
        return ""


def evm_call(rpc_url: str, to: str, selector: str, args: str = "") -> str | None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": normalize_evm_address(to), "data": "0x" + selector + args}, "latest"],
    }
    body = http_post_json(rpc_url, payload)
    if isinstance(body, dict) and body.get("error"):
        return None
    return body.get("result") if isinstance(body, dict) else None


def human_amount(raw: int, decimals: int) -> float | str:
    if decimals < 0 or decimals > 36:
        return str(raw)
    value = raw / (10**decimals)
    if value == 0:
        return 0.0
    if value < 0.000001:
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return round(value, min(decimals, 6))


def resolve_spender(value: str) -> str:
    alias = value.lower()
    if alias in SPENDER_ALIASES:
        return SPENDER_ALIASES[alias]
    return normalize_evm_address(value)


def audit_base_allowances(
    *,
    owner: str,
    spenders: list[str],
    tokens: list[str],
    rpc_url: str,
    allow_private_rpc: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rpc_url = validate_rpc_url(rpc_url, allow_private_rpc)
    owner = normalize_evm_address(owner)
    normalized_spenders = [resolve_spender(spender) for spender in spenders]
    normalized_tokens = [normalize_evm_address(token) for token in tokens]

    findings: list[dict[str, Any]] = []
    allowances: list[dict[str, Any]] = []

    for token in normalized_tokens:
        known = BASE_KNOWN_TOKENS.get(token)
        symbol = known[0] if known else ""
        decimals = known[1] if known else 18

        if not known:
            decimals_raw = evm_call(rpc_url, token, SEL_DECIMALS)
            decimals = decode_uint(decimals_raw) if decimals_raw else 18
            symbol = decode_abi_string(evm_call(rpc_url, token, SEL_SYMBOL)) or token[:10]

        balance_raw = decode_uint(evm_call(rpc_url, token, SEL_BALANCE_OF, encode_evm_address(owner)))
        for spender in normalized_spenders:
            allowance_raw = decode_uint(
                evm_call(
                    rpc_url,
                    token,
                    SEL_ALLOWANCE,
                    encode_evm_address(owner) + encode_evm_address(spender),
                )
            )
            entry = {
                "chain": "base",
                "token": token,
                "symbol": symbol,
                "owner": owner,
                "spender": spender,
                "allowance_raw": str(allowance_raw),
                "allowance": human_amount(allowance_raw, decimals),
                "owner_balance_raw": str(balance_raw),
                "owner_balance": human_amount(balance_raw, decimals),
                "decimals": decimals,
            }
            allowances.append(entry)

            if allowance_raw == 0:
                continue

            if allowance_raw >= int(UINT256_MAX * 0.9):
                severity = "critical" if balance_raw > 0 else "high"
                title = f"Unlimited ERC-20 allowance for {symbol}"
                impact = (
                    "The spender can transfer any current or future balance of this token "
                    "from the owner until the allowance is revoked or reduced."
                )
            elif balance_raw > 0 and allowance_raw >= balance_raw:
                severity = "high"
                title = f"Allowance covers current {symbol} balance"
                impact = "The spender can transfer the owner's current token balance."
            else:
                severity = "medium"
                title = f"Non-zero ERC-20 allowance for {symbol}"
                impact = "The spender can transfer tokens up to the approved allowance."

            findings.append(
                make_finding(
                    severity=severity,
                    chain="base",
                    category="evm_erc20_allowance",
                    title=title,
                    subject=f"{owner}:{token}",
                    grantee=spender,
                    evidence=entry,
                    impact=impact,
                    action=(
                        "If this approval is not actively needed, revoke it in a trusted "
                        "wallet or token approval manager. For active dapps, prefer exact "
                        "amount approvals over unlimited approvals."
                    ),
                )
            )

    report = {
        "owner": owner,
        "spenders": normalized_spenders,
        "tokens_checked": len(normalized_tokens),
        "allowances": allowances,
        "coverage": "targeted_owner_spender_token_matrix",
    }
    limitations = [
        "Plain EVM JSON-RPC cannot enumerate all historical ERC-20 approvals for an address without scanning logs or using an indexer. This command checks only the provided spender(s) against provided tokens, or the bundled known Base token set.",
        "ERC-721, ERC-1155, Permit signatures, and app-specific session keys are not discovered in this first read-only version.",
    ]
    return report, findings, limitations


def solana_rpc(rpc_url: str, method: str, params: list[Any] | None = None) -> Any:
    return json_rpc(rpc_url, method, params or [])


def parse_solana_token_accounts(result: Any, program_id: str) -> list[dict[str, Any]]:
    accounts: list[dict[str, Any]] = []
    for item in (result or {}).get("value", []) or []:
        account_pubkey = item.get("pubkey")
        account_data = item.get("account", {}).get("data", {})
        parsed = account_data.get("parsed", {}) if isinstance(account_data, dict) else {}
        info = parsed.get("info", {})
        token_amount = info.get("tokenAmount", {})
        delegated_amount = info.get("delegatedAmount") or {}
        delegated_raw = 0
        if isinstance(delegated_amount, dict):
            delegated_raw = int(delegated_amount.get("amount") or 0)
        elif delegated_amount:
            delegated_raw = int(delegated_amount)
        accounts.append(
            {
                "program_id": program_id,
                "token_account": account_pubkey,
                "mint": info.get("mint"),
                "owner": info.get("owner"),
                "state": info.get("state"),
                "delegate": info.get("delegate"),
                "delegated_amount_raw": delegated_raw,
                "delegated_amount": delegated_amount,
                "close_authority": info.get("closeAuthority"),
                "amount_raw": int(token_amount.get("amount") or 0),
                "amount": token_amount.get("uiAmountString") or token_amount.get("uiAmount"),
                "decimals": token_amount.get("decimals"),
                "is_native": info.get("isNative", False),
            }
        )
    return accounts


def fetch_solana_mint_info(rpc_url: str, mint: str) -> dict[str, Any]:
    try:
        result = solana_rpc(rpc_url, "getAccountInfo", [mint, {"encoding": "jsonParsed"}])
    except RuntimeError as exc:
        return {"mint": mint, "error": str(exc)}
    value = (result or {}).get("value")
    if not value:
        return {"mint": mint, "error": "mint account not found"}
    data = value.get("data", {})
    parsed = data.get("parsed", {}) if isinstance(data, dict) else {}
    info = parsed.get("info", {})
    return {
        "mint": mint,
        "decimals": info.get("decimals"),
        "supply_raw": info.get("supply"),
        "mint_authority": info.get("mintAuthority"),
        "freeze_authority": info.get("freezeAuthority"),
    }


def audit_solana(
    *,
    address: str,
    rpc_url: str,
    allow_private_rpc: bool,
    mint_limit: int,
    detail_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    validate_solana_address(address)
    rpc_url = validate_rpc_url(rpc_url, allow_private_rpc)
    findings: list[dict[str, Any]] = []
    token_accounts: list[dict[str, Any]] = []
    rpc_errors: list[str] = []

    for program_id in (SOLANA_TOKEN_PROGRAM, SOLANA_TOKEN_2022_PROGRAM):
        try:
            result = solana_rpc(
                rpc_url,
                "getTokenAccountsByOwner",
                [address, {"programId": program_id}, {"encoding": "jsonParsed"}],
            )
            token_accounts.extend(parse_solana_token_accounts(result, program_id))
        except RuntimeError as exc:
            rpc_errors.append(f"{program_id}: {exc}")

    mint_infos: dict[str, dict[str, Any]] = {}
    unique_mints = [mint for mint in dict.fromkeys(a.get("mint") for a in token_accounts) if mint]
    for mint in unique_mints[:mint_limit]:
        mint_infos[mint] = fetch_solana_mint_info(rpc_url, mint)

    for account in token_accounts:
        token_account = account.get("token_account") or "unknown"
        mint = account.get("mint") or "unknown"
        delegate = account.get("delegate")
        amount_raw = int(account.get("amount_raw") or 0)
        delegated_raw = int(account.get("delegated_amount_raw") or 0)

        if delegate and delegated_raw > 0:
            severity = "critical" if amount_raw > 0 and delegated_raw >= amount_raw else "high"
            findings.append(
                make_finding(
                    severity=severity,
                    chain="solana",
                    category="spl_token_delegate",
                    title="SPL token delegate can transfer wallet tokens",
                    subject=token_account,
                    grantee=delegate,
                    evidence=account,
                    impact=(
                        "A delegated account can transfer tokens from this token account "
                        "up to the delegated amount without another wallet signature."
                    ),
                    action=(
                        "If the delegate is not expected, revoke the SPL token delegate "
                        "from a trusted wallet or SPL token tool."
                    ),
                )
            )

        close_authority = account.get("close_authority")
        if close_authority and close_authority != address:
            findings.append(
                make_finding(
                    severity="medium",
                    chain="solana",
                    category="spl_close_authority",
                    title="Token account has an external close authority",
                    subject=token_account,
                    grantee=close_authority,
                    evidence=account,
                    impact=(
                        "An external close authority can close the token account when it is "
                        "eligible to be closed, which can surprise wallet recovery and asset "
                        "inventory workflows."
                    ),
                    action="Confirm the close authority is intentional; otherwise rotate it back to the wallet owner.",
                )
            )

        if account.get("state") == "frozen":
            findings.append(
                make_finding(
                    severity="high",
                    chain="solana",
                    category="spl_frozen_account",
                    title="Token account is frozen",
                    subject=token_account,
                    grantee=None,
                    evidence=account,
                    impact="The account cannot freely transfer this token while frozen.",
                    action="Identify the token mint freeze authority and treat the asset as issuer-controlled.",
                )
            )

    for mint, info in mint_infos.items():
        if info.get("freeze_authority"):
            findings.append(
                make_finding(
                    severity="medium",
                    chain="solana",
                    category="spl_mint_freeze_authority",
                    title="Token mint has an active freeze authority",
                    subject=mint,
                    grantee=info.get("freeze_authority"),
                    evidence=info,
                    impact="The issuer or controller can freeze token accounts for this mint.",
                    action="For high-value holdings, verify that the freeze authority is expected or has been revoked.",
                )
            )
        if info.get("mint_authority"):
            findings.append(
                make_finding(
                    severity="low",
                    chain="solana",
                    category="spl_mint_authority",
                    title="Token mint has an active mint authority",
                    subject=mint,
                    grantee=info.get("mint_authority"),
                    evidence=info,
                    impact="The token supply may be increased by the mint authority.",
                    action="Check whether active mint authority is normal for this asset before treating it as scarce.",
                )
            )

    token_account_sample = sample_items(token_accounts, detail_limit)
    report = {
        "address": address,
        "token_accounts_checked": len(token_accounts),
        "unique_mints": len(unique_mints),
        "mints_inspected": len(mint_infos),
        "token_accounts_sample": token_account_sample,
        "token_accounts_sample_size": len(token_account_sample),
        "token_accounts_omitted": max(0, len(token_accounts) - len(token_account_sample)),
        "mint_authorities": mint_infos,
        "rpc_errors": rpc_errors,
    }
    limitations = [
        "Solana compressed NFTs, program-specific authorities, multisig owners, durable nonce authorities, and DeFi positions are outside this first token-permission scan.",
        "Token-2022 extensions are surfaced only where the public RPC returns jsonParsed fields used by this script.",
    ]
    return report, findings, limitations


def xrpl_amount(value: str | int | float | None) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def decode_xrpl_currency(code: str) -> str:
    if len(code) == 40 and all(ch in "0123456789ABCDEFabcdef" for ch in code):
        try:
            raw = bytes.fromhex(code).rstrip(b"\x00")
            text = raw.decode("ascii", errors="ignore").strip()
            return text or code
        except ValueError:
            return code
    return code


def audit_xrpl(
    *,
    address: str,
    rpc_url: str,
    allow_private_rpc: bool,
    limit: int,
    max_pages: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    validate_xrpl_address(address)
    rpc_url = validate_rpc_url(rpc_url, allow_private_rpc)
    findings: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    marker: Any = None

    for _page in range(max_pages):
        request: dict[str, Any] = {
            "account": address,
            "ledger_index": "validated",
            "limit": limit,
        }
        if marker:
            request["marker"] = marker
        result = json_rpc(rpc_url, "account_lines", [request])
        if not result:
            break
        lines.extend(result.get("lines", []) or [])
        marker = result.get("marker")
        if not marker:
            break

    normalized_lines: list[dict[str, Any]] = []
    for line in lines:
        currency = decode_xrpl_currency(str(line.get("currency", "")))
        balance = xrpl_amount(line.get("balance"))
        limit_self = xrpl_amount(line.get("limit"))
        limit_peer = xrpl_amount(line.get("limit_peer"))
        normalized = {
            "account": line.get("account"),
            "currency": currency,
            "currency_raw": line.get("currency"),
            "balance": line.get("balance"),
            "limit": line.get("limit"),
            "limit_peer": line.get("limit_peer"),
            "no_ripple": line.get("no_ripple"),
            "no_ripple_peer": line.get("no_ripple_peer"),
            "authorized": line.get("authorized"),
            "peer_authorized": line.get("peer_authorized"),
            "freeze": line.get("freeze"),
            "freeze_peer": line.get("freeze_peer"),
            "quality_in": line.get("quality_in"),
            "quality_out": line.get("quality_out"),
        }
        normalized_lines.append(normalized)

        subject = f"{address}:{currency}:{line.get('account')}"
        if line.get("freeze") or line.get("freeze_peer"):
            findings.append(
                make_finding(
                    severity="high",
                    chain="xrpl",
                    category="xrpl_trustline_freeze",
                    title=f"XRPL trust line freeze flag present for {currency}",
                    subject=subject,
                    grantee=line.get("account"),
                    evidence=normalized,
                    impact="A freeze flag can restrict movement of the issued asset on this trust line.",
                    action="Treat the asset as issuer-controlled and verify whether the freeze is expected.",
                )
            )

        if balance < 0:
            findings.append(
                make_finding(
                    severity="medium",
                    chain="xrpl",
                    category="xrpl_negative_trustline_balance",
                    title=f"Negative XRPL trust line balance for {currency}",
                    subject=subject,
                    grantee=line.get("account"),
                    evidence=normalized,
                    impact="A negative trust line balance means the account owes issued value on this line.",
                    action="Review whether the outstanding issued balance is intentional.",
                )
            )

        if limit_self > 0 and not bool(line.get("no_ripple")):
            findings.append(
                make_finding(
                    severity="medium",
                    chain="xrpl",
                    category="xrpl_rippling_enabled",
                    title=f"Rippling appears enabled on {currency} trust line",
                    subject=subject,
                    grantee=line.get("account"),
                    evidence=normalized,
                    impact=(
                        "If rippling is enabled, path payments may use this trust line in "
                        "ways that are hard for non-specialists to reason about."
                    ),
                    action="If this is a regular user wallet, consider enabling NoRipple where appropriate.",
                )
            )

        if balance > 0:
            findings.append(
                make_finding(
                    severity="low",
                    chain="xrpl",
                    category="xrpl_issuer_exposure",
                    title=f"Issued asset exposure: {currency}",
                    subject=subject,
                    grantee=line.get("account"),
                    evidence=normalized,
                    impact="The wallet holds an issued asset whose value depends on the issuer/counterparty.",
                    action="For material balances, verify issuer reputation, freeze policy, and redemption assumptions.",
                )
            )

        if limit_peer > 0:
            findings.append(
                make_finding(
                    severity="info",
                    chain="xrpl",
                    category="xrpl_peer_trust_limit",
                    title=f"Peer has extended trust limit on {currency}",
                    subject=subject,
                    grantee=line.get("account"),
                    evidence=normalized,
                    impact="This is usually informational, but it can help explain bilateral trust-line behavior.",
                    action="No action needed unless this relationship is unexpected.",
                )
            )

    report = {
        "address": address,
        "trust_lines_checked": len(normalized_lines),
        "trust_lines": normalized_lines,
        "pagination_complete": marker is None,
        "max_pages": max_pages,
    }
    limitations = [
        "XRPL trust lines do not behave like ERC-20 allowances; the report focuses on issuer exposure, rippling, freeze flags, limits, and negative balances rather than token-spender approvals.",
        "This command does not submit AccountSet, TrustSet, or other remediation transactions.",
    ]
    return report, findings, limitations


def default_base_tokens() -> list[str]:
    return list(BASE_KNOWN_TOKENS.keys())


def command_base(args: argparse.Namespace) -> None:
    tokens = [normalize_evm_address(t) for t in (args.token or default_base_tokens())]
    spenders = args.spender or []
    if args.spender_alias:
        spenders.extend(args.spender_alias)
    if not spenders:
        raise SystemExit("At least one --spender or --spender-alias is required for evm-allowance")
    chain_report, findings, limitations = audit_base_allowances(
        owner=args.owner,
        spenders=spenders,
        tokens=tokens,
        rpc_url=args.rpc_url,
        allow_private_rpc=args.allow_private_rpc,
    )
    print_json(
        build_report(
            command="evm-allowance",
            chain_reports={"base": chain_report},
            findings=findings,
            limitations=limitations,
        )
    )


def command_solana(args: argparse.Namespace) -> None:
    chain_report, findings, limitations = audit_solana(
        address=args.address,
        rpc_url=args.rpc_url,
        allow_private_rpc=args.allow_private_rpc,
        mint_limit=args.mint_limit,
        detail_limit=args.detail_limit,
    )
    print_json(
        build_report(
            command="solana",
            chain_reports={"solana": chain_report},
            findings=findings,
            limitations=limitations,
        )
    )


def command_xrpl(args: argparse.Namespace) -> None:
    chain_report, findings, limitations = audit_xrpl(
        address=args.address,
        rpc_url=args.rpc_url,
        allow_private_rpc=args.allow_private_rpc,
        limit=args.limit,
        max_pages=args.max_pages,
    )
    print_json(
        build_report(
            command="xrpl",
            chain_reports={"xrpl": chain_report},
            findings=findings,
            limitations=limitations,
        )
    )


def command_audit(args: argparse.Namespace) -> None:
    chain_reports: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    limitations: list[str] = []
    errors: dict[str, str] = {}

    if args.solana:
        try:
            report, chain_findings, chain_limitations = audit_solana(
                address=args.solana,
                rpc_url=args.solana_rpc_url,
                allow_private_rpc=args.allow_private_rpc,
                mint_limit=args.mint_limit,
                detail_limit=args.detail_limit,
            )
            chain_reports["solana"] = report
            findings.extend(chain_findings)
            limitations.extend(chain_limitations)
        except (RuntimeError, ValueError) as exc:
            errors["solana"] = str(exc)

    if args.xrpl:
        try:
            report, chain_findings, chain_limitations = audit_xrpl(
                address=args.xrpl,
                rpc_url=args.xrpl_rpc_url,
                allow_private_rpc=args.allow_private_rpc,
                limit=args.xrpl_limit,
                max_pages=args.xrpl_max_pages,
            )
            chain_reports["xrpl"] = report
            findings.extend(chain_findings)
            limitations.extend(chain_limitations)
        except (RuntimeError, ValueError) as exc:
            errors["xrpl"] = str(exc)

    evm_spenders = list(args.evm_spender or [])
    if args.evm_spender_alias:
        evm_spenders.extend(args.evm_spender_alias)
    if args.evm_owner or evm_spenders or args.evm_token:
        if not args.evm_owner or not evm_spenders:
            errors["base"] = "EVM audit requires --evm-owner plus --evm-spender or --evm-spender-alias."
        else:
            try:
                tokens = [normalize_evm_address(t) for t in (args.evm_token or default_base_tokens())]
                report, chain_findings, chain_limitations = audit_base_allowances(
                    owner=args.evm_owner,
                    spenders=evm_spenders,
                    tokens=tokens,
                    rpc_url=args.base_rpc_url,
                    allow_private_rpc=args.allow_private_rpc,
                )
                chain_reports["base"] = report
                findings.extend(chain_findings)
                limitations.extend(chain_limitations)
            except (RuntimeError, ValueError) as exc:
                errors["base"] = str(exc)

    if not chain_reports and not errors:
        raise SystemExit(
            "Provide at least one target: --solana ADDRESS, --xrpl ADDRESS, or "
            "--evm-owner OWNER with --evm-spender/--evm-spender-alias."
        )

    if errors:
        chain_reports["errors"] = errors
    report = build_report(
        command="audit",
        chain_reports=chain_reports,
        findings=findings,
        limitations=list(dict.fromkeys(limitations)),
    )
    print_json(report)


def command_explain(_args: argparse.Namespace) -> None:
    print_json(
        {
            "tool": "web3-permission-audit",
            "version": VERSION,
            "risk_model": {
                "critical": "An external principal can transfer the current balance or has an unlimited approval on a funded token.",
                "high": "An external principal can transfer tokens, or an account/asset is frozen.",
                "medium": "A permission or issuer-control surface could affect assets but needs context before remediation.",
                "low": "Issuer/counterparty exposure or supply-control information worth reviewing for material holdings.",
                "info": "Context that helps explain the account's permission graph.",
            },
            "coverage": {
                "solana": [
                    "SPL Token and Token-2022 token account delegates",
                    "External close authorities",
                    "Frozen token accounts",
                    "Mint authority and freeze authority where jsonParsed RPC exposes them",
                ],
                "xrpl": [
                    "Trust line balances and limits",
                    "Freeze flags",
                    "Rippling flags",
                    "Issuer/counterparty exposure",
                ],
                "base_evm": [
                    "Targeted ERC-20 allowance checks for explicit owner/spender/token sets",
                    "Bundled common Base token set when --token is omitted",
                    "Permit2 spender alias",
                ],
            },
            "non_goals": [
                "No private keys, signing, or transaction submission",
                "No claim to discover every EVM approval without an indexer",
                "No remediation automation in the first version",
            ],
        }
    )


def add_common_flags(parser: argparse.ArgumentParser, *, suppress_default: bool = False) -> None:
    kwargs: dict[str, Any] = {
        "action": "store_true",
        "help": "Allow localhost/private RPC endpoints when intentionally querying a trusted private node.",
    }
    if suppress_default:
        kwargs["default"] = argparse.SUPPRESS
    else:
        kwargs["default"] = ALLOW_PRIVATE_RPC
    parser.add_argument("--allow-private-rpc", **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="web3_permission_audit.py",
        description="Read-only cross-chain wallet permission audit for Hermes Agent.",
    )
    add_common_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_explain = sub.add_parser("explain", help="Print the risk model and coverage boundaries")
    add_common_flags(p_explain, suppress_default=True)
    p_explain.set_defaults(func=command_explain)

    p_solana = sub.add_parser("solana", help="Audit Solana token delegates and authorities")
    add_common_flags(p_solana, suppress_default=True)
    p_solana.add_argument("address", help="Solana wallet address")
    p_solana.add_argument("--rpc-url", default=DEFAULT_SOLANA_RPC_URL, help="Solana RPC URL")
    p_solana.add_argument("--mint-limit", type=int, default=25, help="Max unique mints to inspect")
    p_solana.add_argument(
        "--detail-limit",
        type=int,
        default=100,
        help="Max raw token-account evidence rows to include; use -1 for all",
    )
    p_solana.set_defaults(func=command_solana)

    p_xrpl = sub.add_parser("xrpl", help="Audit XRPL trust line risk surface")
    add_common_flags(p_xrpl, suppress_default=True)
    p_xrpl.add_argument("address", help="XRPL classic account address")
    p_xrpl.add_argument("--rpc-url", default=DEFAULT_XRPL_RPC_URL, help="XRPL JSON-RPC URL")
    p_xrpl.add_argument("--limit", type=int, default=200, help="Trust lines per RPC page")
    p_xrpl.add_argument("--max-pages", type=int, default=5, help="Maximum trust-line pages to fetch")
    p_xrpl.set_defaults(func=command_xrpl)

    p_evm = sub.add_parser("evm-allowance", help="Audit targeted Base/EVM ERC-20 allowances")
    add_common_flags(p_evm, suppress_default=True)
    p_evm.add_argument("--owner", required=True, help="EVM owner address")
    p_evm.add_argument("--spender", action="append", help="Spender address; repeatable")
    p_evm.add_argument(
        "--spender-alias",
        action="append",
        choices=sorted(SPENDER_ALIASES),
        help="Known spender alias; currently: permit2",
    )
    p_evm.add_argument("--token", action="append", help="ERC-20 token contract; repeatable")
    p_evm.add_argument("--rpc-url", default=DEFAULT_BASE_RPC_URL, help="Base RPC URL")
    p_evm.set_defaults(func=command_base)

    p_audit = sub.add_parser("audit", help="Run a multi-chain audit in one report")
    add_common_flags(p_audit, suppress_default=True)
    p_audit.add_argument("--solana", help="Solana wallet address")
    p_audit.add_argument("--xrpl", help="XRPL classic account address")
    p_audit.add_argument("--evm-owner", help="Base/EVM owner address")
    p_audit.add_argument("--evm-spender", action="append", help="Base/EVM spender address; repeatable")
    p_audit.add_argument(
        "--evm-spender-alias",
        action="append",
        choices=sorted(SPENDER_ALIASES),
        help="Known Base/EVM spender alias; currently: permit2",
    )
    p_audit.add_argument("--evm-token", action="append", help="Base ERC-20 token contract; repeatable")
    p_audit.add_argument("--solana-rpc-url", default=DEFAULT_SOLANA_RPC_URL, help="Solana RPC URL")
    p_audit.add_argument("--xrpl-rpc-url", default=DEFAULT_XRPL_RPC_URL, help="XRPL JSON-RPC URL")
    p_audit.add_argument("--base-rpc-url", default=DEFAULT_BASE_RPC_URL, help="Base RPC URL")
    p_audit.add_argument("--mint-limit", type=int, default=25, help="Max Solana mints to inspect")
    p_audit.add_argument(
        "--detail-limit",
        type=int,
        default=100,
        help="Max Solana raw token-account evidence rows to include; use -1 for all",
    )
    p_audit.add_argument("--xrpl-limit", type=int, default=200, help="XRPL trust lines per page")
    p_audit.add_argument("--xrpl-max-pages", type=int, default=5, help="Max XRPL trust-line pages")
    p_audit.set_defaults(func=command_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
