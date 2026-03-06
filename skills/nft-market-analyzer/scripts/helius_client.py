"""
helius_client.py
----------------
Read-only Helius API wrapper for the nft-analytics Hermes skill.

Endpoints covered
-----------------
  Enhanced Transaction History  GET  /v0/addresses/{address}/transactions
  Digital Asset Standard (DAS)  POST /  — getAssetsByOwner, getAsset
  Token Balances                GET  /v0/addresses/{address}/balances

No wallet seed or signing is ever required.
"""

import json
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL      = "https://api.helius.xyz/v0"
DAS_URL       = "https://mainnet.helius-rpc.com"
MAX_RETRIES   = 3
RETRY_DELAY_S = 1.5    # base seconds between retries (multiplied by attempt#)
PAGE_LIMIT    = 100    # maximum items per paginated request


# ─────────────────────────────────────────────────────────────────────────────
# Low-level HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[Dict] = None, verbose: bool = False) -> Any:
    """HTTP GET with retry and exponential back-off on HTTP 429."""
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(f"  [GET] {url}", file=sys.stderr)
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                wait = RETRY_DELAY_S * attempt
                print(
                    f"  [WARN] Rate limited. Waiting {wait:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Helius API HTTP {exc.code} on GET {url}: {body[:300]}"
            ) from exc

        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Helius API GET failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            time.sleep(RETRY_DELAY_S)

    return None


def _post(url: str, payload: Dict, verbose: bool = False) -> Any:
    """HTTP POST with retry and exponential back-off on HTTP 429."""
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(f"  [POST] {url}", file=sys.stderr)
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept":       "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                wait = RETRY_DELAY_S * attempt
                print(
                    f"  [WARN] Rate limited. Waiting {wait:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Helius API HTTP {exc.code} on POST {url}: {body[:300]}"
            ) from exc

        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Helius API POST failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            time.sleep(RETRY_DELAY_S)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HeliusClient
# ─────────────────────────────────────────────────────────────────────────────

class HeliusClient:
    """
    Thin, read-only wrapper around the Helius REST + DAS APIs.

    Parameters
    ----------
    api_key : str
        Your Helius API key (https://helius.dev). Required.
    verbose : bool
        When True, each HTTP request URL is printed to stderr.
    """

    def __init__(self, api_key: str, verbose: bool = False) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("HeliusClient requires a non-empty API key.")
        self.api_key = api_key.strip()
        self.verbose = verbose

    # ── Transaction History ───────────────────────────────────────────────────

    def get_transactions(
        self,
        wallet: str,
        limit: int = PAGE_LIMIT,
        max_pages: int = 10,
    ) -> List[Dict]:
        """
        Fetch the full enhanced transaction history for *wallet*.

        Paginates automatically using the ``before`` cursor until all pages
        are retrieved or *max_pages* is reached.

        Returns
        -------
        list[dict]
            Flat list of Helius enhanced-transaction objects.
        """
        url    = f"{BASE_URL}/addresses/{wallet}/transactions"
        all_txs: List[Dict] = []
        before: Optional[str] = None

        for _ in range(max_pages):
            params: Dict[str, Any] = {
                "api-key": self.api_key,
                "limit":   min(limit, PAGE_LIMIT),
            }
            if before:
                params["before"] = before

            batch = _get(url, params=params, verbose=self.verbose)
            if not batch:
                break

            all_txs.extend(batch)

            if len(batch) < PAGE_LIMIT:
                break  # last page

            before = batch[-1].get("signature")
            if not before:
                break

            time.sleep(0.2)  # polite pacing

        return all_txs

    # ── Digital Asset Standard (DAS) ──────────────────────────────────────────

    def get_assets_by_owner(
        self,
        wallet: str,
        limit: int = PAGE_LIMIT,
        max_pages: int = 5,
    ) -> List[Dict]:
        """
        Fetch all NFTs currently owned by *wallet* via DAS ``getAssetsByOwner``.

        Returns
        -------
        list[dict]
            Flat list of DAS asset objects.
        """
        url       = f"{DAS_URL}/?api-key={self.api_key}"
        all_assets: List[Dict] = []
        page = 1

        for _ in range(max_pages):
            payload = {
                "jsonrpc": "2.0",
                "id":      "nft-analytics",
                "method":  "getAssetsByOwner",
                "params": {
                    "ownerAddress": wallet,
                    "page":         page,
                    "limit":        limit,
                    "displayOptions": {
                        "showFungible":      False,
                        "showNativeBalance": False,
                    },
                },
            }
            resp  = _post(url, payload, verbose=self.verbose)
            items = resp.get("result", {}).get("items", []) if resp else []
            if not items:
                break
            all_assets.extend(items)
            if len(items) < limit:
                break
            page += 1
            time.sleep(0.2)

        return all_assets

    def get_asset(self, mint_address: str) -> Optional[Dict]:
        """
        Fetch full metadata for a single NFT by *mint_address* via DAS ``getAsset``.

        Returns ``None`` if the request fails or the asset is not found.
        """
        url = f"{DAS_URL}/?api-key={self.api_key}"
        payload = {
            "jsonrpc": "2.0",
            "id":      "nft-analytics",
            "method":  "getAsset",
            "params":  {"id": mint_address},
        }
        resp = _post(url, payload, verbose=self.verbose)
        return resp.get("result") if resp else None

    # ── Token Balances ────────────────────────────────────────────────────────

    def get_token_balances(self, wallet: str) -> Dict:
        """
        Fetch SPL token balances for *wallet*.

        Returns
        -------
        dict
            Full Helius balances response, or ``{}`` on failure.
        """
        url    = f"{BASE_URL}/addresses/{wallet}/balances"
        params = {"api-key": self.api_key}
        result = _get(url, params=params, verbose=self.verbose)
        return result or {}


# ─────────────────────────────────────────────────────────────────────────────
# TensorClient
# ─────────────────────────────────────────────────────────────────────────────


TENSOR_REST_URL = "https://api.mainnet.tensordev.io/api/v1/user/transactions"

class TensorClient:
    """
    Read-only Tensor Trade REST API wrapper.
    Fetches real NFT buy/sell history for accurate ROI calculation.
    """

    def __init__(self, api_key: str, verbose: bool = False) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("TensorClient requires a non-empty API key.")
        self.api_key = api_key.strip()
        self.verbose = verbose

    def get_wallet_transactions(self, wallet: str) -> list:
        """
        Fetch NFT transaction history for a wallet from Tensor REST API.
        Returns a list of buy/sell events with real prices.
        """
        try:
            url = f"{TENSOR_REST_URL}?wallets={wallet}&limit=100"
            if self.verbose:
                print(f"  [TENSOR] {url}", file=sys.stderr)
            req = urllib.request.Request(
                url,
                headers={
                    "X-TENSOR-API-KEY": self.api_key,
                    "User-Agent": "Mozilla/5.0",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                txs = result.get("txs", [])
                # Normalize to match profiler expectations
                normalized = []
                for tx in txs:
                    normalized.append({
                        "txType": tx.get("txType", ""),
                        "grossAmount": int(tx.get("grossAmount") or 0),
                        "mint": {"onchainId": tx.get("mintOnchainId", "")},
                        "buyer": {"address": tx.get("buyerId") or ""},
                        "seller": {"address": tx.get("sellerId") or ""},
                        "txAt": tx.get("txAt", ""),
                    })
                return normalized
        except Exception as exc:
            print(f"  [WARN] Tensor API error: {exc}", file=sys.stderr)
            return []

