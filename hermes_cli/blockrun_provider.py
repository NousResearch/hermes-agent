"""
BlockRun / ClawRouter provider for hermes-agent.

Plugs into hermes's runtime_provider system.  When the user selects
provider="blockrun" (or "clawrouter"), we:

  1. Load the user's wallet key from the environment (Base or Solana).
  2. Attach a custom httpx transport to the OpenAI client constructed in
     hermes's inference loop.
  3. The transport intercepts HTTP 402 Payment Required responses from
     blockrun.ai / sol.blockrun.ai, signs the payment locally, and retries
     the request — fully transparent to the rest of hermes.

No API key is required: the wallet private key IS the credential.
Payments are non-custodial signatures; the key never leaves the user's machine.

Supported chains:
  Base (default) — USDC on Base mainnet, EIP-712 signing
    gateway : https://blockrun.ai/api/v1
    env var : BLOCKRUN_WALLET_KEY=0x...

  Solana        — USDC on Solana, Ed25519 signing
    gateway : https://sol.blockrun.ai/api/v1
    env var : SOLANA_WALLET_KEY=<base58-private-key>

Chain auto-detection:
  • If SOLANA_WALLET_KEY is set (and BLOCKRUN_WALLET_KEY is not) → Solana
  • If BLOCKRUN_CHAIN=solana                                      → Solana
  • Otherwise                                                     → Base

Usage in cli-config.yaml:
  model:
    provider: "blockrun"          # or "clawrouter" (alias)
    default: "openai/gpt-5.2"    # any model from blockrun.ai/api/v1/models
    # chain: "solana"             # optional — defaults to "base"
"""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx

# ── Endpoints ────────────────────────────────────────────────────────────────
BLOCKRUN_BASE_URL        = "https://blockrun.ai/api/v1"
BLOCKRUN_BASE_TESTNET    = "https://testnet.blockrun.ai/api/v1"
BLOCKRUN_SOLANA_URL      = "https://sol.blockrun.ai/api/v1"

# ── Env var names ─────────────────────────────────────────────────────────────
_BASE_KEY_VARS   = ("BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY")
_SOLANA_KEY_VARS = ("SOLANA_WALLET_KEY", "BLOCKRUN_SOLANA_KEY")

Chain = Literal["base", "solana"]


# ── Wallet helpers ────────────────────────────────────────────────────────────

def _load_base_key() -> str | None:
    for var in _BASE_KEY_VARS:
        val = os.environ.get(var)
        if val:
            return val
    session_file = os.path.expanduser("~/.blockrun/.session")
    if os.path.exists(session_file):
        try:
            key = open(session_file).read().strip()
            if key:
                return key
        except OSError:
            pass
    return None


def _load_solana_key() -> str | None:
    for var in _SOLANA_KEY_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def _detect_chain() -> Chain:
    """
    Auto-detect chain from env vars only (not session files, which are storage
    not preference signals).  Priority:
      1. BLOCKRUN_CHAIN=solana → Solana
      2. SOLANA_WALLET_KEY set and BLOCKRUN_WALLET_KEY not set → Solana
      3. Otherwise → Base
    """
    if os.environ.get("BLOCKRUN_CHAIN", "").lower() == "solana":
        return "solana"
    has_solana_env = any(os.environ.get(v) for v in _SOLANA_KEY_VARS)
    has_base_env   = any(os.environ.get(v) for v in _BASE_KEY_VARS)
    if has_solana_env and not has_base_env:
        return "solana"
    return "base"


# ---------------------------------------------------------------------------
# Shared transport mixin — sign x402 payments (Base or Solana)
# ---------------------------------------------------------------------------

_USER_AGENT = "hermes-agent/blockrun-integration/1.0.0"


class _X402Mixin:
    """
    Common signing logic shared by sync and async transports.
    Detects chain from the 402 response headers and dispatches to the
    appropriate signing implementation.
    """

    def _inject_ua(self, request: httpx.Request) -> httpx.Request:
        """Stamp every outgoing request with the hermes-agent User-Agent."""
        headers = dict(request.headers)
        headers["user-agent"] = _USER_AGENT
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
        )

    def _sign_base_payment(
        self, account: Any, request: httpx.Request, payment_header: str
    ) -> str:
        """EIP-712 signing for Base (USDC on eip155:8453)."""
        from blockrun_llm.x402 import (
            parse_payment_required,
            extract_payment_details,
            create_payment_payload,
        )
        payment_required = parse_payment_required(payment_header)
        details = extract_payment_details(payment_required)
        return create_payment_payload(
            account=account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=str(request.url),
            resource_description="BlockRun AI API call via hermes-agent",
        )

    def _sign_solana_payment(
        self, x402_client: Any, request: httpx.Request, payment_header: str
    ) -> str:
        """
        Ed25519 signing for Solana (USDC-SPL).
        Uses x402ClientSync (same approach as SolanaLLMClient internally).
        """
        from blockrun_llm.solana_client import (
            decode_payment_required_header,
            encode_payment_signature_header,
        )
        payment_required = decode_payment_required_header(payment_header)
        payment_payload  = x402_client.create_payment_payload(payment_required)
        return encode_payment_signature_header(payment_payload)

    def _sign_payment(self, request: httpx.Request, payment_header: str) -> str:
        if self._chain == "solana":
            return self._sign_solana_payment(self._signer, request, payment_header)
        return self._sign_base_payment(self._signer, request, payment_header)

    def _payment_header(self, response: httpx.Response) -> str | None:
        return (
            response.headers.get("X-Payment-Required")
            or response.headers.get("x-payment-required")
        )

    def _make_retry(
        self, request: httpx.Request, signed: str
    ) -> httpx.Request:
        headers = dict(request.headers)
        headers["PAYMENT-SIGNATURE"] = signed
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
        )


# ---------------------------------------------------------------------------
# Sync transport
# ---------------------------------------------------------------------------

class BlockRunX402Transport(_X402Mixin, httpx.HTTPTransport):
    """
    Synchronous httpx transport that handles HTTP 402 Payment Required
    responses from blockrun.ai (Base) and sol.blockrun.ai (Solana).
    """

    def __init__(self, private_key: str, chain: Chain = "base", **kwargs: Any) -> None:
        httpx.HTTPTransport.__init__(self, **kwargs)
        self._chain = chain
        self._signer = _load_signer(private_key, chain)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request = self._inject_ua(request)
        response = super().handle_request(request)
        payment_header = self._payment_header(response)
        if response.status_code != 402 or not payment_header:
            return response
        try:
            signed = self._sign_payment(request, payment_header)
        except Exception as exc:
            raise RuntimeError(f"BlockRun x402 payment failed: {exc}") from exc
        return super().handle_request(self._make_retry(request, signed))


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------

class AsyncBlockRunX402Transport(_X402Mixin, httpx.AsyncHTTPTransport):
    """Async variant — used when hermes constructs an async OpenAI client."""

    def __init__(self, private_key: str, chain: Chain = "base", **kwargs: Any) -> None:
        httpx.AsyncHTTPTransport.__init__(self, **kwargs)
        self._chain = chain
        self._signer = _load_signer(private_key, chain)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = self._inject_ua(request)
        response = await super().handle_async_request(request)
        payment_header = self._payment_header(response)
        if response.status_code != 402 or not payment_header:
            return response
        try:
            signed = self._sign_payment(request, payment_header)
        except Exception as exc:
            raise RuntimeError(f"BlockRun x402 payment failed: {exc}") from exc
        return await super().handle_async_request(self._make_retry(request, signed))


# ---------------------------------------------------------------------------
# Signer loader
# ---------------------------------------------------------------------------

def _load_signer(private_key: str, chain: Chain) -> Any:
    """
    Return the appropriate signer for the chain.

    Base   → eth_account.LocalAccount  (used directly in create_payment_payload)
    Solana → x402ClientSync pre-configured with SVM keypair signer
             (mirrors SolanaLLMClient's internal setup exactly)
    """
    if chain == "solana":
        try:
            from blockrun_llm.solana_client import (
                x402ClientSync,
                register_exact_svm_client,
                _create_signer as _solana_create_signer,
            )
        except ImportError as exc:
            raise ImportError(
                "blockrun_llm Solana support required. "
                "Run: pip install blockrun-llm solders base58"
            ) from exc
        x402_client = x402ClientSync()
        signer = _solana_create_signer(private_key)
        register_exact_svm_client(x402_client, signer)
        return x402_client  # stored as self._signer; passed to _sign_solana_payment
    else:
        try:
            from eth_account import Account
            return Account.from_key(private_key)
        except ImportError as exc:
            raise ImportError(
                "eth-account is required for Base payments. "
                "Run: pip install eth-account"
            ) from exc


# ---------------------------------------------------------------------------
# Client factories (called from the hermes inference loop)
# ---------------------------------------------------------------------------

def create_sync_client(
    private_key: str,
    chain: Chain = "base",
    base_url: str | None = None,
) -> Any:
    """
    Return an openai.OpenAI client wired to BlockRun with x402 payment support.
    Automatically selects Base or Solana gateway based on `chain`.
    """
    import openai

    url = base_url or (BLOCKRUN_SOLANA_URL if chain == "solana" else BLOCKRUN_BASE_URL)
    transport = BlockRunX402Transport(private_key=private_key, chain=chain)
    http_client = httpx.Client(
        transport=transport,
        timeout=120.0,
        headers={"User-Agent": "hermes-agent/blockrun-integration/1.0.0"},
    )

    return openai.OpenAI(
        base_url=url,
        api_key="x402-wallet",  # placeholder — auth is the x402 payment signature
        http_client=http_client,
    )


def create_async_client(
    private_key: str,
    chain: Chain = "base",
    base_url: str | None = None,
) -> Any:
    """Async variant of create_sync_client."""
    import openai

    url = base_url or (BLOCKRUN_SOLANA_URL if chain == "solana" else BLOCKRUN_BASE_URL)
    transport = AsyncBlockRunX402Transport(private_key=private_key, chain=chain)
    http_client = httpx.AsyncClient(
        transport=transport,
        timeout=120.0,
        headers={"User-Agent": "hermes-agent/blockrun-integration/1.0.0"},
    )

    return openai.AsyncOpenAI(
        base_url=url,
        api_key="x402-wallet",
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Provider resolver — called from hermes_cli/runtime_provider.py
# ---------------------------------------------------------------------------

def resolve_blockrun_provider(
    *,
    explicit_api_key: str | None,
    explicit_base_url: str | None,
    chain: Chain | None = None,
) -> dict[str, Any]:
    """
    Resolve BlockRun / ClawRouter runtime provider config.

    Chain selection (in priority order):
      1. `chain` argument (from cli-config.yaml model.chain)
      2. BLOCKRUN_CHAIN env var
      3. Auto-detect: Solana if only SOLANA_WALLET_KEY is set, else Base

    Returns the standard hermes provider dict plus blockrun_client_factory
    so the inference layer can build an x402-enabled OpenAI client.
    """
    resolved_chain: Chain = chain or _detect_chain()

    # Load wallet key for the chosen chain
    if resolved_chain == "solana":
        private_key = explicit_api_key or _load_solana_key()
        default_url = BLOCKRUN_SOLANA_URL
        key_hint = "SOLANA_WALLET_KEY=<base58-key>"
    else:
        private_key = explicit_api_key or _load_base_key()
        default_url = (
            BLOCKRUN_BASE_TESTNET
            if os.environ.get("NETWORK_MODE") == "testnet"
            else BLOCKRUN_BASE_URL
        )
        key_hint = "BLOCKRUN_WALLET_KEY=0x..."

    if not private_key:
        raise ValueError(
            f"BlockRun provider ({resolved_chain}) requires a wallet key.\n"
            f"  • Set {key_hint} in your environment, or\n"
            f"  • Run the `blockrun_wallet_setup` tool to create one automatically."
        )

    base_url = explicit_base_url or os.environ.get("BLOCKRUN_BASE_URL") or default_url

    return {
        "provider": "blockrun",
        "api_mode": "chat_completions",
        "base_url": base_url,
        "api_key": "x402-wallet",
        "source": f"blockrun_{resolved_chain}_wallet_key",
        # Extra metadata for the inference layer
        "blockrun_chain": resolved_chain,
        "blockrun_private_key": private_key,
        "blockrun_client_factory": lambda: create_sync_client(private_key, resolved_chain, base_url),
        "blockrun_async_client_factory": lambda: create_async_client(private_key, resolved_chain, base_url),
    }
