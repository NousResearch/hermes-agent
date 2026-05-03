# Bankr-Preferred x402 Flow

Use this flow when the agent has Bankr wallet/signing capability available.

This is an x402-specific path. If you are using `x-quotient-api-key` auth, you can call endpoints directly without the x402 challenge/settle loop.

## Why Bankr First

- Wallet provisioning is already handled in typical Bankr setups.
- Signing and submission tooling is streamlined for agents.
- Reduces integration friction for autonomous request loops.
- Bankr signing path requires `X-API-Key` credentials for Bankr Agent API calls (for example, `/agent/sign`).

## Runtime Requirements

- Runtime can call Bankr Agent API endpoints with `X-API-Key`.
- API key has Agent API access enabled and is not read-only, so typed-data signing is permitted.

## Request Sequence

1. Send request to a monetized Quotient endpoint with no payment header.
2. Receive `402 Payment Required` and parse `PAYMENT-REQUIRED`.
3. Produce a valid x402 payment signature using the Bankr-controlled wallet.
4. Retry the same request with `PAYMENT-SIGNATURE`.
5. Parse `PAYMENT-RESPONSE` from the successful response.

## Required Headers

- On paid retry: `PAYMENT-SIGNATURE`
- Optional idempotency extension if available from your client stack: `Payment-Identifier`

## Practical Notes

- Keep request method/path/query/body identical between initial and paid retry.
- Treat malformed challenge payloads as hard failures and do not guess values.
- If settlement succeeds, cache reusable session/payment state only if your client confirms it is valid.

## Implementation Reference

If your Bankr client does not provide native x402 request wrapping, use the shared implementation in:

- `references/vanilla-x402-flow.md` -> "Concrete TypeScript Example (x402 Client Wrapper)"
- `references/vanilla-x402-flow.md` -> "Bankr-Compatible Signer Adapter (If Needed)"

This gives Bankr and non-Bankr agents one common x402 execution path.
