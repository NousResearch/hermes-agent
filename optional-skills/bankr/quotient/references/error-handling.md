# Error Handling (API Key and x402 Skill Paths)

This file documents the error contract for both supported auth paths: API key and x402.

## Status Codes

- `200` - Success.
- `401 invalid_api_key` - API key missing, invalid, or revoked.
- `401 gateway_required` - Gateway auth requirement not satisfied.
- `402 payment_required` - Payment challenge issued; read `PAYMENT-REQUIRED`, sign, retry with `PAYMENT-SIGNATURE`.
- `403 insufficient_credits` - Account credits are exhausted.
- `404` - Resource slug not found.
- `422` - Invalid request parameters or cursor mismatch.
- `429` - Rate limited; back off and retry.
- `5xx` - Upstream or gateway transient failure; retry with bounded backoff.

## API Key-Specific Failure Cases

- `401 invalid_api_key`: rotate or replace key from the Quotient account/developer area.
- `403 insufficient_credits`: top up billing/credits or switch to x402 flow for paid requests.

## x402-Specific Failure Cases

- Missing `PAYMENT-REQUIRED` on `402`: treat as gateway/proxy error and stop.
- Missing `PAYMENT-RESPONSE` after paid success: treat response as incomplete; log request id.
- Payment signature rejected: request a new challenge and sign again.

## Retry Guidance

- Use exponential backoff with jitter for `429` and `5xx`.
- Do not retry `422` without correcting inputs.
- Keep retries idempotent and bounded.
