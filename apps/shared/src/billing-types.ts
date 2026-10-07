/**
 * Shared billing wire contracts.
 *
 * These shapes round-trip between the Python tui_gateway and TypeScript clients
 * such as the TUI and desktop app. Keep rendering state, client logic, and the
 * gateway event union out of this runtime-free module.
 */

// ── Billing wall (inference credit exhaustion) ───────────────────────

/**
 * Structured billing-wall descriptor emitted by the gateway on the
 * `message.complete` event (`payload.billing`) when an inference call fails
 * because the account is out of credits / payment is required — mirrors the
 * Python `agent/billing_links.py::BillingBlock`.
 *
 * Detection is backend-only (`agent/error_classifier.py` →
 * `FailoverReason.billing`), so every surface renders from this one signal and
 * never re-classifies free-form error text. Recovery deep-links to the
 * provider's own `billing_url`.
 */
export type { BillingBlock } from './gateway-contract.generated.js'
