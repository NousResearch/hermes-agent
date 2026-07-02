# ADR: Telegram Mini App frontend SDK — hand-rolled wrapper vs @telegram-apps/sdk-react

Date: 2026-07-02
Status: accepted (revisit before the action-gate implementation, see M18 spec)

## Context

`apps/telegram-miniapp` currently integrates with Telegram through a
hand-rolled typed wrapper over the raw `window.Telegram.WebApp` object
(`src/telegram.ts`, ~150 lines: viewport, BackButton, MainButton, haptics,
theme, initData access). The community-standard alternative is the
`@telegram-apps/sdk-react` + `@telegram-apps/telegram-ui` ecosystem
(tma.js monorepo — active, last push 2026-06, not deprecated).

An audit on 2026-07-02 flagged the divergence as an undocumented drift risk:
`telegram.ts` mirrors 13 themeParams fields by hand and new Bot API fields
(Mini Apps 2.0: fullscreen, home-screen shortcut, secondary button) would
each need manual support.

## Decision

Keep the hand-rolled wrapper for the current read-only phase. Adopt
`@telegram-apps/sdk-react` only together with the action-gate milestone
(M18 Phase 1), when the frontend gains real interaction complexity.

Rationale:

- The read-only surface uses a narrow slice of the WebApp API; the wrapper
  covers it in ~150 audited lines with zero dependencies. The static
  guardrails (`test_telegram_miniapp_frontend_guardrails.py`) scan all
  shipped sources; adding a third-party SDK now would put network-capable
  code outside that audit boundary for no functional gain.
- Supply-chain surface stays minimal while the sidecar is being hardened.
- Mini Apps 2.0 features (fullscreen, home-screen shortcut, secondary
  button) are wanted for the Control Deck UX; implementing them via the SDK
  at action-gate time avoids doing the migration twice.

## Consequences

- `telegram.ts` remains the single integration point; any new WebApp field
  must be added there (and only there).
- When the action-gate milestone starts: introduce `@telegram-apps/sdk-react`,
  migrate `telegram.ts` call sites, and add a JS test runner (vitest) in the
  same milestone — the deferred mocked-fetch allowlist test from the
  2026-07-02 guardrail review lands then as well.
- If the action-gate is rejected, this ADR stands and the wrapper stays.
