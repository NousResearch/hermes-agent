import type { BillingBlock } from '@hermes/shared'
import { atom } from 'nanostores'

import { openExternalLink } from '@/lib/external-link'

/**
 * The active inference billing wall, if any. Set from the gateway
 * `message.complete` / `error` event when a turn fails with
 * `FailoverReason.billing` (see `agent/billing_links.py`). One global slot: a
 * credit wall on the active session's provider is the whole app's problem, and
 * the newest block wins. Cleared when a new turn starts or the user dismisses.
 */
export interface ActiveBillingBlock {
  block: BillingBlock
  sessionId: string
  at: number
}

export const $billingBlock = atom<ActiveBillingBlock | null>(null)

/**
 * Navigation intent counter. A toast fired outside React (or any surface
 * without router context) bumps this to ask the shell — which owns
 * `useNavigate` — to open Settings → Billing in-app. See `contrib/wiring.tsx`.
 */
export const $billingSettingsRequest = atom(0)

export function setBillingBlock(sessionId: string, block: BillingBlock): void {
  $billingBlock.set({ at: Date.now(), block, sessionId })
}

export function clearBillingBlock(sessionId?: string): void {
  const current = $billingBlock.get()

  if (!current) {
    return
  }

  // A scoped clear (new turn on session X) must not wipe a block raised by a
  // different session's provider.
  if (sessionId && current.sessionId !== sessionId) {
    return
  }

  $billingBlock.set(null)
}

export function requestBillingSettings(): void {
  $billingSettingsRequest.set($billingSettingsRequest.get() + 1)
}

/**
 * The single recovery action for a billing wall, shared by the toast and the
 * in-chat banner so both behave identically: Nous routes to the in-app
 * Settings → Billing surface; a third-party provider deep-links to its own
 * billing page (falling back to the in-app surface only if we have no URL).
 */
export function runBillingRecovery(block: BillingBlock): void {
  if (block.is_nous) {
    requestBillingSettings()

    return
  }

  if (block.billing_url) {
    openExternalLink(block.billing_url)

    return
  }

  requestBillingSettings()
}

export function billingCtaLabel(block: BillingBlock, copy: { addCredits: string; openBilling: string }): string {
  return block.is_nous ? copy.openBilling : copy.addCredits
}
