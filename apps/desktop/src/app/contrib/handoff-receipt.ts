import { atom } from 'nanostores'

import { $setupHandoff } from '@/components/onboarding-chat/setup-profile'
import { readKey, writeJson } from '@/lib/storage'

import type { HandoffReceipt } from './handoff-leg'

export const $handoffError = atom<string | null>(null)
// A failed disk write still remembers the original identity for this window.
// Nothing is submitted until the next save verifies durable persistence.
const unsavedReceipts = new Map<string, HandoffReceipt>()

/** Only a deliberate retry lifts an error; re-rendering a directive does not. */
export function retrySetupHandoff(): void {
  const state = $setupHandoff.get()

  if (state?.phase !== 'error') {
    return
  }

  $handoffError.set(null)
  $setupHandoff.set({ ...state, phase: 'pending' })
}

/** A navigation/submit receipt, never a copy of either profile's memory. */
export function handoffReceiptKey(connection: null | string, guideStoredId: string): string {
  return `hermes.onboarding.handoff.v1.connection.${encodeURIComponent(connection ?? 'ambient')}.profile.default.guide.${encodeURIComponent(guideStoredId)}`
}

/** A receipt owner is either a registry source id or null, the ambient route. */
const isOwnerConnection = (value: unknown): value is null | string =>
  value === null || (typeof value === 'string' && value.length > 0)

export function readHandoffReceipt(key: string): HandoffReceipt | null {
  const unsaved = unsavedReceipts.get(key)

  if (unsaved) {
    return unsaved
  }

  const raw = readKey(key)

  if (raw === null) {
    return null
  }

  let value: HandoffReceipt

  try {
    value = JSON.parse(raw)
  } catch {
    throw new Error(
      'The saved first-build receipt could not be read. Check your sessions before starting another build.'
    )
  }

  if (
    !value ||
    typeof value.storedId !== 'string' ||
    !value.storedId ||
    typeof value.runtimeId !== 'string' ||
    typeof value.task !== 'string' ||
    typeof value.brief !== 'string' ||
    !value.owner ||
    !isOwnerConnection(value.owner.connectionId) ||
    value.owner.profile !== 'default' ||
    !['build', 'plugin', 'machine-setup'].includes(value.plan) ||
    !['created', 'submitting', 'accepted'].includes(value.status)
  ) {
    throw new Error(
      'The saved first-build receipt could not be read. Check your sessions before starting another build.'
    )
  }

  return value
}

export function saveHandoffReceipt(key: string, receipt: HandoffReceipt): void {
  unsavedReceipts.set(key, receipt)
  writeJson(key, receipt)

  if (readKey(key) !== JSON.stringify(receipt)) {
    throw new Error('Could not save the first-build session for recovery. No new start was sent.')
  }

  unsavedReceipts.delete(key)
}
