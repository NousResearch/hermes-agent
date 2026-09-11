import { readKey, writeJson } from '@/lib/storage'

import type { HandoffReceipt } from './handoff-leg'

// A failed disk write still remembers the original identity for this window.
// Nothing is submitted until the next save verifies durable persistence.
const unsavedReceipts = new Map<string, HandoffReceipt>()

/** A navigation/submit receipt, never a copy of either profile's memory. */
export function handoffReceiptKey(connection: null | string, guideStoredId: string): string {
  return `hermes.onboarding.handoff.v1.connection.${encodeURIComponent(connection ?? 'ambient')}.profile.default.guide.${encodeURIComponent(guideStoredId)}`
}

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

  // JSON cannot encode a constructor function: only primitive strings have
  // String as their constructor here. Validate without coercing corrupt ids.
  const hasTextFields = [value?.storedId, value?.runtimeId, value?.task, value?.brief].every(
    field => field?.constructor === String
  )

  const connectionId = value?.owner?.connectionId
  const validConnection = connectionId === null || (connectionId?.constructor === String && connectionId.length > 0)

  if (
    !hasTextFields ||
    !value.storedId ||
    !validConnection ||
    value.owner?.profile !== 'default' ||
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
