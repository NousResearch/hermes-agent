import { readKey, writeJson, writeKey } from '@/lib/storage'

import type { HandoffReceipt } from './handoff-leg'

// Holds the receipt in memory when the disk write fails, so this window still has the session identity.
// saveHandoffReceipt throws when the write does not read back, so nothing is submitted without a saved receipt.
const unsavedReceipts = new Map<string, HandoffReceipt>()

export function markFirstBuildSession(storedId: string): void {
  writeKey('hermes.onboarding.first-build.v1', storedId)
}

export function endFirstBuildConnect(storedId: string): void {
  writeKey('hermes.onboarding.first-build.done.v1', storedId)
}

export function isFirstBuildSession(storedId: string | null | undefined): boolean {
  return (
    !!storedId &&
    readKey('hermes.onboarding.first-build.v1') === storedId &&
    readKey('hermes.onboarding.first-build.done.v1') !== storedId
  )
}

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

  // JSON cannot encode a constructor, so only a primitive string has String as its constructor here.
  // Comparing constructors rejects a corrupt id instead of coercing it to text.
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

export function quarantineHandoffReceipt(key: string): void {
  writeKey(`${key}.unreadable`, readKey(key))
  writeKey(key, null)
}

export function saveHandoffReceipt(key: string, receipt: HandoffReceipt): void {
  unsavedReceipts.set(key, receipt)
  writeJson(key, receipt)

  if (readKey(key) !== JSON.stringify(receipt)) {
    throw new Error('Could not save the first-build session for recovery. No new start was sent.')
  }

  unsavedReceipts.delete(key)
}
