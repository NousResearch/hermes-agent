import { atom } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.investigationOnly'

const stored = readKey(STORAGE_KEY)
export const $investigationOnly = atom<boolean | null>(stored === null ? null : stored === 'true')

export function setInvestigationOnly(enabled: boolean | null): void {
  $investigationOnly.set(enabled)
  writeKey(STORAGE_KEY, enabled === null ? null : String(enabled))
}

export function selectedMutationPolicy(): 'allowed' | 'forbidden' | undefined {
  const enabled = $investigationOnly.get()

  return enabled === null ? undefined : enabled ? 'forbidden' : 'allowed'
}
