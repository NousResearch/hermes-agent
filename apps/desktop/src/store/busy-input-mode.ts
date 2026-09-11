import { atom } from 'nanostores'

export type BusyInputMode = 'interrupt' | 'queue' | 'steer'

export function normalizeBusyInputMode(value: unknown): BusyInputMode {
  const mode = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return mode === 'steer' || mode === 'queue' ? mode : 'interrupt'
}

export function busyInputOwnerKey(connectionId: string | null | undefined, profile: string | null | undefined): string {
  return JSON.stringify([connectionId ?? '', profile?.trim() || 'default'])
}

// A temporary config snapshot, not a persisted preference or session store.
export const $busyInputConfig = atom<{ owner: string; connection: unknown; mode: BusyInputMode } | null>(null)
