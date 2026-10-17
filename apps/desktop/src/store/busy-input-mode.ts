import { atom } from 'nanostores'

export type BusyInputMode = 'interrupt' | 'queue' | 'steer'

export function normalizeBusyInputMode(value: unknown): BusyInputMode {
  const mode = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return mode === 'steer' || mode === 'queue' ? mode : 'interrupt'
}

export function busyInputOwnerKey(connectionId?: string | null, profile?: string | null): string {
  return JSON.stringify([connectionId ?? '', profile?.trim() || 'default'])
}

// A backend config snapshot, never a persisted Desktop-only preference.
export const $busyInputConfig = atom<{ owner: string; connection: unknown; mode: BusyInputMode } | null>(null)

/** Config-level hint for the active connection/profile; session dispatch uses useBusyInputMode. */
export function configuredBusyInputMode(
  config: ReturnType<typeof $busyInputConfig.get>,
  connection: { connectionId?: string | null } | null,
  profile: string | null | undefined
): BusyInputMode {
  return config?.connection === connection && config.owner === busyInputOwnerKey(connection?.connectionId, profile)
    ? config.mode
    : 'interrupt'
}
