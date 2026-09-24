/**
 * Operator settings lock — the client half of `config.lock.status` / `config.unlock` /
 * `config.relock`.
 *
 * The lock is enforced on the gateway inside `save_config`, so nothing here can protect a
 * setting; this store only knows whether the lock is on, what it covers, and whether an unlock
 * window is currently open, so the UI can say so and offer to open one.
 */

import { atom } from 'nanostores'

export type SettingsLockRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export interface SettingsLockStatus {
  enabled: boolean
  keys: string[]
  passwordRequired: boolean
  unusable: boolean
  unlocked: boolean
  unlockedUntil: number | null
}

export const SETTINGS_LOCK_UNKNOWN: SettingsLockStatus = {
  enabled: false,
  keys: [],
  passwordRequired: false,
  unusable: false,
  unlocked: false,
  unlockedUntil: null
}

export const $settingsLock = atom<SettingsLockStatus>(SETTINGS_LOCK_UNKNOWN)

/** A gateway that predates the lock answers with an error; that reads as "no lock", not a crash. */
function normalize(raw: unknown): SettingsLockStatus {
  const value = (raw ?? {}) as Record<string, unknown>
  const until = Number(value.unlocked_until)

  return {
    enabled: value.enabled === true,
    keys: Array.isArray(value.keys) ? value.keys.map(key => String(key)) : [],
    passwordRequired: value.password_required === true,
    unusable: value.unusable === true,
    unlocked: value.unlocked === true,
    unlockedUntil: Number.isFinite(until) && until > 0 ? until : null
  }
}

export async function syncSettingsLock(requestGateway: SettingsLockRequester): Promise<SettingsLockStatus> {
  try {
    const status = normalize(await requestGateway('config.lock.status'))
    $settingsLock.set(status)

    return status
  } catch {
    $settingsLock.set(SETTINGS_LOCK_UNKNOWN)

    return SETTINGS_LOCK_UNKNOWN
  }
}

/**
 * Open an unlock window. Returns false on a wrong password — the only outcome the caller has to
 * tell the user about. The password is passed straight to the gateway and never stored here.
 */
export async function unlockSettings(
  requestGateway: SettingsLockRequester,
  password: string,
  minutes = 15
): Promise<boolean> {
  try {
    await requestGateway('config.unlock', { minutes, password })
  } catch {
    return false
  }

  await syncSettingsLock(requestGateway)

  return true
}

export async function relockSettings(requestGateway: SettingsLockRequester): Promise<void> {
  try {
    await requestGateway('config.relock')
  } catch {
    // Already closed, or an older gateway: the resync below reports the truth either way.
  }

  await syncSettingsLock(requestGateway)
}
