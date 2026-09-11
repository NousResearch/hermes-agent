import { atom } from 'nanostores'

import { hashPasscode, type PasscodeRecord, verifyPasscode } from '@/lib/profile-passcode'
import { normalizeProfileKey } from '@/store/profile'

/**
 * Per-profile desktop passcode lock (#94028).
 *
 * Records live in renderer-persisted storage keyed by normalized profile key —
 * the same home as the other per-profile desktop prefs (user themes, layout
 * tree). This is a UI-level privacy gate, not a security boundary: nothing
 * here blocks the CLI or the gateway, only the desktop surface.
 *
 * An unlock is scoped to the current profile activation: switching to a
 * different profile forgets it, so coming back re-prompts (the issue asks for
 * a re-prompt on switch, not silent reuse).
 */

export const PROFILE_LOCK_STORAGE_KEY = 'hermes.desktop.profile-lock.v1'

export type ProfileLockMap = Record<string, PasscodeRecord>

function readPersistedLocks(): ProfileLockMap {
  try {
    const raw = window.localStorage.getItem(PROFILE_LOCK_STORAGE_KEY)
    if (!raw) {
      return {}
    }
    const parsed: unknown = JSON.parse(raw)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as ProfileLockMap) : {}
  } catch {
    return {}
  }
}

function writePersistedLocks(locks: ProfileLockMap): void {
  window.localStorage.setItem(PROFILE_LOCK_STORAGE_KEY, JSON.stringify(locks))
}

export const $profileLocks = atom<ProfileLockMap>(readPersistedLocks())

/** Profile key unlocked in this window session (null = nothing unlocked). */
export const $unlockedProfile = atom<string | null>(null)

export function hasProfilePasscode(profileKey: string): boolean {
  return Boolean($profileLocks.get()[normalizeProfileKey(profileKey)])
}

export function isProfileLocked(profileKey: string): boolean {
  const key = normalizeProfileKey(profileKey)
  return hasProfilePasscode(key) && $unlockedProfile.get() !== key
}

export async function setProfilePasscode(profileKey: string, passcode: string): Promise<void> {
  const key = normalizeProfileKey(profileKey)
  const record = await hashPasscode(passcode)
  const next = { ...$profileLocks.get(), [key]: record }
  $profileLocks.set(next)
  writePersistedLocks(next)
}

export function clearProfilePasscode(profileKey: string): void {
  const key = normalizeProfileKey(profileKey)
  const next = { ...$profileLocks.get() }
  delete next[key]
  $profileLocks.set(next)
  writePersistedLocks(next)
  if ($unlockedProfile.get() === key) {
    $unlockedProfile.set(null)
  }
}

export async function tryUnlockProfile(profileKey: string, passcode: string): Promise<boolean> {
  const key = normalizeProfileKey(profileKey)
  const record = $profileLocks.get()[key]
  if (!record) {
    return true
  }
  const ok = await verifyPasscode(passcode, record)
  if (ok) {
    $unlockedProfile.set(key)
  }
  return ok
}

/** Forget the current unlock when the active profile changes away. */
export function noteProfileChange(nextProfileKey: string): void {
  const key = normalizeProfileKey(nextProfileKey)
  const unlocked = $unlockedProfile.get()
  if (unlocked !== null && unlocked !== key) {
    $unlockedProfile.set(null)
  }
}
