import { atom } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'
import { storedString } from '@/lib/storage'

export type ProfileSwitchBehavior = 'fresh_draft' | 'restore_last_session'

export interface ProfileSwitchRestoreIntent {
  connectionId?: null | string
  profile: string
  sequence: number
}

const STORAGE_KEY = 'hermes.desktop.profile-switch-behavior.v1'

function normalizeProfileSwitchBehavior(value: null | string): ProfileSwitchBehavior {
  return value === 'restore_last_session' ? value : 'fresh_draft'
}

const behaviorCodec: Codec<ProfileSwitchBehavior> = {
  decode: normalizeProfileSwitchBehavior,
  encode: value => value
}

export const $profileSwitchBehavior = persistentAtom<ProfileSwitchBehavior>(
  STORAGE_KEY,
  'fresh_draft',
  behaviorCodec
)

export const $profileSwitchRestoreIntent = atom<ProfileSwitchRestoreIntent | null>(null)

let restoreSequence = 0
let captureAnchor: null | (() => void) = null

export function getProfileSwitchBehavior(): ProfileSwitchBehavior {
  return $profileSwitchBehavior.get()
}

export function setProfileSwitchBehavior(value: ProfileSwitchBehavior): void {
  $profileSwitchBehavior.set(value)
}

export function requestProfileSwitchRestore(profile: string): ProfileSwitchRestoreIntent {
  captureAnchor?.()
  const intent = { profile, sequence: ++restoreSequence }
  $profileSwitchRestoreIntent.set(intent)

  return intent
}

/** Bind a pending profile-switch intent to the backend that actually activated. */
export function scopeProfileSwitchRestoreIntent(sequence: number, connectionId: null | string): void {
  const intent = $profileSwitchRestoreIntent.get()

  if (intent?.sequence === sequence) {
    $profileSwitchRestoreIntent.set({ ...intent, connectionId })
  }
}

/** Register the Desktop integration owner's synchronous visible-session capture. */
export function registerProfileSwitchAnchorCapture(capture: () => void): () => void {
  captureAnchor = capture

  return () => {
    if (captureAnchor === capture) {
      captureAnchor = null
    }
  }
}

export function clearProfileSwitchRestoreIntent(sequence: number): void {
  if ($profileSwitchRestoreIntent.get()?.sequence === sequence) {
    $profileSwitchRestoreIntent.set(null)
  }
}

/** @internal Reset client-local preference and pending intent state for tests. */
export function _resetProfileSwitchBehaviorForTests(): void {
  restoreSequence = 0
  captureAnchor = null
  $profileSwitchBehavior.set(typeof window === 'undefined' ? 'fresh_draft' : normalizeProfileSwitchBehavior(storedString(STORAGE_KEY)))
  $profileSwitchRestoreIntent.set(null)
}
