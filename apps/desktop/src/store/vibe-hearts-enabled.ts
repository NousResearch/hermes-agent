/**
 * Floating vibe hearts (affection burst on ily / <3 / good bot / thanks).
 *
 * On by default: that is today's always-on behavior. Settings → Appearance owns
 * the lever so hearts stay presentation-scoped (desktop AGENTS.md: state lives
 * with its authority). Distinct from Message Reactions, which are iMessage-style
 * tapbacks on message rows.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.vibeHearts.v1'

// Absent key and anything other than "off" keep hearts on, matching the
// pre-toggle always-on default for existing installs.
export const $vibeHeartsEnabled = atom<boolean>(typeof window === 'undefined' ? true : storedString(KEY) !== 'off')

export function setVibeHeartsEnabled(enabled: boolean): void {
  $vibeHeartsEnabled.set(enabled)
}

if (typeof window !== 'undefined') {
  $vibeHeartsEnabled.listen(enabled => {
    persistString(KEY, enabled ? 'on' : 'off')
  })
}
