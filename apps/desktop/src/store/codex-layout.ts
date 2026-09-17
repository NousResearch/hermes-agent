/**
 * Codex-style conversation layout.
 *
 * On by default: assistant transcripts use a centered reading column and
 * fenced code blocks occupy the available width. Settings → Appearance owns
 * the lever so the layout stays presentation-scoped (desktop AGENTS.md: state
 * lives with its authority).
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.codexLayout.v1'

// Absent key and anything other than "off" keep the Codex layout on, matching
// the post-change default for existing installs.
export const $codexLayout = atom<boolean>(typeof window === 'undefined' ? true : storedString(KEY) !== 'off')

export function setCodexLayout(enabled: boolean): void {
  $codexLayout.set(enabled)
}

if (typeof window !== 'undefined') {
  $codexLayout.listen(enabled => {
    persistString(KEY, enabled ? 'on' : 'off')
  })
}
