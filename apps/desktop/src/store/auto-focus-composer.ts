/**
 * Auto-focus the chat composer when the window regains focus (#38993).
 *
 * A device-local presentation preference, off by default: coming back to
 * Hermes (Alt+Tab, dock click) normally leaves the caret wherever the last
 * click landed, so the first typed character is lost. The renderer owns this
 * flag — it says how THIS window presents, nothing another Hermes surface can
 * change (desktop AGENTS.md: state lives with its authority).
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.autoFocusComposer.v1'

export const $autoFocusComposer = atom<boolean>(typeof window === 'undefined' ? false : storedBoolean(KEY, false))

export function setAutoFocusComposer(on: boolean): void {
  $autoFocusComposer.set(on)
}

if (typeof window !== 'undefined') {
  $autoFocusComposer.listen(on => persistBoolean(KEY, on))
}
