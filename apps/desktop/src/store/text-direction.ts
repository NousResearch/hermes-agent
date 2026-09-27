import { useStore } from '@nanostores/react'

import { type Codec, persistentAtom } from '@/lib/persisted'

/**
 * How chat prose and the composers pick their base direction.
 *
 * - `auto`: each block resolves its own direction from the mixed-script
 *   resolver (`resolveTextDirection`), not the browser's first-strong vote.
 *   Forced RTL/LTR still win when the reader picks them.
 * - `rtl` / `ltr`: the reader's explicit choice. Code (fenced and inline) and
 *   KaTeX stay isolated LTR either way.
 */
export type TextDirection = 'auto' | 'ltr' | 'rtl'

export const TEXT_DIRECTIONS = ['auto', 'rtl', 'ltr'] as const satisfies readonly TextDirection[]

// Scope: global to this desktop install (every window and profile). It is a
// reading preference about the person at the keyboard, not about a backend,
// profile or session, so no connection/profile segment belongs in the key.
const STORAGE_KEY = 'hermes.desktop.textDirection'

// Auto is stored as absence, so users who never touch the setting keep an
// untouched storage record as well as today's rendering.
const textDirectionCodec: Codec<TextDirection> = {
  decode: raw => (raw === 'rtl' || raw === 'ltr' ? raw : 'auto'),
  encode: value => (value === 'auto' ? null : value)
}

export const $textDirection = persistentAtom<TextDirection>(STORAGE_KEY, 'auto', textDirectionCodec)

export function setTextDirection(direction: TextDirection) {
  $textDirection.set(direction)
}

/** The `dir` value to stamp on chat prose/composers; undefined keeps Auto attribute-free. */
export function forcedTextDirection(direction: TextDirection): 'ltr' | 'rtl' | undefined {
  return direction === 'auto' ? undefined : direction
}

export function useForcedTextDirection(): 'ltr' | 'rtl' | undefined {
  return forcedTextDirection(useStore($textDirection))
}
