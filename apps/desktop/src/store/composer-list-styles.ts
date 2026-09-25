/**
 * Which extra list styles the composer continues (Settings → Appearance).
 *
 * Numbered and bullet lists always continue; they are Markdown. Letters, Roman
 * numerals, outline numbers, `(1)` parentheses and symbol bullets are not, so
 * each is opt-in. Desktop-local: it changes how this window's keyboard behaves,
 * not what any other Hermes surface does.
 */

import { atom } from 'nanostores'

import { EXTRA_LIST_STYLES, type ExtraListStyle } from '@/lib/markdown-lists'
import { persistStringArray, storedStringArray } from '@/lib/storage'

const KEY = 'hermes.desktop.composerListStyles.v1'

const isStyle = (value: string): value is ExtraListStyle => (EXTRA_LIST_STYLES as readonly string[]).includes(value)

export const $composerListStyles = atom<ReadonlySet<ExtraListStyle>>(
  new Set(typeof window === 'undefined' ? [] : storedStringArray(KEY).filter(isStyle))
)

export function setComposerListStyle(style: ExtraListStyle, enabled: boolean): void {
  const next = new Set($composerListStyles.get())

  if (enabled) {
    next.add(style)
  } else {
    next.delete(style)
  }

  $composerListStyles.set(next)
}

if (typeof window !== 'undefined') {
  $composerListStyles.listen(styles => {
    persistStringArray(
      KEY,
      EXTRA_LIST_STYLES.filter(style => styles.has(style))
    )
  })
}
