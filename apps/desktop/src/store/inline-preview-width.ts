import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const INLINE_PREVIEW_WIDE_STORAGE_KEY = 'hermes.desktop.inlinePreview.wide'

/**
 * Whether an HTML reply (`::preview{file=…}`) fills the conversation column.
 *
 * On by default: an HTML panel is a document, not an image, so it should use
 * the width the transcript already gives it. Off restores the old 40rem
 * reading-column cap, which keeps a dense panel beside prose instead of
 * spanning it.
 */
const $inlinePreviewWidePref = atom(storedBoolean(INLINE_PREVIEW_WIDE_STORAGE_KEY, true))

$inlinePreviewWidePref.subscribe(wide => persistBoolean(INLINE_PREVIEW_WIDE_STORAGE_KEY, wide))

export const $inlinePreviewWide = $inlinePreviewWidePref

export function setInlinePreviewWide(wide: boolean) {
  $inlinePreviewWidePref.set(wide)
}
