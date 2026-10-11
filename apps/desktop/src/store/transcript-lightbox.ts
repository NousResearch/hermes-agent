import { atom } from 'nanostores'

import { $activeSessionId } from '@/store/session'

/**
 * The transcript lightbox's open source, hoisted out of `ZoomableImage`.
 *
 * Transcript rows — and the markdown leaves inside them — legitimately remount
 * while a turn streams (the render-budget slice drops older rows; the markdown
 * AST re-parses as text grows), and the dialog's open flag used to be
 * component-local `useState`. A user-opened preview therefore closed itself
 * with no gesture the moment its row recycled (#123018). Keying the flag by
 * source identity here lets a remounted row re-present its own open dialog.
 *
 * One preview at a time (same as the component-local flag ever allowed),
 * process-memory only, cleared by an explicit close (Escape, backdrop, or
 * click) — never by a remount — and on a session switch, so a new transcript
 * never inherits the previous chat's open preview.
 */
export const $transcriptLightbox = atom<string | null>(null)

// A stale open entry must not follow the user into another chat: the same
// media path can legitimately appear in two sessions, and a lightbox opening
// with no gesture there would be the bug again, inverted.
$activeSessionId.subscribe(() => resetTranscriptLightbox())

export function openTranscriptLightbox(src: string): void {
  if (src) {
    $transcriptLightbox.set(src)
  }
}

/** Identity-guarded: only the row owning the currently-open source clears it. */
export function closeTranscriptLightbox(src: string): void {
  if ($transcriptLightbox.get() === src) {
    $transcriptLightbox.set(null)
  }
}

/** Mount/test isolation: drop any open entry without a gesture. */
export function resetTranscriptLightbox(): void {
  $transcriptLightbox.set(null)
}
