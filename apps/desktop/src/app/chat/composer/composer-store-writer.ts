import { SLASH_COMMAND_RE } from '@hermes/shared'

/**
 * Every fact the app derives from the assistant-ui composer text through
 * `useAuiState` (send-button state, the `?` help sigil, steer vs slash, the
 * empty placeholder, hard-newline layout). A keystroke that leaves this
 * signature unchanged is invisible to every one of those selectors, so its
 * store write can wait. Add a field here when a new `s.composer.text`
 * selector appears.
 */
export function composerTextEdges(text: string): string {
  const trimmed = text.trim()

  return [
    text.length === 0,
    trimmed.length > 0,
    text === '?',
    trimmed.length > 0 && !SLASH_COMMAND_RE.test(trimmed),
    text.trimEnd().includes('\n')
  ].join('|')
}

// Typing pause after which the trailing text lands in the store (draft
// persistence and the runtime mirror read it from there).
export const TYPING_STORE_IDLE_MS = 180

export interface ComposerStoreWriter {
  /** Write now, superseding any pending typed text. */
  now: (text: string) => void
  /** Per-keystroke path: write only when an edge flips, else after a typing pause. */
  typing: (text: string) => void
  /** Land pending typed text immediately (before a read that needs the store current). */
  flush: () => void
  /** Drop pending typed text without writing it (unmount). */
  cancel: () => void
  /** True while typed text the store has not seen is waiting. */
  pending: () => boolean
  /** The store changed under us (runtime subscription): adopt it as the written baseline. */
  observed: (text: string) => void
}

/**
 * Each assistant-ui composer write re-renders the thread's client resources
 * and re-checks every `useAuiState` selector of every mounted message, so its
 * cost grows with the transcript (live: 19 ms of main-thread time per
 * keystroke once a long session's history is mounted, vs 0.5 ms on a short
 * one). Keystrokes go through `typing`, which coalesces a burst into one write.
 */
export function createComposerStoreWriter(
  write: (text: string) => void,
  idleMs = TYPING_STORE_IDLE_MS
): ComposerStoreWriter {
  let written = ''
  let pendingText: string | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  const clear = () => {
    if (timer !== undefined) {
      clearTimeout(timer)
      timer = undefined
    }

    pendingText = null
  }

  const now = (text: string) => {
    clear()
    written = text
    write(text)
  }

  const flush = () => {
    if (pendingText !== null) {
      now(pendingText)
    }
  }

  const typing = (text: string) => {
    if (composerTextEdges(text) !== composerTextEdges(written)) {
      now(text)

      return
    }

    pendingText = text

    if (timer !== undefined) {
      clearTimeout(timer)
    }

    timer = setTimeout(flush, idleMs)
  }

  const observed = (text: string) => {
    if (pendingText === null) {
      written = text
    }
  }

  return { cancel: clear, flush, now, observed, pending: () => pendingText !== null, typing }
}
