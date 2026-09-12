import { atom } from 'nanostores'

/**
 * One shared document-level `selectionchange` listener feeding every
 * subscription; each row/interested leaf intersects the current selection
 * with its own subtree locally — the alternative (one listener per row)
 * fires N handlers + N React state updaters on every caret move in the
 * composer (each keystroke), and reads `window.getSelection()` N times.
 *
 * The value IS the current document selection (or null) — subscribers that
 * need "does this selection intersect me" run that check against their own
 * host; the listener itself stays out of per-row business.
 */
export const $documentSelection = atom<Selection | null>(null)

let listenerCount = 0

function publish(): void {
  const next = window.getSelection()

  // A selection object identity changes on every selectionchange even when
  // the highlight is unchanged; nanostores set() with an identical value
  // is a no-op for subscribers, so only publish when the reference moved.
  if ($documentSelection.get() !== next) {
    $documentSelection.set(next)
  }
}

/**
 * Subscribe to the live document selection. The listener is attached lazily
 * on the first subscriber and removed when the last unsubscribes — a fresh
 * chat that mounts zero message rows pays zero listener cost.
 */
export function subscribeToDocumentSelection(handler: (selection: Selection | null) => void): () => void {
  if (listenerCount === 0 && typeof document !== 'undefined') {
    document.addEventListener('selectionchange', publish)
  }
  listenerCount++

  const unsubscribeFromAtom = $documentSelection.listen(value => {
    handler(value)
  })

  return () => {
    unsubscribeFromAtom()
    listenerCount--
    if (listenerCount === 0 && typeof document !== 'undefined') {
      document.removeEventListener('selectionchange', publish)
    }
  }
}
