import { atom } from 'nanostores'

// Durable row ids of the user messages open in the inline edit composer. The
// edit draft lives in the thread runtime keyed by the message's renderer id,
// and a transcript refresh re-derives that id; the merge reads this set to keep
// the id of a row being edited, so the open draft survives the refresh.
export const $editingRowIds = atom<ReadonlyMap<number, number>>(new Map())

/** Mark `rowId` as being edited; returns the release. Counted: two panes may edit one row. */
export function holdEditingRow(rowId: number): () => void {
  const hold = (delta: 1 | -1) => {
    const next = new Map($editingRowIds.get())
    const count = (next.get(rowId) ?? 0) + delta

    if (count > 0) {
      next.set(rowId, count)
    } else {
      next.delete(rowId)
    }

    $editingRowIds.set(next)
  }

  hold(1)

  let released = false

  return () => {
    if (!released) {
      released = true
      hold(-1)
    }
  }
}
