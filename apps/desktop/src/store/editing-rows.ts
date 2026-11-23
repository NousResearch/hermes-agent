import { atom, type WritableAtom } from 'nanostores'

// Durable row ids of the user messages open in the inline edit composer. The
// edit draft lives in the thread runtime keyed by the message's renderer id,
// and a transcript refresh re-derives that id; the merge reads this set to keep
// the id of a row being edited, so the open draft survives the refresh.
export const $editingRowIds = atom<ReadonlyMap<number, number>>(new Map())

// Renderer ids of user messages that were still unstored (no durable row id)
// when their edit composer opened. Their React key stays the live id while the
// composer is open: binding the stored row id on Stop would otherwise flip the
// key to `row:<id>` and remount the editor (#145872).
export const $liveKeyIds = atom<ReadonlyMap<string, number>>(new Map())

function hold<K>(store: WritableAtom<ReadonlyMap<K, number>>, key: K): () => void {
  const step = (delta: 1 | -1) => {
    const next = new Map(store.get())
    const count = (next.get(key) ?? 0) + delta

    if (count > 0) {
      next.set(key, count)
    } else {
      next.delete(key)
    }

    store.set(next)
  }

  step(1)

  let released = false

  return () => {
    if (!released) {
      released = true
      step(-1)
    }
  }
}

/** Mark `rowId` as being edited; returns the release. Counted: two panes may edit one row. */
export const holdEditingRow = (rowId: number) => hold($editingRowIds, rowId)

/** Keep the live renderer id as the React key of `messageId`; returns the release. */
export const holdLiveKey = (messageId: string) => hold($liveKeyIds, messageId)
