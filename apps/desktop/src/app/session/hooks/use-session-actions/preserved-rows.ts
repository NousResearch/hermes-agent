import { type ChatMessage, spliceOlderPreservedRows } from '@/lib/chat-messages'
import { $editingRowIds } from '@/store/editing-rows'

/** The preserved local rows merged into the refreshed ones, with the id of every row open in the
 *  inline edit composer kept as it was. */
export function mergePreservedRows(
  withReplacements: ChatMessage[],
  preserved: ChatMessage[],
  previousMessages: ChatMessage[]
): ChatMessage[] {
  // #120978: a kept run whose rowIds predate the whole hydrated page belongs
  // earlier — splice it in front of the first newer row instead of appending it
  // below the newest turn (non-qualifying runs keep the trailing behavior).
  return keepEditedRowIds(
    preserved.length ? spliceOlderPreservedRows(withReplacements, preserved) : withReplacements,
    previousMessages
  )
}

/**
 * A user row open in the inline edit composer keeps the id it had. The
 * runtime holds the edit draft under that id, and a stored re-read re-derives
 * it (`user-<uuid>` of the live send, or a page-index id) for the very same
 * durable row — which closes the edit box and discards what was typed (#145872).
 * Only rows being edited are touched; every other row takes the refreshed id.
 */
function keepEditedRowIds(merged: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  const editing = $editingRowIds.get()

  if (editing.size === 0) {
    return merged
  }

  const idByRow = new Map<number, string>()

  for (const message of previous) {
    if (message.role === 'user' && message.rowId !== undefined && editing.has(message.rowId)) {
      idByRow.set(message.rowId, message.id)
    }
  }

  const taken = new Set(merged.map(message => message.id))
  let changed = false

  const kept = merged.map(message => {
    const id = message.role === 'user' && message.rowId !== undefined ? idByRow.get(message.rowId) : undefined

    if (!id || id === message.id || taken.has(id)) {
      return message
    }

    taken.add(id)
    changed = true

    return { ...message, id }
  })

  return changed ? kept : merged
}
