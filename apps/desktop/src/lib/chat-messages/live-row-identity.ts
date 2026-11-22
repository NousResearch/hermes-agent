import { isLiveTailReplyId } from '@/lib/spoken-reply'

import type { ChatMessage, ChatMessagePart } from './types'

/**
 * The stored row a live `message.interim`, `tool.start`, `tool.complete` or `message.complete`
 * names (`row_id`, `message_uid`; agent/live_row_identity.py): the backend names a row only once
 * it is durable, so the part drawn from the event is that row's copy, as a hydrated part is the
 * copy of the row it came from (`sourceRowId` / `sourceCallRowId`, `sourceMessageUid`). Older
 * backends name none, and the part stays unaddressed.
 *
 * `message_uid` decides when both sides carry it; the row id stands in only when one side has
 * no uid, because a compaction during the turn re-issues row ids and keeps uids.
 */
export interface LiveRowIdentity {
  rowId?: number
  messageUid?: string
}

export function liveRowIdentity(payload: unknown): LiveRowIdentity {
  const record = (payload ?? {}) as Record<string, unknown>
  const rowId = record.row_id
  const uid = record.message_uid

  return {
    ...(typeof rowId === 'number' && Number.isSafeInteger(rowId) && rowId > 0 ? { rowId } : {}),
    ...(typeof uid === 'string' && uid ? { messageUid: uid } : {})
  }
}

/** The part fields that address `identity`'s row, for a text part or a tool call. */
export function storedRowFields(
  type: 'text' | 'tool-call',
  identity: LiveRowIdentity
): Pick<ChatMessagePart, 'sourceCallRowId' | 'sourceMessageUid' | 'sourceRowId'> {
  const row = identity.rowId

  return {
    ...(row === undefined ? {} : type === 'text' ? { sourceRowId: row } : { sourceCallRowId: row }),
    ...(identity.messageUid === undefined ? {} : { sourceMessageUid: identity.messageUid })
  }
}

type ToolCallPart = Extract<ChatMessagePart, { type: 'tool-call' }>

export interface PartAt {
  at: number
  part: number
}

/** The id a stored call's live card carries: the provider's, when hydration renamed a repeat. */
const providerIdOf = (part: ToolCallPart) => part.providerToolCallId ?? part.toolCallId

/** Lookup keys of the stored row a text part or tool call names: its uid first, then its row id. */
function storedKeys(part: ChatMessagePart): string[] {
  if (part.type !== 'text' && part.type !== 'tool-call') {
    return []
  }

  const piece = part.type === 'text' ? 't' : `c:${providerIdOf(part)}`
  const row = part.type === 'text' ? part.sourceRowId : part.sourceCallRowId

  return [
    ...(part.sourceMessageUid ? [`${piece}|uid:${part.sourceMessageUid}`] : []),
    ...(row !== undefined ? [`${piece}|row:${row}`] : [])
  ]
}

/** Where each stored text part and tool call of `messages` sits, by the row it names. */
export interface StoredParts {
  size: number
  /** The stored piece `part` is a copy of, if any. */
  copyOf: (part: ChatMessagePart) => PartAt | undefined
}

export function indexStoredParts(messages: ChatMessage[]): StoredParts {
  const index = new Map<string, PartAt>()

  messages.forEach((message, at) => {
    if (message.role !== 'assistant') {
      return
    }

    message.parts.forEach((part, partIndex) => {
      for (const key of storedKeys(part)) {
        if (!index.has(key)) {
          index.set(key, { at, part: partIndex })
        }
      }
    })
  })

  const copyOf = (part: ChatMessagePart) => {
    const [first, second] = storedKeys(part)
    const byFirst = first ? index.get(first) : undefined

    if (byFirst || !part.sourceMessageUid || !second) {
      return byFirst
    }

    // The part has a uid the stored side lacks: only a stored row without a uid may match by id.
    const byRow = index.get(second)

    return byRow && !messages[byRow.at].parts[byRow.part].sourceMessageUid ? byRow : undefined
  }

  return { size: index.size, copyOf }
}

/**
 * An assistant row the stream drew, not one an earlier history read hydrated: it has no stored row
 * of its own, or it is the live tail a completion receipt bound to its final row.
 */
export const isLiveCopy = (message: ChatMessage) =>
  message.role === 'assistant' && (message.rowId === undefined || isLiveTailReplyId(message.id))

/** A live row whose own state the page cannot carry: its error card, stopped mark or open stream. */
export const keepsItsLiveState = (message: ChatMessage) =>
  Boolean(message.pending || message.error || message.interrupted)

/** Every reply part of live row `message` has a stored copy in `stored` (reasoning alone is no reply). */
export function storedWhole(stored: StoredParts, message: ChatMessage): boolean {
  const replies = message.parts.filter(part => part.type !== 'reasoning')

  return (
    isLiveCopy(message) &&
    !keepsItsLiveState(message) &&
    replies.length > 0 &&
    replies.every(part => stored.copyOf(part) !== undefined)
  )
}

/**
 * A row only this window holds: an optimistic prompt, or a live assistant row (streaming, or the
 * settled tail the backend has not committed) unless every piece of it names a stored row the page
 * paints. Such a row is that page's copy, under whatever row
 * ids a compaction since gave those rows.
 */
export function isLocalRowToKeep(
  message: ChatMessage,
  storedParts: StoredParts,
  isOptimisticUser: boolean,
  isLiveAssistant: boolean
): boolean {
  return isOptimisticUser || (isLiveAssistant && !storedWhole(storedParts, message))
}

/** A stored call slot the page has no result for yet takes the live card's result. */
function withResultOf(stored: ToolCallPart, live: ToolCallPart): ToolCallPart {
  if (stored.result !== undefined || live.result === undefined) {
    return stored
  }

  return {
    ...stored,
    result: live.result,
    ...(live.isError !== undefined ? { isError: live.isError } : {}),
    ...(live.artifact !== undefined ? { artifact: live.artifact } : {}),
    ...(live.completedAt !== undefined ? { completedAt: live.completedAt } : {}),
    ...(live.toolResultMetadata !== undefined ? { toolResultMetadata: live.toolResultMetadata } : {})
  }
}

/**
 * `page` with the results only the window's cards had: a stored call the page holds before its result
 * row is stored takes the result its live card (or a cached fold from an earlier read) already drew,
 * so the card that gives way to the page's copy does not take the result with it.
 */
export function lendLiveResults(page: ChatMessage[], window: ChatMessage[]): ChatMessage[] {
  const stored = indexStoredParts(page)
  const patches = new Map<string, ToolCallPart>()

  for (const part of window.flatMap(message => (message.role === 'assistant' ? message.parts : []))) {
    const copy = part.type === 'tool-call' ? stored.copyOf(part) : undefined
    const slot = copy && page[copy.at].parts[copy.part]

    if (part.type === 'tool-call' && copy && slot?.type === 'tool-call') {
      const key = `${copy.at}:${copy.part}`
      const filled = withResultOf(patches.get(key) ?? slot, part)

      if (filled !== slot) {
        patches.set(key, filled)
      }
    }
  }

  return patches.size
    ? page.map((message, at) =>
        message.parts.some((_, index) => patches.has(`${at}:${index}`))
          ? { ...message, parts: message.parts.map((part, index) => patches.get(`${at}:${index}`) ?? part) }
          : message
      )
    : page
}
