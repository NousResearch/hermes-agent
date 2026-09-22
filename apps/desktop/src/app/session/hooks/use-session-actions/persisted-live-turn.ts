import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { assistantTextPart, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { withoutCoveredAssistantPrefix } from '@/lib/chat-messages/coverage'
import { parseErrorSurface } from '@/lib/error-surface'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { mergeLiveAssistantRun } from './live-turn-remainder'

const rowId = (row: SessionMessage) => row.row_id ?? row.id
const userText = (text: string) => textWithoutReferenceLines(text).trim()
const hasTools = (row: SessionMessage) => Array.isArray(row.tool_calls) && row.tool_calls.length > 0

const renderedText = (raw: string) => {
  const part = assistantTextPart(raw)

  return part.type === 'text' ? part.text : ''
}

interface PersistedTurn {
  prompt: ChatMessage
  start: number
  rawIntervals: string[]
  messages: ChatMessage[]
}

/** Validate the current turn's source rows against the occurrences actually painted. */
function candidateTurn(
  rows: SessionMessage[],
  messages: ChatMessage[],
  start: number,
  corrections: string[]
): PersistedTurn | null {
  const promptId = rowId(rows[start])
  const displayStart = messages.findIndex(message => message.role === 'user' && message.rowId === promptId)

  if (promptId === undefined || displayStart < 0) {
    return null
  }

  const turnMessages = messages.slice(displayStart + 1)
  const intervals: string[][] = [[]]

  for (const row of rows.slice(start + 1)) {
    if (row.role === 'user') {
      if (userText(String(row.content ?? '')) !== userText(corrections[intervals.length - 1] ?? '')) {
        return null
      }

      const id = rowId(row)

      if (id === undefined || !turnMessages.some(message => message.role === 'user' && message.rowId === id)) {
        return null
      }

      intervals.push([])
    } else if (row.role === 'assistant') {
      if (!hasTools(row) || typeof row.content !== 'string' || row.display_kind === 'hidden') {
        return null
      }

      const raw = row.content

      if (!raw.trim()) {
        continue
      }

      const id = rowId(row)

      // MEDIA rendering and folding change lengths. Only source-row provenance,
      // not equal prose elsewhere in the transcript, proves display coverage.
      if (
        id === undefined ||
        !turnMessages.some(message =>
          message.parts.some(
            part => part.type === 'text' && part.sourceRowId === id && part.text.trim() === renderedText(raw).trim()
          )
        )
      ) {
        return null
      }

      intervals.at(-1)!.push(raw)
    } else if (row.role !== 'tool') {
      return null
    }
  }

  return {
    prompt: messages[displayStart],
    start: displayStart,
    rawIntervals: intervals.map(texts => texts.join('\n\n')),
    messages: turnMessages
  }
}

function locateTurn(
  rows: SessionMessage[],
  messages: ChatMessage[],
  inflight: NonNullable<SessionResumeResult['inflight']>
): PersistedTurn | null {
  let candidate: PersistedTurn | null = null

  for (let index = rows.length - 1; index >= 0; index--) {
    const row = rows[index]

    // A final reply is an actual boundary; durable tool commentary is not.
    if (row.role === 'assistant' && !hasTools(row) && typeof row.content === 'string' && row.content.trim()) {
      break
    }

    if (
      row.role === 'user' &&
      row.display_kind !== 'steer' &&
      userText(String(row.content ?? '')) === userText(inflight.user ?? '')
    ) {
      candidate = candidateTurn(rows, messages, index, inflight.corrections ?? []) ?? candidate
    }
  }

  return candidate
}

function splitAtUsers(messages: ChatMessage[]) {
  const intervals: ChatMessage[][] = [[]]
  const users: ChatMessage[] = []

  for (const message of messages) {
    if (message.role === 'user') {
      users.push(message)
      intervals.push([])
    } else {
      intervals.at(-1)!.push(message)
    }
  }

  return { intervals, users }
}

/** Prefix races are symmetric; cuts address raw producer text, never rendered Markdown. */
function unrepresentedText(snapshot: string, stored: string): string {
  // Separators between rounds can lead an interval after a correction.
  const leading = snapshot.length - snapshot.trimStart().length
  const text = snapshot.slice(leading)
  const prefix = stored.trim()

  if (!prefix) {
    return snapshot
  }

  if (text.startsWith(prefix)) {
    return text.slice(prefix.length)
  }

  if (prefix.startsWith(text)) {
    return ''
  }

  return snapshot
}

function snapshotIntervals(inflight: NonNullable<SessionResumeResult['inflight']>): string[] {
  const text = inflight.assistant ?? ''
  const corrections = inflight.corrections ?? []
  const offsets = inflight.correction_offsets
  // Python's len() counts Unicode code points, unlike JS string offsets.
  const characters = Array.from(text)

  const usable =
    offsets?.length === corrections.length &&
    offsets.every(
      (offset, index) =>
        Number.isInteger(offset) &&
        offset >= 0 &&
        offset <= characters.length &&
        (!index || offset >= offsets[index - 1])
    )

  if (!usable) {
    return [text, ...corrections.map(() => '')]
  }

  return [...offsets, characters.length].map((end, index) =>
    characters.slice(index ? offsets[index - 1] : 0, end).join('')
  )
}

/**
 * Reconcile the anchored partial turn before role-ordinal pairing sees it.
 * Missing provenance takes the legacy path without subtracting any snapshot text.
 */
export function reconcilePersistedLiveTurn(
  messages: ChatMessage[],
  previous: ChatMessage[],
  rows: SessionMessage[],
  projection: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'>,
  reconcileHistory: (messages: ChatMessage[], previous: ChatMessage[]) => ChatMessage[]
): ChatMessage[] | null {
  const inflight = projection.inflight

  if (!inflight?.user) {
    return null
  }

  const turn = locateTurn(rows, messages, inflight)

  if (!turn) {
    return null
  }

  let localStart = previous.findIndex(message => message.role === 'user' && message.rowId === turn.prompt.rowId)

  if (localStart < 0) {
    const tools = new Set(
      turn.messages.flatMap(message =>
        message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
      )
    )

    // Optimistic prompt rows have no durable id yet. Shared tool identity after
    // the matching prompt can anchor the turn, never the prompt text alone.
    localStart = previous.findLastIndex(
      (message, index) =>
        message.role === 'user' &&
        userText(chatMessageText(message)) === userText(chatMessageText(turn.prompt)) &&
        previous
          .slice(index + 1)
          .some(row => row.parts.some(part => part.type === 'tool-call' && tools.has(part.toolCallId)))
    )
  }

  const stored = splitAtUsers(turn.messages)
  const cached = splitAtUsers(localStart >= 0 ? previous.slice(localStart + 1) : [])
  const snapshots = snapshotIntervals(inflight)
  const corrections = inflight.corrections ?? []

  const result = reconcileHistory(
    messages.slice(0, turn.start + 1),
    localStart >= 0 ? previous.slice(0, localStart + 1) : previous
  )

  const turnStart = result.length

  let pairedLocal = localStart >= 0
  let unpairedLocal: ChatMessage[] = []

  for (let index = 0; index < snapshots.length; index++) {
    const durable = stored.intervals[index] ?? []
    const raw = turn.rawIntervals[index] ?? ''
    // No usable correction offsets means only the complete initial prefix can
    // be proven; don't apply a later interval's normalized length to this dump.
    const text = unrepresentedText(snapshots[index], raw)

    const final = index === snapshots.length - 1
    const error = final ? inflight.error?.trim() : undefined

    const projected: ChatMessage[] =
      text.trim() || final
        ? [
            {
              id: final
                ? `assistant-stream-${projection.session_id}`
                : `inflight-assistant-segment-${index}-${projection.session_id}`,
              role: 'assistant',
              parts: text.trim() ? [assistantTextPart(text)] : [],
              pending: final && Boolean(inflight.streaming),
              ...(!final ? { interim: true } : {}),
              ...(error ? { error, errorSurface: parseErrorSurface(inflight.error_surface) ?? undefined } : {})
            }
          ]
        : []

    const local = pairedLocal ? withoutCoveredAssistantPrefix(durable, cached.intervals[index] ?? []) : []
    result.push(...durable, ...mergeLiveAssistantRun(projected, local))

    if (!final) {
      const correction = stored.users[index] ?? {
        id: `user-inflight-correction-${index}-${projection.session_id}`,
        role: 'user' as const,
        parts: [textPart(corrections[index])]
      }

      result.push(correction)

      if (
        pairedLocal &&
        (!cached.users[index] || userText(chatMessageText(cached.users[index])) !== userText(corrections[index]))
      ) {
        unpairedLocal = cached.users
          .slice(index)
          .flatMap((user, offset) => [user, ...cached.intervals[index + offset + 1]])
        pairedLocal = false
      }
    }
  }

  if (pairedLocal) {
    result.push(
      ...cached.users
        .slice(corrections.length)
        .flatMap((user, index) => [user, ...cached.intervals[corrections.length + index + 1]])
    )
  }

  result.push(...unpairedLocal)

  if (projection.queued?.user) {
    const id = `user-queued-${projection.session_id}`

    const queuedIndex = result.findIndex(
      (message, index) => index >= turnStart && message.role === 'user' && message.id === id
    )

    const parts = [textPart(projection.queued.user)]

    // The anchored turn has one next-turn queue slot. Refresh its projection
    // in place; equal correction/optimistic user text is a different occurrence.
    if (queuedIndex >= 0) {
      result[queuedIndex] = { ...result[queuedIndex], parts }
    } else {
      result.push({ id, role: 'user', parts })
    }
  }

  return result
}
