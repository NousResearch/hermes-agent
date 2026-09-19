import type { ThreadMessageLike } from '@assistant-ui/react'
import type { ToolCompletePayload, ToolStartPayload } from '@hermes/shared'

import type { ErrorSurface } from '@/lib/error-surface'
import type { ToolResultMetadata } from '@/lib/tool-result-metadata'
import type { MessageReaction, SessionMessage } from '@/types/hermes'

export interface TimelinePartMetadata {
  toolResultMetadata?: ToolResultMetadata
  /** Unix seconds when this visible activity segment began. Fractional values
   * preserve the millisecond precision available on live gateway events. */
  timestamp?: number
  /** Unix seconds when this segment stopped or handed off to the next one. */
  completedAt?: number
  /** Raw streamed text behind a `text` part whose MEDIA tags are already rendered,
   * so the next delta re-renders from the source instead of the render. */
  mediaSource?: string
}

export type ChatMessagePart = Exclude<ThreadMessageLike['content'], string>[number] & TimelinePartMetadata

export type ChatMessage = {
  id: string
  role: SessionMessage['role']
  parts: ChatMessagePart[]
  /** Result body only; the system text remains the compact completion label. */
  asyncResult?: string
  asyncResultKind?: 'process'
  timestamp?: number
  completedAt?: number
  pending?: boolean
  error?: string
  /** Structured layer descriptor for a failed turn (parsed error_surface).
   *  Drives the error card's layer label + actions; absent on older
   *  backends, where the card falls back to generic copy. */
  errorSurface?: ErrorSurface
  branchGroupId?: string
  hidden?: boolean
  /** Sealed mid-turn commentary (`message.interim`) — rendered without the
   *  action footer so only the turn's final reply carries copy/refresh, and
   *  the live view matches rehydration (which merges the turn into one bubble). */
  interim?: boolean
  /** Whole-turn wall-clock seconds (message.start → message.complete),
   *  stamped by the desktop when it watched the turn run. Absent for
   *  messages hydrated from history — the backend doesn't persist it. */
  durationS?: number
  /** Composer attachment ref strings (`@file:...`, `@image:...`) sent with this user message. */
  attachmentRefs?: string[]
  /** Durable backend `messages.id`. Absent until the row is persisted. */
  rowId?: number
  /** Emoji reactions on this message — one per author (see MessageReaction). */
  reactions?: MessageReaction[]
}

/** What a transcript tool row is built from: the wire's `tool.start` / `tool.complete`
 * payloads, plus the desktop's own restored blocking rows (clarify and connection cards),
 * which it shapes as a `ToolStartPayload`. */
export type ToolRowPayload = ToolCompletePayload | ToolStartPayload

// `context` and `result` are required (nullable) on their generated shapes and absent on the
// other, so key presence is the reliable discriminant; restored rows set `context: null` for this.
export function isToolStartPayload(payload: ToolRowPayload): payload is ToolStartPayload {
  return 'context' in payload
}

export function isToolCompletePayload(payload: ToolRowPayload): payload is ToolCompletePayload {
  return 'result' in payload
}
