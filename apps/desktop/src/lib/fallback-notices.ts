import type { ChatMessage } from '@/lib/chat-messages'
import { textPart } from '@/lib/chat-messages'
import { readKey, writeKey } from '@/lib/storage'

export const FALLBACK_NOTICE_STORAGE_KEY = 'hermes.desktop.fallbackNotices.v1'

const FALLBACK_NOTICE_ID_PREFIX = 'fallback-status-'
const MAX_NOTICES_PER_SESSION = 40
const MAX_SESSIONS = 100
const MAX_NOTICE_TEXT_LENGTH = 1_000

let fallbackNoticeSequence = 0

interface PersistedFallbackNotice {
  beforeMessageCount: number
  id: string
  text: string
  timestamp: number
}

interface PersistedSessionFallbackNotices {
  notices: PersistedFallbackNotice[]
  updatedAt: number
}

type PersistedFallbackNoticeRegistry = Record<string, PersistedSessionFallbackNotices>

interface PendingFallbackNotice {
  beforeMessageCount: number
  message: ChatMessage
}

const pendingFallbackNoticesByRuntimeId = new Map<string, PendingFallbackNotice[]>()

function fallbackNoticeMessage(notice: PersistedFallbackNotice): ChatMessage {
  return {
    id: notice.id,
    role: 'system',
    parts: [textPart(notice.text)],
    timestamp: notice.timestamp
  }
}

function parseNotice(value: unknown): PersistedFallbackNotice | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const row = value as Record<string, unknown>

  if (
    typeof row.id !== 'string' ||
    !row.id.startsWith(FALLBACK_NOTICE_ID_PREFIX) ||
    typeof row.text !== 'string' ||
    !row.text.trim() ||
    typeof row.timestamp !== 'number' ||
    !Number.isFinite(row.timestamp) ||
    typeof row.beforeMessageCount !== 'number' ||
    !Number.isFinite(row.beforeMessageCount)
  ) {
    return null
  }

  return {
    beforeMessageCount: Math.max(0, Math.floor(row.beforeMessageCount)),
    id: row.id,
    text: row.text.trim().slice(0, MAX_NOTICE_TEXT_LENGTH),
    timestamp: row.timestamp
  }
}

function readRegistry(): PersistedFallbackNoticeRegistry {
  const raw = readKey(FALLBACK_NOTICE_STORAGE_KEY)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    const entries = Object.entries(parsed as Record<string, unknown>).flatMap(([sessionId, value]) => {
      if (!sessionId || !value || typeof value !== 'object' || Array.isArray(value)) {
        return []
      }

      const row = value as Record<string, unknown>

      const notices = Array.isArray(row.notices)
        ? row.notices.map(parseNotice).filter((notice): notice is PersistedFallbackNotice => notice !== null)
        : []

      const updatedAt = typeof row.updatedAt === 'number' && Number.isFinite(row.updatedAt) ? row.updatedAt : 0

      return notices.length ? [[sessionId, { notices, updatedAt }] as const] : []
    })

    return Object.fromEntries(entries)
  } catch {
    return {}
  }
}

function writeRegistry(registry: PersistedFallbackNoticeRegistry) {
  const bounded = Object.fromEntries(
    Object.entries(registry)
      .sort((left, right) => right[1].updatedAt - left[1].updatedAt)
      .slice(0, MAX_SESSIONS)
  )

  writeKey(FALLBACK_NOTICE_STORAGE_KEY, Object.keys(bounded).length ? JSON.stringify(bounded) : null)
}

export function createFallbackNotice(text: string, nowMs = Date.now()): ChatMessage {
  fallbackNoticeSequence += 1

  return {
    id: `${FALLBACK_NOTICE_ID_PREFIX}${nowMs}-${fallbackNoticeSequence}`,
    role: 'system',
    parts: [textPart(text.trim().slice(0, MAX_NOTICE_TEXT_LENGTH))],
    timestamp: nowMs / 1_000
  }
}

export function deferFallbackNotice(runtimeSessionId: string, message: ChatMessage, beforeMessageCount: number) {
  const runtimeId = runtimeSessionId.trim()

  if (!runtimeId || !isFallbackNoticeMessage(message)) {
    return
  }

  const pending = pendingFallbackNoticesByRuntimeId.get(runtimeId) ?? []
  pendingFallbackNoticesByRuntimeId.set(runtimeId, [
    ...pending.filter(item => item.message.id !== message.id),
    { beforeMessageCount, message }
  ])
}

export function flushPendingFallbackNotices(runtimeSessionId: string, storedSessionId: string) {
  const runtimeId = runtimeSessionId.trim()
  const storedId = storedSessionId.trim()
  const pending = pendingFallbackNoticesByRuntimeId.get(runtimeId)

  if (!runtimeId || !storedId || !pending?.length) {
    return
  }

  for (const item of pending) {
    persistFallbackNotice(storedId, item.message, item.beforeMessageCount)
  }

  pendingFallbackNoticesByRuntimeId.delete(runtimeId)
}

export function isFallbackNoticeMessage(message: ChatMessage): boolean {
  return message.role === 'system' && message.id.startsWith(FALLBACK_NOTICE_ID_PREFIX)
}

export function countTranscriptMessages(messages: readonly ChatMessage[]): number {
  return messages.reduce(
    (count, message) => count + (message.role === 'user' || message.role === 'assistant' ? 1 : 0),
    0
  )
}

export function persistFallbackNotice(
  storedSessionId: string,
  message: ChatMessage,
  beforeMessageCount: number,
  nowMs = Date.now()
) {
  const sessionId = storedSessionId.trim()

  const text = message.parts
    .filter(part => part.type === 'text')
    .map(part => part.text)
    .join('')
    .trim()

  if (!sessionId || !isFallbackNoticeMessage(message) || !text) {
    return
  }

  const registry = readRegistry()
  const previous = registry[sessionId]?.notices ?? []

  const notice: PersistedFallbackNotice = {
    beforeMessageCount: Math.max(0, Math.floor(beforeMessageCount)),
    id: message.id,
    text: text.slice(0, MAX_NOTICE_TEXT_LENGTH),
    timestamp: message.timestamp ?? nowMs / 1_000
  }

  registry[sessionId] = {
    notices: [...previous.filter(item => item.id !== notice.id), notice].slice(-MAX_NOTICES_PER_SESSION),
    updatedAt: nowMs
  }
  writeRegistry(registry)
}

export function pruneFallbackNoticesAfter(storedSessionId: string, beforeMessageCount: number) {
  const sessionId = storedSessionId.trim()

  if (!sessionId) {
    return
  }

  const registry = readRegistry()
  const session = registry[sessionId]

  if (!session) {
    return
  }

  const boundary = Math.max(0, Math.floor(beforeMessageCount))
  const notices = session.notices.filter(notice => notice.beforeMessageCount <= boundary)

  if (notices.length) {
    registry[sessionId] = { notices, updatedAt: Date.now() }
  } else {
    delete registry[sessionId]
  }

  writeRegistry(registry)
}

export function restoreFallbackNotices(storedSessionId: string, messages: ChatMessage[]): ChatMessage[] {
  const notices = readRegistry()[storedSessionId.trim()]?.notices

  if (!notices?.length) {
    return messages
  }

  const existingIds = new Set(messages.map(message => message.id))
  const missing = notices.filter(notice => !existingIds.has(notice.id))

  if (!missing.length) {
    return messages
  }

  const restored = [...messages]

  for (const notice of missing.sort(
    (left, right) => left.beforeMessageCount - right.beforeMessageCount || left.timestamp - right.timestamp
  )) {
    let insertionIndex = 0
    let transcriptMessageCount = 0

    while (insertionIndex < restored.length && transcriptMessageCount < notice.beforeMessageCount) {
      const message = restored[insertionIndex]

      if (message.role === 'user' || message.role === 'assistant') {
        transcriptMessageCount += 1
      }

      insertionIndex += 1
    }

    // Preserve emission order when more than one fallback notice belongs at
    // the same transcript boundary, while still keeping every notice before
    // the assistant response produced by the selected fallback.
    while (insertionIndex < restored.length && isFallbackNoticeMessage(restored[insertionIndex])) {
      insertionIndex += 1
    }

    restored.splice(insertionIndex, 0, fallbackNoticeMessage(notice))
  }

  return restored
}
