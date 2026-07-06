import {
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  textPart
} from '@/lib/chat-messages'

const STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const STORE_VERSION = 1
const MAX_ENTRIES = 24
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000

export interface BackendInFlightTurn {
  assistant?: string
  streaming?: boolean
  user?: string
}

export interface InFlightTurnSnapshot {
  awaitingResponse: boolean
  busy: boolean
  messages: ChatMessage[]
  runtimeSessionId: string
  storedSessionId: string
  streamId: null | string
  turnStartedAt: null | number
  updatedAt: number
  version: typeof STORE_VERSION
}

export interface JournalableSessionState {
  awaitingResponse: boolean
  busy: boolean
  messages: ChatMessage[]
  storedSessionId: null | string
  streamId: null | string
  turnStartedAt: null | number
}

interface JournalStore {
  entries: Record<string, InFlightTurnSnapshot>
  version: typeof STORE_VERSION
}

export interface InFlightRecoveryResult {
  applied: boolean
  caughtUp: boolean
  messages: ChatMessage[]
  streamId: null | string
  turnStartedAt: null | number
}

function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage
  } catch {
    return null
  }
}

function emptyStore(): JournalStore {
  return { entries: {}, version: STORE_VERSION }
}

function loadStore(): JournalStore {
  const store = storage()

  if (!store) {
    return emptyStore()
  }

  try {
    const raw = store.getItem(STORAGE_KEY)

    if (!raw) {
      return emptyStore()
    }

    const parsed = JSON.parse(raw)

    if (!parsed || parsed.version !== STORE_VERSION || typeof parsed.entries !== 'object' || Array.isArray(parsed.entries)) {
      return emptyStore()
    }

    return {
      entries: parsed.entries as Record<string, InFlightTurnSnapshot>,
      version: STORE_VERSION
    }
  } catch {
    return emptyStore()
  }
}

function saveStore(journal: JournalStore): void {
  const store = storage()

  if (!store) {
    return
  }

  try {
    const entries = Object.fromEntries(
      Object.entries(journal.entries)
        .filter(([, entry]) => !isExpired(entry))
        .sort((a, b) => b[1].updatedAt - a[1].updatedAt)
        .slice(0, MAX_ENTRIES)
    )

    if (Object.keys(entries).length === 0) {
      store.removeItem(STORAGE_KEY)

      return
    }

    store.setItem(STORAGE_KEY, JSON.stringify({ entries, version: STORE_VERSION }))
  } catch {
    // The journal is a best-effort crash-recovery aid. Storage quota/private
    // mode failures must never break chat streaming.
  }
}

function isExpired(entry: InFlightTurnSnapshot, now = Date.now()): boolean {
  return now - entry.updatedAt > MAX_AGE_MS
}

function cloneMessages(messages: ChatMessage[]): ChatMessage[] {
  try {
    return JSON.parse(JSON.stringify(messages)) as ChatMessage[]
  } catch {
    return []
  }
}

function normalizedText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function attachmentSignature(message: ChatMessage): string {
  return (message.attachmentRefs ?? []).join('\n')
}

function userMessagesMatch(left: ChatMessage, right: ChatMessage): boolean {
  return (
    left.role === 'user' &&
    right.role === 'user' &&
    normalizedText(chatMessageText(left)) === normalizedText(chatMessageText(right)) &&
    attachmentSignature(left) === attachmentSignature(right)
  )
}

function partHasRecoverableContent(part: ChatMessagePart): boolean {
  if (part.type === 'text' || part.type === 'reasoning') {
    return typeof part.text === 'string' && part.text.trim().length > 0
  }

  return part.type === 'tool-call'
}

function assistantHasRecoverableContent(message: ChatMessage): boolean {
  return (
    message.role === 'assistant' &&
    (Boolean(message.error) || message.parts.some(partHasRecoverableContent))
  )
}

function recoverableTail(messages: ChatMessage[], streamId: null | string): ChatMessage[] {
  const visible = messages.filter(message => !message.hidden)
  let assistantIndex = -1

  if (streamId) {
    assistantIndex = visible.findIndex(message => message.id === streamId && assistantHasRecoverableContent(message))
  }

  if (assistantIndex < 0) {
    for (let index = visible.length - 1; index >= 0; index -= 1) {
      if (assistantHasRecoverableContent(visible[index])) {
        assistantIndex = index

        break
      }
    }
  }

  if (assistantIndex < 0) {
    return []
  }

  let start = assistantIndex

  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (visible[index].role === 'user') {
      start = index

      break
    }
  }

  return cloneMessages(visible.slice(start))
}

function normalizeRecoveredTail(tail: ChatMessage[], keepPending: boolean): ChatMessage[] {
  return cloneMessages(tail).map(message =>
    message.role === 'assistant'
      ? {
          ...message,
          pending: keepPending ? (message.pending ?? true) : false
        }
      : { ...message, pending: false }
  )
}

function findMatchingTailUserIndex(messages: ChatMessage[], tailUser: ChatMessage): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (userMessagesMatch(messages[index], tailUser)) {
      return index
    }
  }

  return -1
}

function appendedStreamId(tail: ChatMessage[]): null | string {
  return tail.find(message => message.role === 'assistant' && assistantHasRecoverableContent(message))?.id ?? null
}

export function mergeInFlightMessages(
  baseMessages: ChatMessage[],
  tailMessages: ChatMessage[],
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  const tail = normalizeRecoveredTail(tailMessages, Boolean(options.keepPending))

  if (!tail.some(assistantHasRecoverableContent)) {
    return {
      applied: false,
      caughtUp: false,
      messages: baseMessages,
      streamId: null,
      turnStartedAt: null
    }
  }

  const tailUserIndex = tail.findIndex(message => message.role === 'user')
  const tailUser = tailUserIndex >= 0 ? tail[tailUserIndex] : null
  const streamId = appendedStreamId(tail)

  if (!tailUser) {
    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...tail],
      streamId,
      turnStartedAt: null
    }
  }

  const matchingUserIndex = findMatchingTailUserIndex(baseMessages, tailUser)

  if (matchingUserIndex < 0) {
    return {
      applied: true,
      caughtUp: false,
      messages: [...baseMessages, ...tail],
      streamId,
      turnStartedAt: null
    }
  }

  const assistantAfterUser = baseMessages.slice(matchingUserIndex + 1).some(assistantHasRecoverableContent)

  if (assistantAfterUser) {
    return {
      applied: false,
      caughtUp: true,
      messages: baseMessages,
      streamId: null,
      turnStartedAt: null
    }
  }

  const append = tail.slice(tailUserIndex + 1)

  if (append.length === 0) {
    return {
      applied: false,
      caughtUp: false,
      messages: baseMessages,
      streamId: null,
      turnStartedAt: null
    }
  }

  return {
    applied: true,
    caughtUp: false,
    messages: [...baseMessages, ...append],
    streamId,
    turnStartedAt: null
  }
}

function hashText(value: string): string {
  let hash = 0

  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0
  }

  return Math.abs(hash).toString(36)
}

export function backendInFlightMessages(inflight: BackendInFlightTurn | null | undefined): ChatMessage[] {
  const user = inflight?.user?.trim() ?? ''
  const assistant = inflight?.assistant ?? ''

  if (!user && !assistant.trim()) {
    return []
  }

  const suffix = hashText(`${user}\0${assistant}`)
  const messages: ChatMessage[] = []

  if (user) {
    messages.push({
      id: `inflight-user-${suffix}`,
      parts: [textPart(user)],
      role: 'user'
    })
  }

  if (assistant.trim()) {
    messages.push({
      id: `inflight-assistant-${suffix}`,
      parts: [assistantTextPart(assistant)],
      pending: Boolean(inflight?.streaming),
      role: 'assistant'
    })
  }

  return messages
}

export function mergeBackendInFlightTurn(
  baseMessages: ChatMessage[],
  inflight: BackendInFlightTurn | null | undefined,
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  return mergeInFlightMessages(baseMessages, backendInFlightMessages(inflight), options)
}

export function persistInFlightTurnState(runtimeSessionId: string, state: JournalableSessionState): void {
  const storedSessionId = state.storedSessionId

  if (!storedSessionId) {
    return
  }

  const active = state.busy || state.awaitingResponse || Boolean(state.streamId)
  const tail = active ? recoverableTail(state.messages, state.streamId) : []

  if (!active || tail.length === 0) {
    clearInFlightTurnJournal(storedSessionId)

    return
  }

  const journal = loadStore()
  journal.entries[storedSessionId] = {
    awaitingResponse: state.awaitingResponse,
    busy: state.busy,
    messages: tail,
    runtimeSessionId,
    storedSessionId,
    streamId: state.streamId,
    turnStartedAt: state.turnStartedAt,
    updatedAt: Date.now(),
    version: STORE_VERSION
  }
  saveStore(journal)
}

export function readInFlightTurnJournal(storedSessionId: null | string): InFlightTurnSnapshot | null {
  if (!storedSessionId) {
    return null
  }

  const journal = loadStore()
  const entry = journal.entries[storedSessionId]

  if (!entry) {
    return null
  }

  if (isExpired(entry)) {
    delete journal.entries[storedSessionId]
    saveStore(journal)

    return null
  }

  return entry
}

export function recoverInFlightTurnJournal(
  storedSessionId: null | string,
  baseMessages: ChatMessage[],
  options: { keepPending?: boolean } = {}
): InFlightRecoveryResult {
  const snapshot = readInFlightTurnJournal(storedSessionId)

  if (!snapshot) {
    return {
      applied: false,
      caughtUp: false,
      messages: baseMessages,
      streamId: null,
      turnStartedAt: null
    }
  }

  const recovered = mergeInFlightMessages(baseMessages, snapshot.messages, options)

  if (recovered.caughtUp) {
    clearInFlightTurnJournal(storedSessionId)
  }

  return {
    ...recovered,
    streamId: recovered.applied ? (snapshot.streamId || recovered.streamId) : null,
    turnStartedAt: recovered.applied ? snapshot.turnStartedAt : null
  }
}

export function clearInFlightTurnJournal(storedSessionId: null | string): void {
  if (!storedSessionId) {
    return
  }

  const journal = loadStore()

  if (!(storedSessionId in journal.entries)) {
    return
  }

  delete journal.entries[storedSessionId]
  saveStore(journal)
}
