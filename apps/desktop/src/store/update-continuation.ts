const UPDATE_CONTINUATION_KEY = 'hermes.desktop.update-continuation'
const DEFAULT_MAX_AGE_MS = 2 * 60 * 60 * 1000

const UPDATE_CONTINUATION_PROMPT_BASE =
  'Hermes wurde für das Update neu gestartet. Setze die zuvor unterbrochene Aufgabe in diesem Chat fort. ' +
  'Prüfe zuerst, ob Update und Laufzeit gesund sind, und beende dann den ursprünglichen Auftrag. ' +
  'Starte keinen weiteren Update-Lauf, wenn der aktuelle bereits erfolgreich war.'

export interface UpdateContinuationStorage {
  getItem: (key: string) => null | string
  removeItem: (key: string) => void
  setItem: (key: string, value: string) => void
}

export interface UpdateContinuation {
  armedAt: number
  attemptedAt?: number
  requestId: string
  sessionId: string
}

function newRequestId(): string {
  return globalThis.crypto?.randomUUID?.().replaceAll('-', '') ?? `${Date.now()}${Math.random()}`.replace(/\D/g, '')
}

export function updateContinuationPrompt(requestId: string): string {
  return `${UPDATE_CONTINUATION_PROMPT_BASE}\n\n<!-- hermes-update-continuation:${requestId} -->`
}

export function updateContinuationToken(requestId: string): string {
  return `hermes-update-continuation:${requestId}`
}

function browserStorage(): null | UpdateContinuationStorage {
  return typeof globalThis.localStorage === 'undefined' ? null : globalThis.localStorage
}

export function continuationSessionId(input: {
  activeStoredSessionId: null | string
  selectedStoredSessionId: null | string
  busy: boolean
  awaitingResponse?: boolean
}): null | string {
  if (!input.busy && !input.awaitingResponse) {
    return null
  }

  return input.activeStoredSessionId || input.selectedStoredSessionId || null
}

export function armUpdateContinuation(
  sessionId: string,
  opts: { now?: number; storage?: null | UpdateContinuationStorage } = {}
): boolean {
  const storage = opts.storage === undefined ? browserStorage() : opts.storage
  const normalized = sessionId.trim()

  if (!storage || !normalized) {
    return false
  }

  storage.setItem(
    UPDATE_CONTINUATION_KEY,
    JSON.stringify({
      armedAt: opts.now ?? Date.now(),
      requestId: newRequestId(),
      sessionId: normalized
    } satisfies UpdateContinuation)
  )

  return true
}

export function clearUpdateContinuation(storage: null | UpdateContinuationStorage = browserStorage()): void {
  storage?.removeItem(UPDATE_CONTINUATION_KEY)
}

export function markUpdateContinuationAttempt(
  continuation: UpdateContinuation,
  opts: { now?: number; storage?: null | UpdateContinuationStorage } = {}
): void {
  const storage = opts.storage === undefined ? browserStorage() : opts.storage

  storage?.setItem(
    UPDATE_CONTINUATION_KEY,
    JSON.stringify({ ...continuation, attemptedAt: opts.now ?? Date.now() } satisfies UpdateContinuation)
  )
}

export function readUpdateContinuation(
  opts: { maxAgeMs?: number; now?: number; storage?: null | UpdateContinuationStorage } = {}
): null | UpdateContinuation {
  const storage = opts.storage === undefined ? browserStorage() : opts.storage

  if (!storage) {
    return null
  }

  const raw = storage.getItem(UPDATE_CONTINUATION_KEY)

  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<UpdateContinuation>
    const sessionId = typeof parsed.sessionId === 'string' ? parsed.sessionId.trim() : ''
    const requestId = typeof parsed.requestId === 'string' ? parsed.requestId.trim() : ''
    const armedAt = Number(parsed.armedAt)
    const attemptedAt = parsed.attemptedAt === undefined ? undefined : Number(parsed.attemptedAt)
    const age = (opts.now ?? Date.now()) - armedAt

    if (
      !sessionId ||
      !/^[a-z0-9]{16,64}$/i.test(requestId) ||
      !Number.isFinite(armedAt) ||
      (attemptedAt !== undefined && !Number.isFinite(attemptedAt)) ||
      age < 0 ||
      age > (opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS)
    ) {
      storage.removeItem(UPDATE_CONTINUATION_KEY)
      return null
    }

    return { armedAt, ...(attemptedAt === undefined ? {} : { attemptedAt }), requestId, sessionId }
  } catch {
    storage.removeItem(UPDATE_CONTINUATION_KEY)
    return null
  }
}
