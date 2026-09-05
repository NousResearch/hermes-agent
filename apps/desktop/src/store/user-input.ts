import { atom, computed, type ReadableAtom } from 'nanostores'

import { markSessionGone } from './runtime-gone'
import { $activeSessionId } from './session'
import { ambientRequestFor, isSessionGoneForBackgroundPolling } from './session-gone-latch'
import { requestForOwnedSession } from './session-states'

export interface UserInputQuestion {
  allowFreeText: boolean
  defaultValue?: unknown
  id: string
  options: string[]
  text: string
}

export type UserInputStatus = 'answered' | 'expired' | 'pending'

export interface UserInputRequest {
  context: string
  expiresAt: number
  questions: UserInputQuestion[]
  requestId: string
  sessionId: string
  status: UserInputStatus
  turnId: string
}

interface UserInputGateway {
  request: <T = unknown>(method: string, params?: Record<string, unknown>, timeoutMs?: number, signal?: AbortSignal) => Promise<T>
}

const emptyRequests: Record<string, UserInputRequest[]> = {}

export const $userInputRequests = atom<Record<string, UserInputRequest[]>>(emptyRequests)
export const $activeUserInputRequests: ReadableAtom<UserInputRequest[]> = computed(
  [$userInputRequests, $activeSessionId],
  (all, activeSessionId) => (activeSessionId ? all[activeSessionId] ?? [] : [])
)

const text = (value: unknown): string => typeof value === 'string' ? value : ''

function normalizeQuestion(value: unknown): UserInputQuestion | null {
  if (!value || typeof value !== 'object') {return null}
  const raw = value as Record<string, unknown>
  const id = text(raw.id).trim()
  const questionText = text(raw.text || raw.question).trim()

  if (!id || !questionText) {return null}

  const options = Array.isArray(raw.options)
    ? raw.options.filter((option): option is string => typeof option === 'string').map(option => option.trim()).filter(Boolean)
    : []

  return {
    allowFreeText: raw.allow_free_text === true || raw.allowFreeText === true,
    defaultValue: raw.default ?? raw.defaultValue,
    id,
    options,
    text: questionText
  }
}

export function normalizeUserInputRequest(value: unknown, sessionIdOverride?: string | null): UserInputRequest | null {
  if (!value || typeof value !== 'object') {return null}
  const raw = value as Record<string, unknown>
  const requestId = text(raw.request_id || raw.requestId).trim()
  const sessionId = text(sessionIdOverride || raw.session_id || raw.sessionId).trim()

  const questions = Array.isArray(raw.questions)
    ? raw.questions.map(normalizeQuestion).filter((question): question is UserInputQuestion => question !== null)
    : []

  if (!requestId || !sessionId || questions.length === 0) {return null}
  const rawStatus = text(raw.status).trim().toLowerCase()
  const status: UserInputStatus = rawStatus === 'answered' || rawStatus === 'expired' ? rawStatus : 'pending'

  const expiresAt = typeof raw.expires_at === 'number'
    ? raw.expires_at
    : typeof raw.expiresAt === 'number' ? raw.expiresAt : 0

  return {
    context: text(raw.context),
    expiresAt,
    questions,
    requestId,
    sessionId,
    status,
    turnId: text(raw.turn_id || raw.turnId)
  }
}

export function setUserInputRequest(request: UserInputRequest | unknown): UserInputRequest | null {
  const normalized = normalizeUserInputRequest(request)

  if (!normalized) {return null}
  const all = $userInputRequests.get()
  const current = all[normalized.sessionId] ?? []
  const next = current.filter(item => item.requestId !== normalized.requestId)
  $userInputRequests.set({
    ...all,
    [normalized.sessionId]: [...next, normalized]
  })

  return normalized
}

export function clearUserInputRequest(sessionId: string | null | undefined, requestId: string | null | undefined): void {
  const key = text(sessionId).trim()
  const id = text(requestId).trim()

  if (!key || !id) {return}
  const all = $userInputRequests.get()
  const current = all[key]

  if (!current) {return}
  const next = current.filter(item => item.requestId !== id)

  if (next.length === current.length) {return}
  const updated = { ...all }

  if (next.length) {updated[key] = next}
  else {delete updated[key]}

  $userInputRequests.set(updated)
}

export function replaceUserInputRequests(sessionId: string, requests: unknown[]): void {
  const key = text(sessionId).trim()

  if (!key) {return}

  const normalized = requests
    .map(request => normalizeUserInputRequest(request, key))
    .filter((request): request is UserInputRequest => request !== null && request.status === 'pending')

  const all = { ...$userInputRequests.get() }

  if (normalized.length) {all[key] = normalized}
  else {delete all[key]}

  $userInputRequests.set(all)
}

export function userInputRequestsForSession(sessionId: string | null | undefined): UserInputRequest[] {
  return sessionId ? $userInputRequests.get()[sessionId] ?? [] : []
}

export const sessionUserInputRequests = (sessionId: string | null) =>
  computed($userInputRequests, all => (sessionId ? all[sessionId] ?? [] : []))

export async function replayPendingUserInput(
  gateway: UserInputGateway | null,
  sessionId: string | null | undefined
): Promise<void> {
  const key = text(sessionId).trim()

  if (!gateway || !key) {return}
  let rawResult: unknown

  try {
    rawResult = await requestForOwnedSession(key, ambientRequestFor(gateway), 'user_input.pending', {
      session_id: key
    })
  } catch (error) {
    if (isSessionGoneForBackgroundPolling(error)) {
      markSessionGone(key)

      return
    }

    throw error
  }

  const result = rawResult && typeof rawResult === 'object' ? rawResult as Record<string, unknown> : {}

  const requests = Array.isArray(result.requests)
    ? result.requests
    : Array.isArray(result.data) ? result.data : []

  replaceUserInputRequests(key, requests)
}

export async function respondUserInput(
  gateway: UserInputGateway | null,
  request: UserInputRequest,
  answers: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (!gateway) {throw new Error('Hermes gateway is disconnected.')}

  const result = await requestForOwnedSession<Record<string, unknown>>(
    request.sessionId,
    ambientRequestFor(gateway),
    'user_input.respond',
    {
      answers,
      request_id: request.requestId,
      session_id: request.sessionId
    }
  )

  if (result?.status === 'not_found') {
    clearUserInputRequest(request.sessionId, request.requestId)
    throw new Error('This Hermes input request is no longer pending.')
  }

  if (result?.accepted !== false) {clearUserInputRequest(request.sessionId, request.requestId)}

  return result ?? {}
}
