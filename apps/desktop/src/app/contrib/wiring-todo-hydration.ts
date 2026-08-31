import { getLatestSessionMessages, getSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { latestSessionTodoState } from '@/lib/todos'
import { isSessionOwnerRoute, ownerRouteProfileScope, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession, storedSessionIdForRuntimeId } from '@/store/session-states'
import {
  $todoContinuationsBySession,
  captureTodoWriteFence,
  clearSessionTodos,
  releaseTodoHydrationToken,
  setSessionTodos,
  type TodoHydrationToken,
  todosForHydration
} from '@/store/todos'
import type { SessionMessage } from '@/types/hermes'

const TODO_STATE_RESPONSE_MAX_MESSAGES = 2
const TODO_STATE_RESPONSE_MAX_UTF8_BYTES = 1_100_000
const POST_TURN_RETRY_DELAY_MS = 250

function sessionOwnerAuthorityKey(owner: SessionOwnerScope): string {
  if (isSessionOwnerRoute(owner)) {
    return JSON.stringify([
      'route',
      owner.connectionId.trim(),
      owner.profile.trim() || 'default',
      owner.targetProfile?.trim() || '',
      owner.mode ?? ''
    ])
  }

  return JSON.stringify(['profile', typeof owner === 'string' ? owner.trim() || 'default' : ''])
}

function encodedUtf8ByteLength(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength
}

interface TodoCandidatePagination {
  exhausted?: boolean
  has_more?: boolean
  next_before_id?: null | number
}

/** Resolve the latest valid Todo state without loading ordinary durable rows. */
export async function resolveStoredSessionTodoMessages(
  storedSessionId: string,
  profile: ProfileScope,
  visibleTail: SessionMessage[]
): Promise<SessionMessage[]> {
  if (latestSessionTodoState(visibleTail)) {
    return visibleTail
  }

  const page = await getSessionMessages(storedSessionId, profile, { projection: 'todo-state' })
  const pagination = page.pagination as TodoCandidatePagination | undefined

  if (
    pagination?.exhausted !== true ||
    pagination.has_more === true ||
    pagination.next_before_id != null ||
    page.messages.length > TODO_STATE_RESPONSE_MAX_MESSAGES ||
    encodedUtf8ByteLength(page.messages) > TODO_STATE_RESPONSE_MAX_UTF8_BYTES
  ) {
    throw new Error('Todo-state projection exceeded its bounded response contract')
  }

  return page.messages
}

/** Rehydrate one finished turn through the exact owner that holds its runtime. */
export async function hydratePostTurnStoredSession({
  attempts = 1,
  isCurrent = () => true,
  publishTranscript,
  runtimeSessionId,
  storedSessionId
}: {
  attempts?: number
  isCurrent?: () => boolean
  publishTranscript?: (messages: ChatMessage[]) => void
  runtimeSessionId: string
  storedSessionId: string
}): Promise<ChatMessage[] | null> {
  const owner = knownOwnerForSession(runtimeSessionId) ?? knownOwnerForSession(storedSessionId)
  const ownerKey = sessionOwnerAuthorityKey(owner)
  const profileScope: ProfileScope = isSessionOwnerRoute(owner) ? ownerRouteProfileScope(owner) : owner
  const todoHydrationFence = captureTodoWriteFence(runtimeSessionId)
  const publicationIsCurrent = () =>
    isCurrent() &&
    storedSessionIdForRuntimeId(runtimeSessionId) === storedSessionId &&
    sessionOwnerAuthorityKey(knownOwnerForSession(runtimeSessionId) ?? knownOwnerForSession(storedSessionId)) ===
      ownerKey

  try {
    for (let index = 0; index < Math.max(1, attempts); index += 1) {
      try {
        const latest = await getLatestSessionMessages(storedSessionId, profileScope)

        if (!publicationIsCurrent()) {
          return null
        }

        const todoMessages = await resolveStoredSessionTodoMessages(storedSessionId, profileScope, latest.messages)

        if (!publicationIsCurrent()) {
          return null
        }

        const todoAccepted = hydrateSessionTodos(runtimeSessionId, todoMessages, todoHydrationFence)

        if (!todoAccepted || !publicationIsCurrent()) {
          return null
        }

        const messages = toChatMessages(latest.messages)

        if (!publicationIsCurrent()) {
          return null
        }

        publishTranscript?.(messages)

        return messages
      } catch {
        // Best-effort fallback when live stream payloads or projected Todo reads fail.
      }

      if (!publicationIsCurrent()) {
        return null
      }

      if (index < attempts - 1) {
        await new Promise(resolve => setTimeout(resolve, POST_TURN_RETRY_DELAY_MS))

        if (!publicationIsCurrent()) {
          return null
        }
      }
    }

    return null
  } finally {
    releaseTodoHydrationToken(todoHydrationFence)
  }
}

/** Apply stored transcript todo state for the exact runtime being hydrated. */
export function hydrateSessionTodos(
  runtimeSessionId: string,
  messages: Parameters<typeof latestSessionTodoState>[0],
  ifUnchangedSince?: TodoHydrationToken
): boolean {
  const state = latestSessionTodoState(messages)
  const restored =
    state?.source === 'carrier'
      ? state.todos
      : todosForHydration(state?.todos ?? null, $todoContinuationsBySession.get()[runtimeSessionId])

  if (restored) {
    return setSessionTodos(runtimeSessionId, restored, {
      ...(ifUnchangedSince === undefined ? {} : { ifUnchangedSince }),
      preserved: state?.source === 'carrier'
    })
  }

  return clearSessionTodos(runtimeSessionId, ifUnchangedSince === undefined ? {} : { ifUnchangedSince })
}

/** Project one persisted transcript only after its raw Todo carrier has been applied. */
export function hydrateStoredSessionMessages(
  runtimeSessionId: string,
  messages: SessionMessage[],
  ifUnchangedSince?: TodoHydrationToken
): ChatMessage[] {
  hydrateSessionTodos(runtimeSessionId, messages, ifUnchangedSince)

  return toChatMessages(messages)
}
