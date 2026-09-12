import { atom, computed, type ReadableAtom } from 'nanostores'

import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { type ChatMessage, restorePendingClarifyToolCall } from '@/lib/chat-messages'

import { $clarifyRequests } from './clarify'

interface TranscriptViewGate {
  token: symbol
  owner?: symbol
  storedSessionId?: string
}

/** Ephemeral display authority only; canonical session history stays untouched. */
export const $sessionTranscriptViewGates = atom<Record<string, TranscriptViewGate>>({})

export function holdTranscriptView(runtimeId: string, owner?: symbol, storedSessionId?: string): () => void {
  const token = Symbol(runtimeId)
  $sessionTranscriptViewGates.set({ ...$sessionTranscriptViewGates.get(), [runtimeId]: { token, owner, storedSessionId } })

  return () => {
    if ($sessionTranscriptViewGates.get()[runtimeId]?.token === token) {clearTranscriptViewGate(runtimeId)}
  }
}

export function clearTranscriptViewGate(runtimeId: string) {
  const current = $sessionTranscriptViewGates.get()

  if (!current[runtimeId]) {return}
  const { [runtimeId]: _removed, ...rest } = current
  $sessionTranscriptViewGates.set(rest)
}

export function clearTranscriptViewGates(owner?: symbol) {
  const current = $sessionTranscriptViewGates.get()
  const next = owner ? Object.fromEntries(Object.entries(current).filter(([, gate]) => gate.owner !== owner)) : {}

  if (Object.keys(next).length !== Object.keys(current).length) {$sessionTranscriptViewGates.set(next)}
}

const NO_MESSAGES: ChatMessage[] = []

/** Select each runtime input before projecting, so unrelated requests and
 * metadata heartbeats cannot rebuild the displayed message array. */
export function transcriptMessagesForView(
  $runtimeId: ReadableAtom<string | null>,
  $messages: ReadableAtom<ChatMessage[]>
): ReadableAtom<ChatMessage[]> {
  const $held = computed([$runtimeId, $sessionTranscriptViewGates], (id, gates) => Boolean(id && gates[id]))

  const $request = computed([$runtimeId, $clarifyRequests], (id, requests) =>
    id && requests[id]?.sessionId === id ? requests[id] : undefined
  )

  const $projection = computed([$held, $request], (held, request) => {
    if (!held || !request) {return NO_MESSAGES}

    return restorePendingClarifyToolCall(
      [{ id: `pending-clarify:${request.sessionId}:${request.requestId}`, role: 'assistant', parts: [] }],
      pendingClarifyToolPayload(request),
      request.receivedAt ?? 0
    ).messages
  })

  return computed([$held, $projection, $messages], (held, projection, messages) => held ? projection : messages)
}
