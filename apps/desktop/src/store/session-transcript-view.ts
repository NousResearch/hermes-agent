import { atom, computed, type ReadableAtom } from 'nanostores'

import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { connectionRequestToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-connection'
import { type ChatMessage, restorePendingBlockingToolCall, restorePendingClarifyToolCall } from '@/lib/chat-messages'

import { $clarifyRequests } from './clarify'
import { $connectionRequests } from './connection-request'

interface TranscriptGate {
  token: symbol
  owner: symbol
  baseline: ChatMessage[]
}

/** Display authority only: never replace the canonical/cache transcript. */
export const $sessionTranscriptViewGates = atom<Record<string, TranscriptGate>>({})

export function holdTranscriptView(runtimeId: string, owner: symbol, baseline: ChatMessage[]): () => void {
  const token = Symbol(runtimeId)
  const gates = $sessionTranscriptViewGates.get()
  $sessionTranscriptViewGates.set({
    ...gates,
    [runtimeId]: { token, owner, baseline: gates[runtimeId]?.baseline ?? baseline }
  })

  return () => {
    if ($sessionTranscriptViewGates.get()[runtimeId]?.token === token) {
      clearTranscriptViewGate(runtimeId)
    }
  }
}

export function clearTranscriptViewGate(runtimeId: string): void {
  const { [runtimeId]: removed, ...rest } = $sessionTranscriptViewGates.get()

  if (removed) {
    $sessionTranscriptViewGates.set(rest)
  }
}

export function clearTranscriptViewGates(owner: symbol): void {
  for (const [id, gate] of Object.entries($sessionTranscriptViewGates.get())) {
    if (gate.owner === owner) {
      clearTranscriptViewGate(id)
    }
  }
}

const EMPTY: ChatMessage[] = []

function liveMessages(messages: ChatMessage[], baseline: ChatMessage[]): ChatMessage[] {
  const baselineById = new Map(baseline.map(message => [message.id, message]))

  return messages.flatMap(message => {
    const previous = baselineById.get(message.id)

    if (!previous) {
      return [message]
    }

    // A replay can append a tool to an old assistant row. Only the new
    // parts (and actual text deltas) are authority, not its old commentary.
    const parts = message.parts.flatMap((part, index): ChatMessage['parts'] => {
      const old = previous.parts[index]

      if (!old) {
        return [part]
      }

      if ((part.type === 'text' || part.type === 'reasoning') && old.type === part.type) {
        const text = part.text.startsWith(old.text) ? part.text.slice(old.text.length) : part.text

        return text ? [{ ...part, text }] : []
      }

      return part.type === 'tool-call' &&
        old.type === 'tool-call' &&
        (part.result !== old.result || part.argsText !== old.argsText)
        ? [part]
        : []
    })

    return parts.length ? [{ ...message, parts }] : []
  })
}

export function transcriptMessagesForView(
  $runtimeId: ReadableAtom<string | null>,
  $messages: ReadableAtom<ChatMessage[]>
): ReadableAtom<ChatMessage[]> {
  const $gate = computed([$runtimeId, $sessionTranscriptViewGates], (id, gates) => (id ? gates[id] : undefined))
  const $request = computed([$runtimeId, $clarifyRequests], (id, requests) => (id ? requests[id] : undefined))
  const $connection = computed([$runtimeId, $connectionRequests], (id, requests) => (id ? requests[id] : undefined))

  return computed([$gate, $request, $connection, $messages], (gate, request, connection, messages) => {
    if (!gate) {
      return messages
    }

    const live = liveMessages(messages, gate.baseline)

    // Request lifetime owns the temporary card, including answer/cancel while
    // REST is pending. Do not preserve a synthetic replay after it is gone.
    let visible = live.flatMap(message => {
      const parts = message.parts.filter(
        part => part.type !== 'tool-call' || part.toolName !== 'clarify' || part.result !== undefined
      )

      return parts.length ? [{ ...message, parts }] : []
    })

    if (connection && !connection.settled) {
      visible = restorePendingBlockingToolCall(
        [
          ...visible,
          { id: `pending-connection:${connection.sessionId}:${connection.opId}`, role: 'assistant', parts: [] }
        ],
        connectionRequestToolPayload(connection),
        connection.receivedAt ?? 0
      ).messages
    }

    if (!request) {
      return visible.length ? visible : EMPTY
    }

    return restorePendingClarifyToolCall(
      [...visible, { id: `pending-clarify:${request.sessionId}:${request.requestId}`, role: 'assistant', parts: [] }],
      pendingClarifyToolPayload(request),
      request.receivedAt ?? 0
    ).messages
  })
}
