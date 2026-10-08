import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import type { ExportedMessageRepository, ThreadMessage } from '@assistant-ui/react'
import { useMemo, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { withUniqueToolCallIdsWithinMessage } from '@/lib/chat-messages'
import { coalesceToolOnlyAssistants, createToolMergeCache, toRuntimeMessage } from '@/lib/chat-runtime'

// The exact fallback status ExportedMessageRepository.fromBranchableArray uses.
// Normalization happens HERE, once per message, so the cached record below is
// already the final ThreadMessage the runtime consumes.
const FALLBACK_STATUS = getAutoStatus(false, false, false, false, undefined)

/**
 * Frontend twin of the backend's `_stable_tool_key` (hermes_state_identity.py):
 * an assistant message whose tool calls ALL carry call ids keys on the sorted
 * set of stable call ids instead of the arguments — a prune rewrites arguments
 * (#117750), so arguments must not carry identity. Deliberately excludes args
 * exactly like the backend. Returns null for any other message and for an
 * incomplete id set, so the caller keeps the full content key.
 */
export function stableToolKey(message: ChatMessage): string | null {
  if (message.role !== 'assistant') return null
  const calls = message.parts.filter(p => p.type === 'tool-call')
  if (calls.length === 0) return null
  const ids = calls.map(p => p.toolCallId)
  if (ids.some(id => id === undefined || id === '')) return null
  // Sorted: parts may arrive in a different order across generations, the
  // set of calls is the identity, not the sequence.
  return `stable|${[...ids].sort().join(',')}`
}

/**
 * Content-level dedupe key for settled messages — the frontend twin of the
 * backend's `_display_dedupe_key` (hermes_state_messages.py): full display
 * identity, not a narrow projection (review #123294: the first version keyed
 * only on role|text|timestamp|firstToolCallId and collapsed distinct tool
 * messages that share a reused first call id, e.g. providers reusing
 * "terminal_0" across turns). Six fields, the backend's tuple in order:
 * role | content | timestamp | tool_call_id | tool_calls | tool_name.
 */
export function displayDedupeKey(message: ChatMessage): string | null {
  // Settled-key guard: only timestamp-bearing rows participate. A live stream
  // row carries a timestamp AND pending=true (review #123294) — it KEYS like
  // its durable twin (so the pair can collapse) but never WINS the election
  // below: rank 0 loses to any durable copy, and a lone pending row still
  // renders while it streams.
  if (message.timestamp === undefined) return null

  const toolCalls = message.parts.filter(p => p.type === 'tool-call')
  const firstToolCallId = toolCalls[0]?.toolCallId ?? ''
  const toolName = toolCalls[0]?.toolName ?? ''

  const stable = stableToolKey(message)
  if (stable !== null) {
    return `${stable}|${message.timestamp}|${firstToolCallId}|${toolName}`
  }
  const normalizedText =
    message.parts
      .filter(p => p.type === 'text')
      .map(p => p.text.trim())
      .join(' ') || ''
  // JSON-canonical tool payload so distinct arguments stay distinct (review:
  // "key all behavior-bearing parts"). Order of calls is part of identity.
  const toolPayload =
    toolCalls.length > 0
      ? JSON.stringify(toolCalls.map(p => ({ id: p.toolCallId ?? '', name: p.toolName ?? '', args: p.args })))
      : ''
  return `${message.role}|${normalizedText}|${message.timestamp}|${firstToolCallId}|${toolPayload}|${toolName}`
}

/**
 * One pass over the settled rows to elect a representative per content key.
 * Rank lattice (review #123294: "select the durable, non-pending
 * representative"): durable+visible (3) > durable+hidden (2) > plain settled
 * (1) > pending streaming residue (0). Keeping the first occurrence had left
 * the live id, dropped the durable row's rowId/reactions and stuck the
 * runtime message in running. Earliest top-rank copy wins ties; the winner
 * renders at the FIRST occurrence's position (swap-in in the render loop).
 */
function rankRepresentative(message: ChatMessage): number {
  return (message.rowId !== undefined ? 2 : 0) + (!message.hidden ? 1 : 0) - (message.pending ? 1 : 0)
}

function electDedupeRepresentatives(messages: ChatMessage[]): Map<string, ChatMessage> {
  const byKey = new Map<string, ChatMessage>()
  for (const message of messages) {
    const key = displayDedupeKey(message)
    if (key === null) continue
    const incumbent = byKey.get(key)
    if (incumbent === undefined || rankRepresentative(message) > rankRepresentative(incumbent)) {
      byKey.set(key, message)
    }
  }
  return byKey
}

/**
 * ChatMessage[] -> assistant-ui message repository, with a WeakMap identity
 * cache so unchanged messages convert once (and a tool-merge cache that folds
 * tool-only assistant turns into their neighbour). Shared by the main chat's
 * runtime boundary and session tiles — one transcript pipeline, N surfaces.
 *
 * The cache stores NORMALIZED messages. `fromBranchableArray` maps the whole
 * array through `fromThreadMessageLike` on every call, so building the export
 * with it threw away the cache's reference identity once per streamed delta —
 * re-normalizing the entire settled transcript ~30x/s. Normalizing inside the
 * cache miss keeps identity stable for settled turns, which is what lets the
 * runtime reconcile detect that only the tail moved.
 */
export function useRuntimeMessageRepository(messages: ChatMessage[]): ExportedMessageRepository {
  const cacheRef = useRef(new WeakMap<ChatMessage, ThreadMessage>())
  const toolMergeCacheRef = useRef(createToolMergeCache())

  return useMemo(() => {
    const items: { message: ThreadMessage; parentId: string | null }[] = []
    const branchParentByGroup = new Map<string, string | null>()
    const seenIds = new Set<string>()
    // Content-level dedup for settled messages only (#101938): after a
    // context compression, the same logical message can reach this boundary
    // twice with different ids (a streaming copy without a rowId and a
    // rehydrated copy with one). The id-only gate above misses that collision
    // and both render as visible duplicates. The key mirrors the backend's
    // display identity (_display_dedupe_key + _stable_tool_key); a durable
    // non-pending copy is elected representative and is SWAPPED INTO the
    // first occurrence's slot, so rowId/reactions survive and a live row
    // never leaks into the settled transcript.
    const coalesced = coalesceToolOnlyAssistants(messages, toolMergeCacheRef.current)
    const electedByKey = electDedupeRepresentatives(coalesced)
    const renderedKeys = new Set<string>()
    let visibleParentId: string | null = null
    let headId: string | null = null

    for (const message of coalesced) {
      let messageToRender = message
      const contentKey = displayDedupeKey(message)
      if (contentKey !== null) {
        // Settled copy of a logical message: if an earlier slot already
        // rendered this key, skip (later duplicate generation); otherwise
        // swap in the ELECTED representative — the durable, non-pending
        // copy — so its rowId/reactions render at the first occurrence's
        // position instead of a streaming residue's.
        if (renderedKeys.has(contentKey)) {
          continue
        }
        renderedKeys.add(contentKey)
        messageToRender = electedByKey.get(contentKey) ?? message
      }

      // A repeated id is a transcript bug upstream, but it must not reach the
      // repository: MessageRepository throws on the second link ("A message
      // with the same id already exists in the parent tree") and takes the
      // whole workspace pane down with it. Keep the first occurrence — the
      // later copy carries the same id, so it is the row we already rendered.
      if (seenIds.has(messageToRender.id)) {
        continue
      }
      seenIds.add(messageToRender.id)

      let parentId = visibleParentId

      if (messageToRender.role === 'assistant' && messageToRender.branchGroupId) {
        if (!branchParentByGroup.has(messageToRender.branchGroupId)) {
          branchParentByGroup.set(messageToRender.branchGroupId, visibleParentId)
        }

        parentId = branchParentByGroup.get(messageToRender.branchGroupId) ?? null
      }

      // Guard against two `tool-call` parts of one message sharing a
      // `toolCallId`: assistant-ui's `useResources` throws on the duplicate key
      // and crash-loops the renderer (#87857). Same class of defensive dedup as
      // the repeated-`message.id` skip above, one level down at the parts. Keeps
      // identity when clean, so the cache below is unaffected in the common case.
      const deduped = withUniqueToolCallIdsWithinMessage(messageToRender)

      const cachedMessage = cacheRef.current.get(messageToRender)

      const runtimeMessage =
        cachedMessage ?? fromThreadMessageLike(toRuntimeMessage(deduped), messageToRender.id, FALLBACK_STATUS)

      if (!cachedMessage) {
        cacheRef.current.set(messageToRender, runtimeMessage)
      }

      items.push({ message: runtimeMessage, parentId })

      if (!messageToRender.hidden) {
        visibleParentId = messageToRender.id
        headId = messageToRender.id
      }
    }

    return { headId, messages: items }
  }, [messages])
}
