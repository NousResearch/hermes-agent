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
 * Same-uid rows pair in electDedupeRepresentatives before this key applies;
 * it remains the bridge that carries a uid-less live residue to its
 * rehydrated twin.
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
 * One pass over the settled rows to elect a representative per logical row.
 * Additive rank (review #123294: "select the durable, non-pending
 * representative"): +2 durable rowId, +1 visible, -1 pending. The common
 * shapes land at durable+visible+settled (3) > durable hidden OR durable
 * pending (2) > live settled (1) > live pending residue (0) — note the
 * ties: a durable PENDING row scores 2, above a plain settled live row (1),
 * because durability outweighs stream state. Keeping the first occurrence
 * had left the live id, dropped the durable row's rowId/reactions and stuck
 * the runtime message in running. Earliest top-rank copy wins ties, except
 * between same-uid rows where the NEWER generation wins; the winner renders
 * at the FIRST occurrence's position (swap-in in the render loop).
 */
function rankRepresentative(message: ChatMessage): number {
  return (message.rowId !== undefined ? 2 : 0) + (!message.hidden ? 1 : 0) - (message.pending ? 1 : 0)
}

/**
 * The later of two same-uid generations. Every re-insert (compaction
 * generation, rotation handoff, replacement) allocates a fresh, higher row
 * id, and the id is the store's address for the row; the timestamp — the
 * copy's commit time, later than the original's (#123985) — decides only
 * when neither row carries an id.
 */
function isNewerGeneration(candidate: ChatMessage, incumbent: ChatMessage): boolean {
  if (candidate.rowId !== undefined && incumbent.rowId !== undefined) {
    return candidate.rowId > incumbent.rowId
  }
  return (candidate.timestamp ?? 0) > (incumbent.timestamp ?? 0)
}

interface DedupeElection {
  /** Group root per row index — a lone row is its own group. */
  rootOf: number[]
  /** The elected representative of each group root. */
  representative: Map<number, ChatMessage>
}

/**
 * Group the rows that are the same logical message and elect one
 * representative per group. Two rows merge when they share a `messageUid` —
 * the backend casts one uid per logical row (agent/message_metadata.py
 * stamp_message_uid) and re-writes it on every re-insert, so #123985's
 * original plus its commit-time-stamped copy share a uid and merge even
 * though their content keys differ. Distinct uids are distinct rows however
 * equal the content. A pair where one side lacks a uid (a live stream
 * residue against its rehydrated twin) keeps the content-key pairing
 * exactly as before.
 */
function electDedupeRepresentatives(messages: ChatMessage[]): DedupeElection {
  // Union-find over rows sharing a logical identity: same uid unions two
  // generations; a shared content key unions only while one side has no uid
  // to pair by.
  const parent = messages.map((_, index) => index)

  const find = (index: number): number => {
    let root = index
    while (parent[root] !== root) {
      parent[root] = parent[parent[root]]
      root = parent[root]
    }
    return root
  }

  const union = (left: number, right: number): void => {
    const leftRoot = find(left)
    const rightRoot = find(right)
    if (leftRoot !== rightRoot) {
      parent[Math.max(leftRoot, rightRoot)] = Math.min(leftRoot, rightRoot)
    }
  }

  const firstByUid = new Map<string, number>()
  const firstByContentKey = new Map<string, number>()

  messages.forEach((message, index) => {
    if (message.messageUid !== undefined) {
      const first = firstByUid.get(message.messageUid)
      if (first === undefined) {
        firstByUid.set(message.messageUid, index)
      } else {
        union(first, index)
      }
    }

    const contentKey = displayDedupeKey(message)
    if (contentKey !== null) {
      const first = firstByContentKey.get(contentKey)
      if (first === undefined) {
        firstByContentKey.set(contentKey, index)
      } else if (message.messageUid === undefined || messages[first].messageUid === undefined) {
        union(first, index)
      }
    }
  })

  const representative = new Map<number, ChatMessage>()

  messages.forEach((message, index) => {
    const root = find(index)
    const incumbent = representative.get(root)
    if (incumbent === undefined) {
      representative.set(root, message)
      return
    }

    const rank = rankRepresentative(message)
    const incumbentRank = rankRepresentative(incumbent)
    if (
      rank > incumbentRank ||
      // A rank tie between same-uid rows is two durable generations of one
      // row: the re-insert the compaction wrote is the generation the store
      // still resolves, so it wins — electing the rewound original would
      // strand rowId and reactions on a row future reads no longer return.
      (rank === incumbentRank &&
        message.messageUid !== undefined &&
        message.messageUid === incumbent.messageUid &&
        isNewerGeneration(message, incumbent))
    ) {
      representative.set(root, message)
    }
  })

  return { rootOf: messages.map((_, index) => find(index)), representative }
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
    // Holds the ids ALREADY RENDERED — of the elected representatives, not of
    // the rows iterated: the elected durable copy renders at the first
    // occurrence's slot, so a later duplicate must skip on the WINNER's id.
    // Order-dependent by design: whatever lost the election is dropped here,
    // never re-rendered at its own position.
    // Logical-row dedup for settled messages (#101938, #123985): after a
    // context compression the same logical message can reach this boundary
    // twice with different ids — a streaming copy without a rowId and a
    // rehydrated copy with one, or two re-inserted generations of one row.
    // The id-only gate above misses that collision and both render as
    // visible duplicates. Rows pair by messageUid first; the content key
    // (the backend's _display_dedupe_key + _stable_tool_key) carries the
    // pairs where a row has no uid. A durable non-pending copy is elected
    // representative and is SWAPPED INTO the first occurrence's slot, so
    // rowId/reactions survive and a live row never leaks into the settled
    // transcript.
    const coalesced = coalesceToolOnlyAssistants(messages, toolMergeCacheRef.current)
    const { rootOf, representative } = electDedupeRepresentatives(coalesced)
    const renderedRoots = new Set<number>()
    let visibleParentId: string | null = null
    let headId: string | null = null

    for (let index = 0; index < coalesced.length; index += 1) {
      const message = coalesced[index]
      let messageToRender = message
      // Settled row of a logical message: if an earlier slot already
      // rendered this group, skip (later duplicate generation); otherwise
      // swap in the ELECTED representative — the durable, non-pending copy,
      // or the newer generation of a same-uid pair — so its rowId and
      // reactions render at the first occurrence's position instead of a
      // streaming residue's or a rewound original's.
      const root = rootOf[index]
      if (renderedRoots.has(root)) {
        continue
      }
      renderedRoots.add(root)
      messageToRender = representative.get(root) ?? message

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

      const cachedMessage = cacheRef.current.get(messageToRender)

      let runtimeMessage = cachedMessage

      if (!runtimeMessage) {
        // Guard against two `tool-call` parts of one message sharing a
        // `toolCallId`: assistant-ui's `useResources` throws on the duplicate
        // key and crash-loops the renderer (#87857). Same class of defensive
        // dedup as the repeated-`message.id` skip above, one level down at the
        // parts. Only a cache miss needs it: the cached record is already the
        // normalized result, and rebuilding the parts array for every settled
        // row on each streamed delta is pure waste.
        const deduped = withUniqueToolCallIdsWithinMessage(messageToRender)

        runtimeMessage = fromThreadMessageLike(toRuntimeMessage(deduped), messageToRender.id, FALLBACK_STATUS)
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
