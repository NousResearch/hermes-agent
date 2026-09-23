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
    let visibleParentId: string | null = null
    let headId: string | null = null

    for (const message of coalesceToolOnlyAssistants(messages, toolMergeCacheRef.current)) {
      // A repeated id is a transcript bug upstream, but it must not reach the
      // repository: MessageRepository throws on the second link ("A message
      // with the same id already exists in the parent tree") and takes the
      // whole workspace pane down with it. Keep the first occurrence — the
      // later copy carries the same id, so it is the row we already rendered.
      if (seenIds.has(message.id)) {
        continue
      }

      seenIds.add(message.id)

      let parentId = visibleParentId

      if (message.role === 'assistant' && message.branchGroupId) {
        if (!branchParentByGroup.has(message.branchGroupId)) {
          branchParentByGroup.set(message.branchGroupId, visibleParentId)
        }

        parentId = branchParentByGroup.get(message.branchGroupId) ?? null
      }

      // Guard against two `tool-call` parts of one message sharing a
      // `toolCallId`: assistant-ui's `useResources` throws on the duplicate key
      // and crash-loops the renderer (#87857). Same class of defensive dedup as
      // the repeated-`message.id` skip above, one level down at the parts. Keeps
      // identity when clean, so the cache below is unaffected in the common case.
      const deduped = withUniqueToolCallIdsWithinMessage(message)

      const cachedMessage = cacheRef.current.get(message)

      const runtimeMessage =
        cachedMessage ?? fromThreadMessageLike(toRuntimeMessage(deduped), message.id, FALLBACK_STATUS)

      if (!cachedMessage) {
        cacheRef.current.set(message, runtimeMessage)
      }

      items.push({ message: runtimeMessage, parentId })

      if (!message.hidden) {
        visibleParentId = message.id
        headId = message.id
      }
    }

    return { headId, messages: items }
  }, [messages])
}
