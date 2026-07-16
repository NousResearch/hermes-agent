import { ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { useMemo, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { coalesceToolOnlyAssistants, createToolMergeCache, toRuntimeMessage } from '@/lib/chat-runtime'

/**
 * ChatMessage[] -> assistant-ui message repository, with a WeakMap identity
 * cache so unchanged messages convert once (and a tool-merge cache that folds
 * tool-only assistant turns into their neighbour). Shared by the main chat's
 * runtime boundary and session tiles — one transcript pipeline, N surfaces.
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
      let parentId = visibleParentId

      if (message.role === 'assistant' && message.branchGroupId) {
        if (!branchParentByGroup.has(message.branchGroupId)) {
          branchParentByGroup.set(message.branchGroupId, visibleParentId)
        }

        parentId = branchParentByGroup.get(message.branchGroupId) ?? null
      }

      const cachedMessage = cacheRef.current.get(message)
      const runtimeMessage = cachedMessage ?? toRuntimeMessage(message)

      if (!cachedMessage) {
        cacheRef.current.set(message, runtimeMessage)
      }

      // assistant-ui's MessageRepository keys every node by id and throws on
      // `performOp/link` if the same id appears twice in a parent chain. The
      // adapter derives ids as `${timestamp}-${index}-${role}`, which is unique
      // within a single toChatMessages() pass but CAN collide once two passes
      // are merged (resume-reconcile grafting a re-fetched transcript onto the
      // live stream, or a hidden rewind branch that shares a sibling's id). One
      // duplicate id crashes the whole renderer, so this funnel — the only
      // caller of fromBranchableArray — is where id-uniqueness is enforced.
      // Render-only: suffix the runtime node's id without touching the
      // $messages store, preserving every message instead of dropping one.
      let uniqueId = runtimeMessage.id

      if (seenIds.has(uniqueId)) {
        let suffix = 1

        while (seenIds.has(`${uniqueId}#${suffix}`)) {
          suffix += 1
        }

        uniqueId = `${uniqueId}#${suffix}`
      }

      seenIds.add(uniqueId)

      const uniqueMessage = uniqueId === runtimeMessage.id ? runtimeMessage : { ...runtimeMessage, id: uniqueId }

      items.push({ message: uniqueMessage, parentId })

      if (!message.hidden) {
        visibleParentId = uniqueId
        headId = uniqueId
      }
    }

    return ExportedMessageRepository.fromBranchableArray(items, { headId })
  }, [messages])
}

// The funnel above suffixes duplicate runtime ids (`id#1`, `id#2`) so
// assistant-ui's MessageRepository never sees a colliding render key. Real
// message ids are `${timestamp}-${index}-${role}` and never contain `#`, so a
// trailing `#<n>` is unambiguously that render-only dedup marker. Strip it
// before an id crosses BACK to the $messages store or the gateway (edit /
// reload / branch / restore), or the backend lookup misses the real message.
// Idempotent on un-suffixed ids.
export function stripRuntimeIdSuffix(id: string): string {
  return id.replace(/#\d+$/, '')
}

export function stripRuntimeIdSuffixNullable(id: string | null): string | null {
  return id === null ? null : stripRuntimeIdSuffix(id)
}
