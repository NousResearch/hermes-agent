import type { ExportedMessageRepository, ThreadMessage } from '@assistant-ui/react'
import { useMemo, useRef } from 'react'

import { type ChatMessage, getChatMessageListUpdate } from '@/lib/chat-messages'
import { coalesceToolOnlyAssistants, createToolMergeCache, toRuntimeMessage } from '@/lib/chat-runtime'

export const runtimeMessageRepositoryDelta = Symbol('runtimeMessageRepositoryDelta')

export type RuntimeMessageRepository = ExportedMessageRepository & {
  [runtimeMessageRepositoryDelta]?: {
    message: ThreadMessage
    parentId: string | null
  }
}

type RepositoryCache = {
  repository: RuntimeMessageRepository
  source: ChatMessage[]
  tailItem: { message: ThreadMessage; parentId: string | null } | null
  tailSource: ChatMessage | null
}

/**
 * ChatMessage[] -> assistant-ui message repository, with a WeakMap identity
 * cache so unchanged messages convert once (and a tool-merge cache that folds
 * tool-only assistant turns into their neighbour). Shared by the main chat's
 * runtime boundary and session tiles — one transcript pipeline, N surfaces.
 */
export function useRuntimeMessageRepository(messages: ChatMessage[]): RuntimeMessageRepository {
  const cacheRef = useRef(new WeakMap<ChatMessage, ThreadMessage>())
  const toolMergeCacheRef = useRef(createToolMergeCache())
  const repositoryCacheRef = useRef<RepositoryCache | null>(null)

  return useMemo(() => {
    const previous = repositoryCacheRef.current
    const update = previous ? getChatMessageListUpdate(previous.source, messages) : null
    const nextTail = messages.at(-1) ?? null

    // Stream deltas replace only the pending tail. The provenance attached by
    // updateChatMessageAt lets us forward that one normalized record without
    // rescanning/coalescing the settled transcript. Completion, branch changes,
    // hydration, and arbitrary edits fail closed to the full reconcile below.
    if (
      previous &&
      update &&
      update.index === messages.length - 1 &&
      update.previousMessage === previous.tailSource &&
      update.previousMessage.pending &&
      update.message.pending &&
      update.previousMessage.id === update.message.id &&
      update.previousMessage.role === update.message.role &&
      update.previousMessage.hidden === update.message.hidden &&
      update.previousMessage.branchGroupId === update.message.branchGroupId &&
      previous.tailItem?.message.id === update.message.id &&
      previous.repository.headId === update.message.id
    ) {
      const cachedMessage = cacheRef.current.get(update.message)
      const runtimeMessage = cachedMessage ?? toRuntimeMessage(update.message)

      if (!cachedMessage) {
        cacheRef.current.set(update.message, runtimeMessage)
      }

      const tailItem = { message: runtimeMessage, parentId: previous.tailItem.parentId }

      const repository: RuntimeMessageRepository = {
        headId: update.message.id,
        messages: previous.repository.messages,
        [runtimeMessageRepositoryDelta]: tailItem
      }

      repositoryCacheRef.current = { repository, source: messages, tailItem, tailSource: nextTail }

      return repository
    }

    const items: { message: ThreadMessage; parentId: string | null }[] = []
    const branchParentByGroup = new Map<string, string | null>()
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

      items.push({ message: runtimeMessage, parentId })

      if (!message.hidden) {
        visibleParentId = message.id
        headId = message.id
      }
    }

    // toRuntimeMessage already returns normalized ThreadMessage objects. Keep
    // those cached references intact instead of normalizing the whole history a
    // second time on every streamed delta.
    const repository: RuntimeMessageRepository = { headId, messages: items }
    const tailItem = nextTail && items.at(-1)?.message.id === nextTail.id ? (items.at(-1) ?? null) : null
    repositoryCacheRef.current = { repository, source: messages, tailItem, tailSource: nextTail }

    return repository
  }, [messages])
}
