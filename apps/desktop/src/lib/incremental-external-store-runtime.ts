import {
  AssistantRuntimeImpl,
  BaseAssistantRuntimeCore,
  ExternalStoreThreadListRuntimeCore,
  ExternalStoreThreadRuntimeCore,
  hasUpcomingMessage
} from '@assistant-ui/core/internal'
import {
  type AssistantRuntime,
  type ExternalStoreAdapter,
  fromThreadMessageLike,
  generateId,
  type ThreadMessage,
  useRuntimeAdapters
} from '@assistant-ui/react'
import { useEffect, useMemo, useState } from 'react'

const EMPTY_ARRAY = Object.freeze([])

const shallowEqual = (a: object, b: object): boolean => {
  const aKeys = Object.keys(a)

  if (aKeys.length !== Object.keys(b).length) {
    return false
  }

  for (const key of aKeys) {
    if (a[key as keyof typeof a] !== b[key as keyof typeof b]) {
      return false
    }
  }

  return true
}

const getThreadListAdapter = (store: ExternalStoreAdapter) => store.adapters?.threadList ?? {}

/**
 * Write only the items whose (message, parentId) pair actually moved.
 *
 * `useRuntimeMessageRepository` caches normalized ThreadMessages by source
 * identity, so a settled turn keeps the SAME object across renders. That makes
 * an identity check a sound "did this change?" test: during streaming exactly
 * one item — the growing tail — differs, and the other N-1 writes were pure
 * overhead that grew with transcript length.
 *
 * Returns false when the export is stale (an id in `existing` is gone, or an
 * incoming message has no repository entry yet), so the caller falls back to
 * the full rebuild rather than guessing.
 */
function applyChangedMessages(
  repository: ExternalStoreThreadRuntimeCore['repository'],
  existing: readonly { message: ThreadMessage; parentId: string | null }[],
  incoming: readonly { message: ThreadMessage; parentId: string | null }[]
): boolean {
  if (existing.length !== incoming.length) {
    return false
  }

  const existingById = new Map(existing.map(item => [item.message.id, item]))

  for (const item of incoming) {
    const current = existingById.get(item.message.id)

    if (!current) {
      return false
    }

    // Reference identity, not deep equality: the conversion cache guarantees a
    // stable object for an unchanged turn, and a changed turn is a new object.
    if (current.message !== item.message || current.parentId !== item.parentId) {
      repository.addOrUpdateMessage(item.parentId, item.message)
    }
  }

  return true
}

/**
 * True when a message carries user-visible text — the floor for pinning a
 * message across an adapter sync. Empty rows (optimistic placeholders, bare
 * tool-only rows) stay ephemeral.
 */
function hasVisibleText(message: ThreadMessage): boolean {
  return message.content.some(part => part.type === 'text' && part.text.trim() !== '')
}

/**
 * While a turn is running, a store snapshot can briefly omit the in-flight
 * streamed row (#119686): a rewrite/tool-phase snapshot, an interrupt-driven
 * re-hydration or a compaction refresh all predate the live turn. Reconciling
 * "absent from the incoming set" as "must be deleted" wiped the bubble the
 * user had just watched stream. The reconcile still has to delete genuinely
 * abandoned rows, so the pin is deliberately narrow: only assistant messages
 * with user-visible text ON THE VISIBLE HEAD BRANCH that are newer than
 * everything in the incoming snapshot (the last synced anchor). A replacement
 * reply the store has already accepted carries a later-or-equal timestamp, so
 * it lands outside the pin window and the abandoned draft prunes as before.
 *
 * Returns the pinned ids plus the deepest pinned id — the reconcile's
 * `resetHead(incoming head)` would otherwise delete the pinned tail as a
 * descendant of the incoming head, so the head must move to the pinned tail
 * instead while it survives.
 */
function pinnedLiveTail(
  repository: ExternalStoreThreadRuntimeCore['repository'],
  incomingIds: ReadonlySet<string>,
  incoming: readonly { message: ThreadMessage; parentId: string | null }[]
): { ids: Set<string>; headId: string | null } {
  // The incoming snapshot still owes this turn's reply when it does not end
  // with a non-empty assistant row: the reply the user watched stream has not
  // landed in the store yet (mid-rewrite snapshot, interrupt re-hydration,
  // compaction refresh). Timestamps alone cannot draw that line — a fast
  // reply can land within the same second (or millisecond) the draft started,
  // and a missing ChatMessage.timestamp makes createdAt a fresh Date.now()
  // (milliseconds) next to second-granularity anchors. Structure first,
  // timestamps only to keep a strictly-newer draft when a replacement reply
  // HAS landed (multi-turn: the snapshot ends with the previous turn's reply
  // while the new turn's row is still in flight).
  const lastIncoming = incoming.at(-1)?.message
  const replyLanded = Boolean(
    lastIncoming && lastIncoming.role === 'assistant' && hasVisibleText(lastIncoming)
  )

  // Anchor = the NEWEST incoming message still present in the repository. A
  // snapshot that has not caught up drops its old tail, so its newest
  // remaining row predates the live streamed turn; anchor on what actually
  // made it into the tree, not on what the snapshot claims is newest.
  let anchorTime = 0

  for (const { message } of incoming) {
    if (repositoryHasId(repository, message.id)) {
      anchorTime = Math.max(anchorTime, message.createdAt?.getTime() ?? 0)
    }
  }

  // The head branch is what the user is looking at; off-branch drafts stay
  // prunable.
  const headBranch = repository.getMessages()
  const ids = new Set<string>()
  let headId: string | null = null

  for (const message of headBranch) {
    if (
      !incomingIds.has(message.id) &&
      message.role === 'assistant' &&
      hasVisibleText(message) &&
      (!replyLanded || (message.createdAt?.getTime() ?? 0) > anchorTime)
    ) {
      ids.add(message.id)
      headId = message.id
    }
  }

  return { ids, headId }
}

function repositoryHasId(
  repository: ExternalStoreThreadRuntimeCore['repository'],
  id: string
): boolean {
  return repository.export().messages.some(({ message }) => message.id === id)
}

export function syncRepositoryIncrementally(
  runtime: ExternalStoreThreadRuntimeCore,
  messageRepository: NonNullable<ExternalStoreAdapter['messageRepository']>,
  { pinLiveTail = false }: { pinLiveTail?: boolean } = {}
): readonly ThreadMessage[] {
  const repository = (runtime as unknown as { repository: ExternalStoreThreadRuntimeCore['repository'] }).repository
  const incoming = messageRepository.messages
  const existing = repository.export().messages
  const headId = messageRepository.headId ?? incoming.at(-1)?.message.id ?? null

  // A thread switch swaps in a fully-DISJOINT transcript (no id carries over).
  // Reconciling two unrelated trees in place — grafting the new chain onto the
  // old one, then pruning — can strand a stale head/branch, so there's nothing
  // to preserve: clear the tree first (leaves→root), then rebuild clean.
  const incomingIds = new Set(incoming.map(({ message }) => message.id))
  const disjoint = existing.length > 0 && !existing.some(({ message }) => incomingIds.has(message.id))
  // Mid-run, an incoming snapshot that omits the live streamed tail must not
  // delete it (#119686). The pin only ever applies to the reconcile below —
  // a fully disjoint thread switch still clears the tree (the transcript
  // moved to an unrelated conversation).
  const pinned = pinLiveTail ? pinnedLiveTail(repository, incomingIds, incoming) : { ids: new Set<string>(), headId: null }

  // Steady-state streaming: same message set, one item changed. Skip the
  // whole-transcript rewrite, the prune scan, and the second export. resetHead
  // deletes the head's descendants, so it only runs when the head really moved.
  if (!disjoint && applyChangedMessages(repository, existing, incoming)) {
    if (repository.headId !== headId) {
      repository.resetHead(headId)
    }

    return repository.getMessages()
  }

  if (disjoint) {
    for (const { message } of [...existing].reverse()) {
      repository.deleteMessage(message.id)
    }
  }

  for (const { message, parentId } of incoming) {
    repository.addOrUpdateMessage(parentId, message)
  }

  for (const { message } of repository.export().messages) {
    if (!incomingIds.has(message.id) && !pinned.ids.has(message.id)) {
      repository.deleteMessage(message.id)
    }
  }

  // While the pinned tail survives, the visible head stays on it: resetting
  // to the incoming head would delete the pinned draft as its descendant.
  repository.resetHead(pinned.headId ?? headId)

  return repository.getMessages()
}

class IncrementalExternalStoreThreadRuntimeCore extends ExternalStoreThreadRuntimeCore {
  override __internal_setAdapter(store: ExternalStoreAdapter): void {
    if (!store.messageRepository) {
      super.__internal_setAdapter(store)

      return
    }

    const self = this as unknown as {
      _assistantOptimisticId: null | string
      _capabilities: object
      _messages: readonly ThreadMessage[]
      _notifyEventSubscribers: (event: string, payload: object) => void
      _notifySubscribers: () => void
      _store?: ExternalStoreAdapter
    }

    if (self._store === store) {
      return
    }

    const isRunning = store.isRunning ?? false
    const newDisabled = store.isDisabled ?? false
    const disabledChanged = this.isDisabled !== newDisabled
    this.isDisabled = newDisabled

    const oldStore = self._store
    self._store = store

    // Track whether anything OBSERVABLE changed. ChatRuntimeBoundary passes a
    // fresh adapter literal on every render, so identity churn of the adapter
    // object itself is NOT a change — notifying on it lets a subscriber whose
    // notification re-renders the boundary drive an unbounded feedback loop
    // (render -> new literal -> setAdapter -> notify -> render), which React
    // kills with "Maximum update depth exceeded" and takes the session tile
    // down with its error boundary.
    let changed = disabledChanged

    if (this.extras !== store.extras) {
      this.extras = store.extras
      changed = true
    }

    const newSuggestions = store.suggestions ?? EMPTY_ARRAY

    if (!shallowEqual(this.suggestions, newSuggestions)) {
      this.suggestions = newSuggestions
      changed = true
    }

    const newCapabilities = {
      switchToBranch: store.setMessages !== undefined,
      switchBranchDuringRun: false,
      edit: store.onEdit !== undefined,
      reload: store.onReload !== undefined,
      cancel: store.onCancel !== undefined,
      speech: store.adapters?.speech !== undefined,
      dictation: store.adapters?.dictation !== undefined,
      voice: store.adapters?.voice !== undefined,
      unstable_copy: store.unstable_capabilities?.copy !== false,
      attachments: !!store.adapters?.attachments,
      feedback: !!store.adapters?.feedback,
      queue: false
    }

    if (!shallowEqual(self._capabilities, newCapabilities)) {
      self._capabilities = newCapabilities
      changed = true
    }

    if (oldStore && oldStore.isRunning === store.isRunning && oldStore.messageRepository === store.messageRepository) {
      // Same transcript, same run state: notify only if extras/suggestions/
      // capabilities actually moved. A silent no-op swap here is what breaks
      // the render feedback loop — see the render-loop guard test.
      if (changed) {
        self._notifySubscribers()
      }

      return
    }

    if (self._assistantOptimisticId) {
      this.repository.deleteMessage(self._assistantOptimisticId)
      self._assistantOptimisticId = null
    }

    // Mid-run, the incoming snapshot can briefly omit the live streamed tail
    // (#119686) — pin it so the reconcile below cannot delete a non-empty
    // streamed reply. Not running: the snapshot is the transcript truth.
    const messages = syncRepositoryIncrementally(this, store.messageRepository, { pinLiveTail: isRunning })

    if (messages.length > 0) {
      this.ensureInitialized()
    }

    if ((oldStore?.isRunning ?? false) !== (store.isRunning ?? false)) {
      self._notifyEventSubscribers(store.isRunning ? 'runStart' : 'runEnd', {})
    }

    // metadata.isOptimistic keeps this placeholder ephemeral: core evicts
    // off-branch optimistic messages on head moves and omits them from export().
    if (hasUpcomingMessage(isRunning, messages)) {
      const optimisticId = generateId()
      this.repository.addOrUpdateMessage(
        messages.at(-1)?.id ?? null,
        fromThreadMessageLike({ role: 'assistant', content: [], metadata: { isOptimistic: true } }, optimisticId, {
          type: 'running'
        })
      )
      self._assistantOptimisticId = optimisticId
    }

    this.repository.resetHead(self._assistantOptimisticId ?? messages.at(-1)?.id ?? null)
    self._messages = this.repository.getMessages()
    self._notifySubscribers()
  }
}

export class IncrementalExternalStoreRuntimeCore extends BaseAssistantRuntimeCore {
  threads: ExternalStoreThreadListRuntimeCore

  constructor(adapter: ExternalStoreAdapter) {
    super()

    this.threads = new ExternalStoreThreadListRuntimeCore(
      getThreadListAdapter(adapter),
      () => new IncrementalExternalStoreThreadRuntimeCore(this._contextProvider, adapter)
    )
  }

  setAdapter(adapter: ExternalStoreAdapter): void {
    this.threads.__internal_setAdapter(getThreadListAdapter(adapter))
    this.threads.getMainThreadRuntimeCore().__internal_setAdapter(adapter)
  }
}

export function useIncrementalExternalStoreRuntime<T extends ThreadMessage>(
  store: ExternalStoreAdapter<T>
): AssistantRuntime {
  const [runtime] = useState(() => new IncrementalExternalStoreRuntimeCore(store as ExternalStoreAdapter))

  // Re-sync the adapter only when it actually changes — a dep-less effect ran
  // on EVERY render of the chat surface. `__internal_setAdapter` early-exits
  // when the store is unchanged, so gating on [runtime, store] is behavior-
  // preserving while skipping the per-render call entirely.
  useEffect(() => {
    runtime.setAdapter(store as ExternalStoreAdapter)
  }, [runtime, store])

  const { modelContext } = useRuntimeAdapters() ?? {}

  useEffect(() => {
    if (!modelContext) {
      return undefined
    }

    return runtime.registerModelContextProvider(modelContext)
  }, [modelContext, runtime])

  return useMemo(() => new AssistantRuntimeImpl(runtime), [runtime])
}
