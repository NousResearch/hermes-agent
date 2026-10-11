/**
 * Regression tests for the adapter-sync prune that deleted a LIVE streamed
 * assistant bubble (#119686):
 *
 * The desktop chat store feeds `useIncrementalExternalStoreRuntime` a fresh
 * adapter literal per render whose `messageRepository` is rebuilt by
 * `useRuntimeMessageRepository(currentMessages)`. During a rewrite/tool phase
 * the store snapshot can briefly omit the in-flight streamed turn (compaction
 * refresh, ws_orphan_reap interrupt, busy_input_mode rewrite) — and the
 * reconcile treated "absent from the incoming set" as "must be deleted",
 * wiping the non-empty streamed text until a later reply landed. Users saw
 * the bubble appear while streaming and then vanish.
 *
 * Invariant pinned here: while a run is in progress, an adapter sync whose
 * incoming message set no longer contains the repository's live tail — a
 * non-empty assistant message — must KEEP that message (and its subtree)
 * rather than delete it. The pinned tail is released by a later sync that
 * includes it again (the ordinary path), or by a non-running sync (turn
 * settled/interrupted: the owner decided the transcript truth).
 */
import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import type { ExportedMessageRepository, ExternalStoreAdapter, ThreadMessage } from '@assistant-ui/react'
import { describe, expect, it } from 'vitest'

import { IncrementalExternalStoreRuntimeCore } from './incremental-external-store-runtime'

const STATUS = getAutoStatus(false, false, false, false, undefined)
const RUNNING_STATUS = { type: 'running' } as const

function message(id: string, text: string, status: typeof STATUS | typeof RUNNING_STATUS = STATUS): ThreadMessage {
  return fromThreadMessageLike({ role: 'assistant', content: [{ type: 'text', text }] }, id, status)
}

function userMessage(id: string, text: string): ThreadMessage {
  return fromThreadMessageLike({ role: 'user', content: [{ type: 'text', text }] }, id, STATUS)
}

function repositoryOf(messages: ThreadMessage[], headId?: string): ExportedMessageRepository {
  return {
    headId: headId ?? messages.at(-1)?.id ?? null,
    messages: messages.map((item, index) => ({
      message: item,
      parentId: index === 0 ? null : messages[index - 1].id
    }))
  }
}

function adapterWith(messageRepository: ExportedMessageRepository, extra: Partial<ExternalStoreAdapter> = {}) {
  // Deliberately a fresh object each call — mirrors the inline literal in
  // ChatRuntimeBoundary whose closures are re-created per render.
  return {
    messageRepository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onCancel: async () => {},
    ...extra
  } as ExternalStoreAdapter
}

function visibleText(core: IncrementalExternalStoreRuntimeCore): string[] {
  const thread = core.threads.getMainThreadRuntimeCore()
  const messages = (thread as unknown as { _messages: readonly ThreadMessage[] })._messages

  return messages
    .map(item =>
      item.content
        .filter(part => part.type === 'text')
        .map(part => (part as { text: string }).text)
        .join('')
    )
    .filter(text => text !== '')
}

describe('adapter sync keeps the live streamed tail when a mid-run snapshot omits it (#119686)', () => {
  it('does not delete a non-empty streamed assistant tail absent from the incoming set while running', () => {
    const user = userMessage('u1', 'question')
    const streamedTail = message('a-stream', 'partial answer streamed so far', RUNNING_STATUS)

    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repositoryOf([user, streamedTail], 'a-stream')))

    // Mid-run adapter sync: the incoming repository omits the streamed turn
    // (e.g. a compaction/rewrite snapshot that has not caught up yet).
    const snapshotWithoutTail = repositoryOf([user])

    core.setAdapter(adapterWith(snapshotWithoutTail, { isRunning: true }))

    // The streamed bubble must survive — the user watched it stream; deleting
    // it reads as the agent erasing its own reply.
    expect(visibleText(core)).toEqual(['question', 'partial answer streamed so far'])
  })

  it('still reconciles deletions once the run has settled (no run in progress)', () => {
    const user = userMessage('u1', 'question')
    const stale = message('a-stale', 'abandoned row')

    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repositoryOf([user, stale])))

    // Not running: the snapshot is the transcript truth. A row that is gone
    // from the incoming set is a real deletion and must be pruned.
    core.setAdapter(adapterWith(repositoryOf([user])))

    expect(visibleText(core)).toEqual(['question'])
  })

  it('releases the pinned tail when a later sync includes it again', () => {
    const user = userMessage('u1', 'question')
    const streamedTail = message('a-stream', 'partial answer', RUNNING_STATUS)

    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repositoryOf([user, streamedTail], 'a-stream')))

    core.setAdapter(adapterWith(repositoryOf([user]), { isRunning: true }))

    // The turn settles and the store snapshot now contains the final text.
    const finalTail = message('a-final', 'the full answer')

    core.setAdapter(adapterWith(repositoryOf([user, finalTail], 'a-final')))

    expect(visibleText(core)).toEqual(['question', 'the full answer'])
  })

  it('drops the abandoned draft once a replacement reply has landed (same-second rewrite)', () => {
    const user = userMessage('u1', 'question')
    const streamedTail = message('a-stream', 'partial answer', RUNNING_STATUS)

    // Force identical millisecond timestamps: a rewrite that lands the same
    // instant the draft started. The pin must yield to the store's truth.
    ;(streamedTail as { createdAt: Date }).createdAt = user.createdAt

    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repositoryOf([user, streamedTail], 'a-stream')))

    core.setAdapter(adapterWith(repositoryOf([user]), { isRunning: true }))

    // The rewrite lands with its OWN id while the run is still marked running.
    const rewritten = message('a-final', 'rewritten answer', RUNNING_STATUS)
    ;(rewritten as { createdAt: Date }).createdAt = user.createdAt

    core.setAdapter(adapterWith(repositoryOf([user, rewritten], 'a-final')))
    // One reply, not the draft + the rewrite.
    expect(visibleText(core)).toEqual(['question', 'rewritten answer'])
  })
})
