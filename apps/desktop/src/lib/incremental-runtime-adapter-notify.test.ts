/**
 * Regression tests for the bot-chat tile crash observed live on 2026-08-23:
 *
 *   [error-boundary:contrib:session-tile:20260823_213059_ed6222]
 *   Error: Maximum update depth exceeded. The result of getSnapshot should be
 *   cached to avoid an infinite loop.
 *     at UseTapEffects (assistant-ui internal)
 *     at AuiProvider
 *     at ChatRuntimeBoundary (src/app/chat/index.tsx)
 *     at TileChat / SessionTilePane (src/app/chat/session-tile.tsx)
 *
 * Mechanism under test: ChatRuntimeBoundary passes a FRESH adapter object
 * literal to useIncrementalExternalStoreRuntime on every render. The
 * [runtime, store] effect therefore re-runs each render, and
 * IncrementalExternalStoreThreadRuntimeCore.__internal_setAdapter's
 * "nothing changed" fast path (same isRunning + same messageRepository)
 * still calls _notifySubscribers() unconditionally. A subscriber whose
 * notification re-renders the boundary then produces:
 *   render -> new store literal -> effect -> setAdapter -> notify -> render
 * an unbounded feedback loop that trips React's update-depth guard and
 * takes the whole session tile down with the error boundary.
 *
 * The invariant pinned here: handing __internal_setAdapter an adapter that is
 * a NEW OBJECT but semantically identical (same messageRepository identity,
 * same isRunning, same capabilities) must NOT notify subscribers. Notify is
 * for observable state changes, not for object-identity churn of the adapter
 * literal — the render loop is impossible once no-op swaps are silent.
 */
import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import type { ExportedMessageRepository, ExternalStoreAdapter, ThreadMessage } from '@assistant-ui/react'
import { describe, expect, it } from 'vitest'

import { IncrementalExternalStoreRuntimeCore } from './incremental-external-store-runtime'

const STATUS = getAutoStatus(false, false, false, false, undefined)

function message(id: string, text: string): ThreadMessage {
  return fromThreadMessageLike({ role: 'assistant', content: [{ type: 'text', text }] }, id, STATUS)
}

function repositoryOf(messages: ThreadMessage[]): ExportedMessageRepository {
  return {
    headId: messages.at(-1)?.id ?? null,
    messages: messages.map((item, index) => ({
      message: item,
      parentId: index === 0 ? null : messages[index - 1].id
    }))
  }
}

function adapterWith(messageRepository: ExportedMessageRepository, extra: Partial<ExternalStoreAdapter> = {}) {
  // Deliberately a fresh object each call — mirrors the inline literal in
  // ChatRuntimeBoundary (src/app/chat/index.tsx) whose closures (onNew,
  // onCancel, ...) are re-created per render.
  return {
    messageRepository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onCancel: async () => {},
    ...extra
  } as ExternalStoreAdapter
}

describe('IncrementalExternalStoreThreadRuntimeCore adapter swap notifications', () => {
  it('does NOT notify subscribers when a new adapter object carries identical state (render-loop guard)', () => {
    const repo = repositoryOf([message('a', 'one'), message('b', 'two')])
    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repo))
    const thread = core.threads.getMainThreadRuntimeCore()

    let notifications = 0
    thread.subscribe(() => {
      notifications += 1
    })

    // Simulate 5 renders of ChatRuntimeBoundary: each passes a NEW adapter
    // literal with the SAME messageRepository identity and same isRunning.
    for (let render = 0; render < 5; render += 1) {
      core.setAdapter(adapterWith(repo))
    }

    // The live bug: this was 5 — one notify per render, each notify able to
    // schedule the next render. Silent no-op swaps break the feedback loop.
    expect(notifications).toBe(0)
  })

  it('still notifies when the message repository actually changes', () => {
    const repo = repositoryOf([message('a', 'one')])
    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repo))
    const thread = core.threads.getMainThreadRuntimeCore()

    let notifications = 0
    thread.subscribe(() => {
      notifications += 1
    })

    const grown = repositoryOf([message('a', 'one'), message('b', 'two')])
    core.setAdapter(adapterWith(grown))

    expect(notifications).toBeGreaterThan(0)
  })

  it('still notifies on an isRunning flip (turn start/stop must reach the thread UI)', () => {
    const repo = repositoryOf([message('a', 'one')])
    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repo))
    const thread = core.threads.getMainThreadRuntimeCore()

    let notifications = 0
    thread.subscribe(() => {
      notifications += 1
    })

    core.setAdapter(adapterWith(repo, { isRunning: true }))

    expect(notifications).toBeGreaterThan(0)
  })

  it('still notifies on an isDisabled flip even when transcript and run state are unchanged', () => {
    const repo = repositoryOf([message('a', 'one')])
    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repo))
    const thread = core.threads.getMainThreadRuntimeCore()

    let notifications = 0
    thread.subscribe(() => {
      notifications += 1
    })

    core.setAdapter(adapterWith(repo, { isDisabled: true }))

    expect(notifications).toBeGreaterThan(0)

    // And flipping back also notifies — but an unchanged repeat stays silent.
    const beforeFlipBack = notifications
    core.setAdapter(adapterWith(repo, { isDisabled: false }))

    expect(notifications).toBeGreaterThan(beforeFlipBack)

    const afterFlipBack = notifications

    core.setAdapter(adapterWith(repo, { isDisabled: false }))

    expect(notifications).toBe(afterFlipBack)
  })

  it('a subscriber that swaps a fresh-but-identical adapter on every notify must not recurse unboundedly', () => {
    // Direct simulation of the feedback loop: the subscriber plays the role of
    // React re-rendering ChatRuntimeBoundary (new literal -> setAdapter). With
    // the bug, every setAdapter notifies, the subscriber re-enters setAdapter,
    // and only the depth guard stops it. Fixed behavior: the first no-op swap
    // is silent, so the subscriber never fires and depth stays 0.
    const repo = repositoryOf([message('a', 'one')])
    const core = new IncrementalExternalStoreRuntimeCore(adapterWith(repo))
    const thread = core.threads.getMainThreadRuntimeCore()

    let depth = 0
    thread.subscribe(() => {
      depth += 1

      if (depth > 25) {
        throw new Error('Maximum update depth exceeded (simulated): adapter no-op swaps are notifying subscribers')
      }

      core.setAdapter(adapterWith(repo))
    })

    core.setAdapter(adapterWith(repo))

    expect(depth).toBe(0)
  })
})
