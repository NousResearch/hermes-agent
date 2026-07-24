// Cancelling a run before the first assistant token must not poison the next
// adapter sync.
//
// The runtime appends an optimistic assistant placeholder while a run is in
// flight. Core owns that placeholder's lifetime: `cancelRun()` deletes it by
// looking up the head, and `resetHead` evicts off-branch optimistic messages.
// Any placeholder id this module holds onto across calls therefore goes stale
// the moment core removes the message, and re-deleting a stale id throws
// `MessageRepository(deleteMessage): Message not found`.
import { ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from './incremental-external-store-runtime'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'do the thing' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as unknown as ThreadMessage
}

// Mirrors chat/index.tsx: incremental runtime driven by a messageRepository,
// with `isRunning` tracking the gateway's busy flag and onCancel wired up.
function renderRuntime(onCancel: () => Promise<void>) {
  const repository = ExportedMessageRepository.fromArray([userMessage()])

  return renderHook(
    ({ isRunning }: { isRunning: boolean }) =>
      useIncrementalExternalStoreRuntime<ThreadMessage>({
        messageRepository: repository,
        isRunning,
        setMessages: () => {},
        onNew: async () => {},
        onCancel,
        onReload: async () => {}
      }),
    { initialProps: { isRunning: true } }
  )
}

describe('incremental external store runtime — cancel before first token', () => {
  it('survives the sync that follows a cancelled run', async () => {
    const onCancel = vi.fn(async () => {})
    const { result, rerender } = renderRuntime(onCancel)

    // A run is in flight with a trailing user message, so the runtime has
    // appended an optimistic assistant placeholder.
    expect(result.current.thread.getState().isRunning).toBe(true)

    // User presses Stop before the first assistant token arrives. Core deletes
    // the empty placeholder by lookup and clears no state of ours.
    await act(async () => {
      result.current.thread.cancelRun()
    })

    expect(onCancel).toHaveBeenCalledTimes(1)

    // The gateway flips busy -> false, which re-syncs the adapter. Before the
    // fix this re-deleted the now-stale placeholder id and threw.
    expect(() => {
      rerender({ isRunning: false })
    }).not.toThrow()
  })

  it('still appends a placeholder while a run is in flight', async () => {
    const { result } = renderRuntime(async () => {})

    // Negative control: the placeholder itself must survive this change --
    // removing the stale field must not remove the optimistic message.
    const messages = result.current.thread.getState().messages

    expect(messages.at(-1)?.role).toBe('assistant')
    expect(messages.at(-1)?.content).toEqual([])
  })

  it('leaves no placeholder behind once the run ends normally', async () => {
    const { result, rerender } = renderRuntime(async () => {})

    rerender({ isRunning: false })

    // Negative control: core's own eviction (resetHead ->
    // evictOffBranchOptimisticMessages) still cleans the placeholder up
    // without this module tracking its id.
    const messages = result.current.thread.getState().messages

    expect(messages).toHaveLength(1)
    expect(messages[0]?.role).toBe('user')
  })
})
