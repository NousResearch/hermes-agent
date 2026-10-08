import { fromThreadMessageLike, getAutoStatus, MessageRepository } from '@assistant-ui/core/internal'
import { ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { syncRepositoryIncrementally, useIncrementalExternalStoreRuntime } from './incremental-external-store-runtime'

const STATUS = getAutoStatus(false, false, false, false, undefined)

function message(id: string, text: string): ThreadMessage {
  return fromThreadMessageLike({ role: 'assistant', content: [{ type: 'text', text }] }, id, STATUS)
}

/** A real MessageRepository behind the same shape syncRepositoryIncrementally drives. */
function runtimeWith(items: { message: ThreadMessage; parentId: string | null }[]) {
  const repository = new MessageRepository()

  for (const { message: item, parentId } of items) {
    repository.addOrUpdateMessage(parentId, item)
  }

  if (items.length > 0) {
    repository.resetHead(items.at(-1)?.message.id ?? null)
  }

  return { repository } as unknown as Parameters<typeof syncRepositoryIncrementally>[0]
}

function chain(messages: ThreadMessage[]) {
  return messages.map((item, index) => ({
    message: item,
    parentId: index === 0 ? null : messages[index - 1].id
  }))
}

function exported(items: { message: ThreadMessage; parentId: string | null }[]): ExportedMessageRepository {
  return { headId: items.at(-1)?.message.id ?? null, messages: items }
}

describe('syncRepositoryIncrementally', () => {
  it('writes only the changed tail instead of the whole transcript', () => {
    const settled = Array.from({ length: 200 }, (_, index) => message(`m-${index}`, `body ${index}`))
    const items = chain(settled)
    const runtime = runtimeWith(items)
    const repository = (runtime as unknown as { repository: MessageRepository }).repository

    const addOrUpdate = vi.spyOn(repository, 'addOrUpdateMessage')
    const resetHead = vi.spyOn(repository, 'resetHead')

    // One streamed delta: the tail grows, every settled message keeps identity.
    const nextTail = message('m-199', 'body 199 + delta')
    const nextItems = [...items.slice(0, -1), { message: nextTail, parentId: 'm-198' }]

    const result = syncRepositoryIncrementally(runtime, exported(nextItems))

    expect(addOrUpdate).toHaveBeenCalledTimes(1)
    expect(addOrUpdate).toHaveBeenCalledWith('m-198', nextTail)
    // The head did not move, so the descendant-pruning reset is skipped.
    expect(resetHead).not.toHaveBeenCalled()
    expect(result).toHaveLength(200)
    expect(result.at(-1)).toBe(nextTail)
  })

  it('does nothing at all when the transcript is unchanged', () => {
    const items = chain([message('a', 'one'), message('b', 'two')])
    const runtime = runtimeWith(items)
    const repository = (runtime as unknown as { repository: MessageRepository }).repository

    const addOrUpdate = vi.spyOn(repository, 'addOrUpdateMessage')
    const deleteMessage = vi.spyOn(repository, 'deleteMessage')

    syncRepositoryIncrementally(runtime, exported(items))

    expect(addOrUpdate).not.toHaveBeenCalled()
    expect(deleteMessage).not.toHaveBeenCalled()
  })

  it('appends a new message through the full path', () => {
    const first = message('a', 'one')
    const items = chain([first])
    const runtime = runtimeWith(items)

    const second = message('b', 'two')
    const result = syncRepositoryIncrementally(runtime, exported(chain([first, second])))

    expect(result.map(item => item.id)).toEqual(['a', 'b'])
  })

  it('honours an authoritative deletion', () => {
    const a = message('a', 'one')
    const b = message('b', 'two')
    const c = message('c', 'three')
    const runtime = runtimeWith(chain([a, b, c]))

    const result = syncRepositoryIncrementally(runtime, exported(chain([a, b])))

    expect(result.map(item => item.id)).toEqual(['a', 'b'])
  })

  it('rebuilds cleanly when a disjoint transcript is swapped in', () => {
    const runtime = runtimeWith(chain([message('old-1', 'one'), message('old-2', 'two')]))

    const next = chain([message('new-1', 'alpha'), message('new-2', 'beta')])
    const result = syncRepositoryIncrementally(runtime, exported(next))

    expect(result.map(item => item.id)).toEqual(['new-1', 'new-2'])
  })

  it('re-parents a message when its branch parent changes', () => {
    const root = message('root', 'root')
    const a = message('a', 'a')
    const b = message('b', 'b')

    const runtime = runtimeWith([
      { message: root, parentId: null },
      { message: a, parentId: 'root' },
      { message: b, parentId: 'a' }
    ])

    // Same ids and same message objects, but `b` moves onto a sibling branch.
    const result = syncRepositoryIncrementally(runtime, {
      headId: 'b',
      messages: [
        { message: root, parentId: null },
        { message: a, parentId: 'root' },
        { message: b, parentId: 'root' }
      ]
    })

    expect(result.map(item => item.id)).toEqual(['root', 'b'])
  })

  it('moves the head when an explicit headId rewinds the branch', () => {
    const a = message('a', 'one')
    const b = message('b', 'two')
    const runtime = runtimeWith(chain([a, b]))

    const result = syncRepositoryIncrementally(runtime, {
      headId: 'a',
      messages: chain([a, b])
    })

    expect(result.map(item => item.id)).toEqual(['a'])
  })
})

const optimisticCreatedAt = new Date('2026-05-01T00:00:00.000Z')

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'do the thing' }],
    attachments: [],
    createdAt: optimisticCreatedAt,
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

// Cancelling a run before the first assistant token must not poison the next
// adapter sync.
//
// The runtime appends an optimistic assistant placeholder while a run is in
// flight. Core owns that placeholder's lifetime: `cancelRun()` deletes it by
// looking up the head, and `resetHead` evicts off-branch optimistic messages.
// Any placeholder id this module holds onto across calls therefore goes stale
// the moment core removes the message, and re-deleting a stale id throws
// `MessageRepository(deleteMessage): Message not found`.
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
