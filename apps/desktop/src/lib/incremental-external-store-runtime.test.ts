import { fromThreadMessageLike, getAutoStatus, MessageRepository } from '@assistant-ui/core/internal'
import type { ExportedMessageRepository, ThreadMessage } from '@assistant-ui/react'
import { describe, expect, it, vi } from 'vitest'

import { syncRepositoryIncrementally } from './incremental-external-store-runtime'

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
