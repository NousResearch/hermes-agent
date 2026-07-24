import { type ExportedMessageRepository, fromThreadMessageLike, type ThreadMessage } from '@assistant-ui/react'
import { describe, expect, it, vi } from 'vitest'

import {
  type RuntimeMessageRepository,
  runtimeMessageRepositoryDelta
} from '@/app/chat/runtime-repository'

import {
  getThreadMessageListToken,
  getThreadMessageListUpdate,
  syncRepositoryIncrementally
} from './incremental-external-store-runtime'

function message(id: string, text: string): ThreadMessage {
  return fromThreadMessageLike({ role: 'assistant', content: [{ type: 'text', text }] }, id, {
    type: 'complete',
    reason: 'stop'
  })
}

function repositoryMock(existing: ExportedMessageRepository) {
  const repository = {
    addOrUpdateMessage: vi.fn(),
    deleteMessage: vi.fn(),
    export: vi.fn(() => existing),
    getMessage: vi.fn((id: string) => {
      const item = existing.messages.find(({ message: existingMessage }) => existingMessage.id === id)

      if (!item) {
        throw new Error(`Missing message: ${id}`)
      }

      return { ...item, index: existing.messages.indexOf(item) }
    }),
    getMessages: vi.fn(() => existing.messages.map(item => item.message)),
    headId: existing.headId,
    resetHead: vi.fn()
  }

  return repository
}

describe('syncRepositoryIncrementally', () => {
  it('does not rewrite stable messages or reset an unchanged head', () => {
    const first = message('assistant-1', 'first')
    const tail = message('assistant-2', 'tail')

    const incoming: ExportedMessageRepository = {
      headId: tail.id,
      messages: [
        { message: first, parentId: null },
        { message: tail, parentId: first.id }
      ]
    }

    const repository = repositoryMock(incoming)

    syncRepositoryIncrementally(repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0], incoming)

    expect(repository.addOrUpdateMessage).not.toHaveBeenCalled()
    expect(repository.deleteMessage).not.toHaveBeenCalled()
    expect(repository.resetHead).not.toHaveBeenCalled()
    expect(repository.getMessages).toHaveBeenCalledOnce()
  })

  it('updates only the changed stream message', () => {
    const first = message('assistant-1', 'first')
    const previousTail = message('assistant-2', 'A')
    const nextTail = message('assistant-2', 'AB')

    const existing: ExportedMessageRepository = {
      headId: previousTail.id,
      messages: [
        { message: first, parentId: null },
        { message: previousTail, parentId: first.id }
      ]
    }

    const incoming: ExportedMessageRepository = {
      headId: nextTail.id,
      messages: [
        { message: first, parentId: null },
        { message: nextTail, parentId: first.id }
      ]
    }

    const repository = repositoryMock(existing)

    syncRepositoryIncrementally(repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0], incoming)

    expect(repository.addOrUpdateMessage).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(first.id, nextTail)
    expect(repository.deleteMessage).not.toHaveBeenCalled()
    expect(repository.resetHead).not.toHaveBeenCalled()
  })

  it('applies a pending-tail delta without exporting or scanning the settled repository', () => {
    const first = message('assistant-1', 'first')
    const previousTail = message('assistant-2', 'A')
    const nextTail = message('assistant-2', 'AB')

    const existing: ExportedMessageRepository = {
      headId: previousTail.id,
      messages: [
        { message: first, parentId: null },
        { message: previousTail, parentId: first.id }
      ]
    }

    const incoming: RuntimeMessageRepository = {
      headId: nextTail.id,
      messages: existing.messages,
      [runtimeMessageRepositoryDelta]: { message: nextTail, parentId: first.id }
    }

    const repository = repositoryMock(existing)
    const visibleMessages = existing.messages.map(item => item.message)

    const messages = syncRepositoryIncrementally(
      repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0],
      incoming,
      visibleMessages
    )

    expect(repository.export).not.toHaveBeenCalled()
    expect(repository.getMessages).not.toHaveBeenCalled()
    expect(repository.getMessage).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(first.id, nextTail)
    expect(repository.deleteMessage).not.toHaveBeenCalled()
    expect(repository.resetHead).not.toHaveBeenCalled()
    expect(getThreadMessageListUpdate(messages)).toMatchObject({
      index: 1,
      message: nextTail,
      previousMessage: previousTail
    })
  })

  it('falls back to authoritative reconcile when a same-id delta is based on an uncommitted snapshot', () => {
    const root = message('assistant-1', 'root')
    const previousTail = message('assistant-2', 'tail')
    const nextRoot = message('assistant-1', 'updated root')
    const nextTail = message('assistant-2', 'updated tail')

    const existing: ExportedMessageRepository = {
      headId: previousTail.id,
      messages: [
        { message: root, parentId: null },
        { message: previousTail, parentId: root.id }
      ]
    }

    const repository = repositoryMock(existing)

    const incoming: RuntimeMessageRepository = {
      headId: nextTail.id,
      messages: [
        { message: nextRoot, parentId: null },
        { message: previousTail, parentId: nextRoot.id }
      ],
      [runtimeMessageRepositoryDelta]: { message: nextTail, parentId: nextRoot.id }
    }

    syncRepositoryIncrementally(
      repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0],
      incoming,
      existing.messages.map(item => item.message),
      false
    )

    expect(repository.export).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(null, nextRoot)
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(nextRoot.id, nextTail)
  })

  it('records consecutive visible-tail provenance without retaining predecessor arrays', () => {
    const first = message('assistant-1', 'first')
    const initialTail = message('assistant-2', 'A')
    const nextTail = message('assistant-2', 'AB')
    const finalTail = message('assistant-2', 'ABC')

    const initial: ExportedMessageRepository = {
      headId: initialTail.id,
      messages: [
        { message: first, parentId: null },
        { message: initialTail, parentId: first.id }
      ]
    }

    const firstRepository = repositoryMock(initial)

    const firstMessages = syncRepositoryIncrementally(
      firstRepository as unknown as Parameters<typeof syncRepositoryIncrementally>[0],
      {
        headId: nextTail.id,
        messages: initial.messages,
        [runtimeMessageRepositoryDelta]: { message: nextTail, parentId: first.id }
      }
    )

    const firstUpdate = getThreadMessageListUpdate(firstMessages)
    const initialVisibleMessages = firstRepository.getMessages.mock.results[0]?.value as ThreadMessage[]

    expect(firstUpdate).toMatchObject({
      previousToken: getThreadMessageListToken(initialVisibleMessages),
      previousMessage: initialTail,
      message: nextTail
    })
    expect(firstUpdate?.previousToken).not.toBe(initialVisibleMessages)

    const afterFirst: ExportedMessageRepository = {
      headId: nextTail.id,
      messages: [
        { message: first, parentId: null },
        { message: nextTail, parentId: first.id }
      ]
    }

    const secondRepository = repositoryMock(afterFirst)
    secondRepository.getMessages.mockReturnValue(firstMessages as ThreadMessage[])

    const secondMessages = syncRepositoryIncrementally(
      secondRepository as unknown as Parameters<typeof syncRepositoryIncrementally>[0],
      {
        headId: finalTail.id,
        messages: afterFirst.messages,
        [runtimeMessageRepositoryDelta]: { message: finalTail, parentId: first.id }
      }
    )

    expect(getThreadMessageListUpdate(secondMessages)).toMatchObject({
      index: 1,
      previousToken: getThreadMessageListToken(firstMessages),
      previousMessage: nextTail,
      message: finalTail
    })
  })

  it('removes messages missing from an authoritative snapshot and moves the head', () => {
    const first = message('assistant-1', 'first')
    const staleTail = message('assistant-2', 'stale')

    const existing: ExportedMessageRepository = {
      headId: staleTail.id,
      messages: [
        { message: first, parentId: null },
        { message: staleTail, parentId: first.id }
      ]
    }

    const incoming: ExportedMessageRepository = {
      headId: first.id,
      messages: [{ message: first, parentId: null }]
    }

    const repository = repositoryMock(existing)

    syncRepositoryIncrementally(repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0], incoming)

    expect(repository.deleteMessage).toHaveBeenCalledOnce()
    expect(repository.deleteMessage).toHaveBeenCalledWith(staleTail.id)
    expect(repository.addOrUpdateMessage).not.toHaveBeenCalled()
    expect(repository.resetHead).toHaveBeenCalledOnce()
    expect(repository.resetHead).toHaveBeenCalledWith(first.id)
  })

  it('rebuilds a fully disjoint transcript from leaves to root', () => {
    const oldRoot = message('old-1', 'old root')
    const oldTail = message('old-2', 'old tail')
    const next = message('new-1', 'new')

    const existing: ExportedMessageRepository = {
      headId: oldTail.id,
      messages: [
        { message: oldRoot, parentId: null },
        { message: oldTail, parentId: oldRoot.id }
      ]
    }

    const incoming: ExportedMessageRepository = {
      headId: next.id,
      messages: [{ message: next, parentId: null }]
    }

    const repository = repositoryMock(existing)

    syncRepositoryIncrementally(repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0], incoming)

    expect(repository.deleteMessage.mock.calls).toEqual([[oldTail.id], [oldRoot.id]])
    expect(repository.addOrUpdateMessage).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(null, next)
    expect(repository.resetHead).toHaveBeenCalledWith(next.id)
  })

  it('updates a stable message when its branch parent changes', () => {
    const root = message('assistant-1', 'root')
    const branch = message('assistant-2', 'branch')

    const existing: ExportedMessageRepository = {
      headId: branch.id,
      messages: [
        { message: root, parentId: null },
        { message: branch, parentId: root.id }
      ]
    }

    const incoming: ExportedMessageRepository = {
      headId: branch.id,
      messages: [
        { message: root, parentId: null },
        { message: branch, parentId: null }
      ]
    }

    const repository = repositoryMock(existing)

    syncRepositoryIncrementally(repository as unknown as Parameters<typeof syncRepositoryIncrementally>[0], incoming)

    expect(repository.addOrUpdateMessage).toHaveBeenCalledOnce()
    expect(repository.addOrUpdateMessage).toHaveBeenCalledWith(null, branch)
    expect(repository.resetHead).not.toHaveBeenCalled()
  })
})
