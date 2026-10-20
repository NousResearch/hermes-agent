import type { ExportedMessageRepository, ExternalStoreAdapter } from '@assistant-ui/react'
import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useIncrementalExternalStoreRuntime } from './incremental-external-store-runtime'

const EMPTY_REPOSITORY: ExportedMessageRepository = {
  headId: null,
  messages: []
}

function adapter(threadId = 'DEFAULT_THREAD_ID'): ExternalStoreAdapter {
  return {
    messageRepository: EMPTY_REPOSITORY,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onCancel: async () => {},
    adapters: {
      threadList: { threadId }
    }
  }
}

describe('incremental runtime thread-list snapshots', () => {
  it('caches the snapshot until the thread-list runtime publishes a change', () => {
    const { result, rerender } = renderHook(
      ({ threadId }: { threadId: string }) => useIncrementalExternalStoreRuntime(adapter(threadId)),
      { initialProps: { threadId: 'thread-one' } }
    )

    const first = result.current.threads.getState()
    const second = result.current.threads.getState()

    expect(second).toBe(first)

    rerender({ threadId: 'thread-two' })

    const changed = result.current.threads.getState()

    expect(changed).not.toBe(first)
    expect(changed.mainThreadId).toBe('thread-two')
    expect(result.current.threads.getState()).toBe(changed)
  })

  it('reflects a publication that fires before the first read', () => {
    const { result, rerender } = renderHook(
      ({ threadId }: { threadId: string }) => useIncrementalExternalStoreRuntime(adapter(threadId)),
      { initialProps: { threadId: 'thread-one' } }
    )

    rerender({ threadId: 'thread-two' })

    const first = result.current.threads.getState()

    expect(first.mainThreadId).toBe('thread-two')
    expect(result.current.threads.getState()).toBe(first)
  })
})
