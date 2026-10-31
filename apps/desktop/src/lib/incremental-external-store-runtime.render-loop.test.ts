import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import {
  AssistantRuntimeProvider,
  type ExportedMessageRepository,
  type ExternalStoreAdapter,
  type ThreadMessage,
  useAuiState
} from '@assistant-ui/react'
import { render, screen, waitFor } from '@testing-library/react'
import { createElement, StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from './incremental-external-store-runtime'

/**
 * The shipped desktop app died with
 * `"workspace" failed to render — Maximum update depth exceeded. The result of
 * getSnapshot should be cached to avoid an infinite loop.` as soon as the chat
 * mounted.
 *
 * Cause: `@assistant-ui/core` 0.2.23's ThreadListRuntimeImpl reads a
 * LazyMemoizeSubject that is never connected (it subscribes straight to the
 * core), so `threads.getState()` allocates a new wrapper object per read. Tap's
 * external-store commit check compares with `Object.is`, so every commit looks
 * like a change, force-updates, and throws after 50 commits.
 *
 * The lockfile pins tap 0.9.8, which has no such check — the packaged app was
 * built against 0.9.17-era semantics. That is why this crash never showed up in
 * CI: the installed dependency cannot fail the way the shipped one does. Run
 * this suite with tap 0.9.17 (or newer) to keep the guard honest.
 */
const STATUS = getAutoStatus(false, false, false, false, undefined)

function message(id: string, text: string): ThreadMessage {
  return fromThreadMessageLike({ role: 'assistant', content: [{ type: 'text', text }] }, id, STATUS)
}

function exported(text: string): ExportedMessageRepository {
  return { headId: 'reply', messages: [{ message: message('reply', text), parentId: null }] }
}

function Consumer() {
  const messages = useAuiState(state => state.thread.messages)

  return createElement(
    'output',
    null,
    messages
      .map(item => item.content.map(part => (part.type === 'text' ? part.text : '')).join(''))
      .join('|')
  )
}

function Harness({ text, threadId }: { text: string; threadId: string }) {
  const store: ExternalStoreAdapter = {
    messageRepository: exported(text),
    onNew: async () => {},
    adapters: { threadList: { threadId } }
  }

  const runtime = useIncrementalExternalStoreRuntime(store)

  return createElement(AssistantRuntimeProvider, { runtime }, createElement(Consumer))
}

const view = (text: string, threadId: string) =>
  createElement(StrictMode, null, createElement(Harness, { text, threadId }))

describe('chat runtime mount', () => {
  let errors: string[]

  beforeEach(() => {
    errors = []
    vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      errors.push(args.map(String).join(' '))
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders the real provider through mount, streaming updates and a session switch', async () => {
    const { rerender, unmount } = render(view('hello', 'first'))

    try {
      expect(screen.getByRole('status').textContent).toBe('hello')

      for (const text of ['hello w', 'hello world']) {
        rerender(view(text, 'first'))
        await waitFor(() => expect(screen.getByRole('status').textContent).toBe(text))
      }

      rerender(view('another session', 'second'))
      await waitFor(() => expect(screen.getByRole('status').textContent).toBe('another session'))

      // The render loop, not the one-shot "getSnapshot should be cached" notice
      // that tap's own first pre-subscription read emits upstream.
      expect(errors.filter(line => /Maximum update depth/i.test(line))).toEqual([])
    } finally {
      unmount()
    }
  })
})
