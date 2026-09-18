import { useAssistantRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { type TranscriptWindowValue, useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import type { ChatMessage } from '@/lib/chat-messages'
import { $transcriptTailBySessionId } from '@/store/transcript-tail'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'

import { ChatRuntimeBoundary } from '.'

stubThreadEnvironment()

const message = (rowId: number): ChatMessage => ({
  id: `live-${rowId}`,
  rowId,
  role: 'user',
  parts: [{ type: 'text', text: `prompt ${rowId}` }]
})

const page = (rowId: number) => ({
  session_id: 'stored',
  pagination: { has_older: true, has_newer: true, limit: 120, offset: 40, returned: 120, order: 'oldest' },
  messages: Array.from({ length: 120 }, (_, index) => ({
    id: rowId + index,
    role: 'user' as const,
    content: `prompt ${rowId + index}`,
    timestamp: rowId + index
  }))
})

beforeEach(() => {
  $transcriptTailBySessionId.set({})
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
})

function mount() {
  const $messages = atom(Array.from({ length: 120 }, (_, index) => message(10_000 + index)))

  const view = {
    ...PRIMARY_SESSION_VIEW,
    $messages,
    $runtimeId: atom<string | null>('runtime'),
    $storedId: atom<string | null>('stored')
  }

  let window!: Required<TranscriptWindowValue>
  let runtime!: NonNullable<ReturnType<typeof useAssistantRuntime>>

  function Observe() {
    window = useTranscriptWindow()
    runtime = useAssistantRuntime()!

    return null
  }

  const mutations = { onEdit: vi.fn(), onReload: vi.fn(), onCancel: vi.fn(), onThreadMessagesChange: vi.fn() }

  const rendered = render(
    <SessionViewProvider value={view}>
      <ChatRuntimeBoundary busy={false} suppressMessages={false} {...mutations}>
        <Observe />
      </ChatRuntimeBoundary>
    </SessionViewProvider>
  )

  return {
    view,
    mutations,
    ...rendered,
    get window() {
      return window
    },
    get runtime() {
      return runtime
    }
  }
}

describe('bounded direct history runtime', () => {
  it('reads one around page and selects it without replacing the live store', async () => {
    const api = vi.spyOn(window.hermesDesktop, 'api').mockResolvedValue(page(40))
    const mounted = mount()
    const live = mounted.view.$messages.get()
    let id: string | null = null
    await act(async () => {
      id = await mounted.window.revealRow(40, new AbortController().signal)
    })

    expect(api).toHaveBeenCalledTimes(1)
    const url = new URL(api.mock.calls[0][0].path, 'http://test')
    expect(url.pathname).toBe('/api/sessions/stored/messages/around')
    expect(url.searchParams.get('row_id')).toBe('40')
    expect(url.searchParams.get('limit')).toBe('120')
    expect(mounted.view.$messages.get()).toBe(live)
    expect(mounted.runtime.thread.getState().messages).toHaveLength(120)
    expect(mounted.runtime.thread.getState().messages.some(message => message.id === id)).toBe(true)
    expect(mounted.window.currentMessages?.find(message => message.rowId === 40)?.id).toBe(id)
    expect(mounted.window.isHistorical).toBe(true)
    expect(mounted.window.newerAvailable).toBe(true)
    act(() => {
      mounted.window.returnToLatest()
    })
    expect(mounted.window.isHistorical).toBe(false)
  })

  it('keeps history static during streaming and restores the newest live tail and capabilities', async () => {
    vi.spyOn(window.hermesDesktop, 'api').mockResolvedValue(page(40))
    const mounted = mount()
    await act(async () => {
      await mounted.window.revealRow(40, new AbortController().signal)
    })
    const historical = mounted.runtime.thread.getState().messages
    const snapshot = mounted.window
    expect(mounted.runtime.thread.getState().capabilities.edit).toBe(false)
    expect(mounted.runtime.thread.getState().capabilities.reload).toBe(false)
    expect(mounted.runtime.thread.getState().capabilities.switchToBranch).toBe(false)
    expect(mounted.runtime.thread.getState().isDisabled).toBe(true)
    act(() => {
      mounted.view.$messages.set([...mounted.view.$messages.get(), message(20_000)])
    })
    expect(mounted.runtime.thread.getState().messages).toBe(historical)
    expect(mounted.window).toBe(snapshot)
    expect(await mounted.window.expandWindow()).toBe(false)
    act(() => {
      mounted.window.returnToLatest()
    })
    expect(mounted.runtime.thread.getState().messages.at(-1)?.id).toBe('live-20000')
    expect(mounted.runtime.thread.getState().capabilities.edit).toBe(true)
    expect(mounted.runtime.thread.getState().capabilities.reload).toBe(true)
    expect(mounted.runtime.thread.getState().isDisabled).toBe(false)
    expect(mounted.window.isHistorical).toBe(false)

    for (const callback of Object.values(mounted.mutations)) {
      expect(callback).not.toHaveBeenCalled()
    }
  })

  it('latest request wins even when the bridge ignores cancellation', async () => {
    const resolves: ((value: ReturnType<typeof page>) => void)[] = []
    vi.spyOn(window.hermesDesktop, 'api').mockImplementation(() => new Promise(resolve => resolves.push(resolve)))
    const mounted = mount()
    let first!: Promise<string | null>
    let second!: Promise<string | null>
    act(() => {
      first = mounted.window.revealRow(40, new AbortController().signal)
      second = mounted.window.revealRow(400, new AbortController().signal)
    })
    expect(await first).toBeNull()
    await act(async () => {
      resolves[1](page(400))
      await second
    })
    const selected = mounted.window.currentMessages
    await act(async () => {
      resolves[0](page(40))
      await Promise.resolve()
    })
    expect(mounted.window.currentMessages).toBe(selected)
    expect(selected[0].rowId).toBe(400)
  })

  it.each(['abort', 'latest', 'session', 'unmount'] as const)('discards pending reads on %s', async action => {
    let resolve!: (value: ReturnType<typeof page>) => void
    vi.spyOn(window.hermesDesktop, 'api').mockImplementation(
      () =>
        new Promise(done => {
          resolve = done
        })
    )
    const mounted = mount()
    const signal = new AbortController()
    let pending!: Promise<string | null>
    act(() => {
      pending = mounted.window.revealRow(40, signal.signal)
    })
    act(() => {
      if (action === 'abort') {
        signal.abort()
      }

      if (action === 'latest') {
        mounted.window.returnToLatest()
      }

      if (action === 'session') {
        mounted.view.$storedId.set('next-session')
        mounted.view.$runtimeId.set('next-runtime')
        mounted.view.$messages.set([message(30_000)])
      }

      if (action === 'unmount') {
        mounted.unmount()
      }
    })
    expect(await pending).toBeNull()
    await act(async () => {
      resolve(page(40))
      await Promise.resolve()
    })
    expect(mounted.view.$messages.get().some(message => message.rowId === 40)).toBe(false)
    expect(mounted.window.isHistorical).toBe(false)
  })

  it('rejects oversized, missing-target and failed responses without losing the selected page', async () => {
    const api = vi.spyOn(window.hermesDesktop, 'api').mockResolvedValue(page(40))
    const mounted = mount()
    await act(async () => {
      await mounted.window.revealRow(40, new AbortController().signal)
    })
    const selected = mounted.window.currentMessages

    for (const response of [
      { ...page(400), messages: [...page(400).messages, ...page(600).messages] },
      page(800),
      null
    ]) {
      if (response) {
        api.mockResolvedValueOnce(response)
      } else {
        api.mockRejectedValueOnce(new Error('offline'))
      }

      await act(async () => {
        expect(await mounted.window.revealRow(400, new AbortController().signal)).toBeNull()
      })
      expect(mounted.window.currentMessages).toBe(selected)
    }
  })
})
