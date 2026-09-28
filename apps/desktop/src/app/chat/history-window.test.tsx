import { useAssistantRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { type TranscriptWindowValue, useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import type { ChatMessage } from '@/lib/chat-messages'
import { $transcriptTailBySessionId } from '@/store/transcript-tail'
import type { SessionMessage } from '@/types/hermes'

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
  pagination: {
    has_older: true,
    has_newer: true,
    limit: 120,
    offset: 40,
    returned: 120,
    order: 'oldest',
    first_cursor: rowId,
    last_cursor: rowId + 119
  },
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

function mount(storedId = 'stored') {
  const $messages = atom(Array.from({ length: 120 }, (_, index) => message(10_000 + index)))

  const view = {
    ...PRIMARY_SESSION_VIEW,
    $messages,
    $runtimeId: atom<string | null>('runtime'),
    $storedId: atom<string | null>(storedId)
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
    expect(mounted.window.expectedRuntimeIds).toBe(
      mounted.runtime.thread.getState().messages.map(message => message.id).join('\n')
    )
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
    // `edit` is the one capability that survives on a bounded page: the rail
    // jump is its only entry and it has no in-thread exit, so dropping it
    // wedged the inline composer after every far jump (#117298).
    expect(mounted.runtime.thread.getState().capabilities.edit).toBe(true)
    expect(mounted.runtime.thread.getState().capabilities.reload).toBe(false)
    expect(mounted.runtime.thread.getState().capabilities.switchToBranch).toBe(false)
    expect(mounted.runtime.thread.getState().isDisabled).toBe(true)
    act(() => {
      mounted.view.$messages.set([...mounted.view.$messages.get(), message(20_000)])
    })
    expect(mounted.runtime.thread.getState().messages).toBe(historical)
    expect(mounted.window).toBe(snapshot)
    await act(async () => {
      expect(await mounted.window.expandWindow()).toBe(false)
    })
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

  it('keeps the edit composer available after a rail jump selects a history page', async () => {
    vi.spyOn(window.hermesDesktop, 'api').mockResolvedValue(page(40))
    const mounted = mount()
    await act(async () => {
      await mounted.window.revealRow(40, new AbortController().signal)
    })

    expect(mounted.window.isHistorical).toBe(true)
    expect(mounted.runtime.thread.getState().capabilities.edit).toBe(true)

    // The exact gesture that was dead: clicking a message on the bounded page
    // the rail selected. With `onEdit` dropped on a history page this threw
    // "Runtime does not support editing" (ExternalStoreThreadRuntimeCore
    // .beginEdit) and left the composer unopenable for every message, healed
    // only by the floating jump button's returnToLatest.
    const composer = mounted.runtime.thread.getMessageByIndex(0).composer

    expect(() =>
      act(() => {
        composer.beginEdit()
      })
    ).not.toThrow()
    expect(composer.getState().isEditing).toBe(true)

    act(() => {
      composer.cancel()
    })
    act(() => {
      mounted.window.returnToLatest()
    })
    expect(composer.getState().isEditing).toBe(false)
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

describe('adjacent history windows', () => {
  it('walks a tool turn longer than 120 rows in both directions, retains a bounded window and rejoins split calls', async () => {
    const rows: SessionMessage[] = [{ id: 1, role: 'user', content: 'one long turn', timestamp: 1 }]

    for (let i = 0; i < 300; i += 1) {
      rows.push(
        {
          id: 2 + i * 2,
          role: 'assistant',
          content: '',
          timestamp: 1,
          tool_calls: [{ id: `call-${i}`, type: 'function', function: { name: 'terminal', arguments: '{}' } }]
        },
        {
          id: 3 + i * 2,
          role: 'tool',
          tool_call_id: `call-${i}`,
          tool_name: 'terminal',
          content: `result-${i}`,
          timestamp: 1
        }
      )
    }

    rows.push({ id: 602, role: 'user', content: 'next turn', timestamp: 1 })

    const api = vi.spyOn(window.hermesDesktop, 'api').mockImplementation(async request => {
      const params = new URL(request.path, 'http://test').searchParams

      const start = params.has('before_cursor')
        ? Math.max(0, Number(params.get('before_cursor')) - 121)
        : params.has('after_cursor')
          ? Number(params.get('after_cursor'))
          : Number(params.get('row_id')) - 1

      const selected = rows.slice(start, start + 120)

      return {
        session_id: 'stored',
        messages: selected,
        pagination: {
          limit: 120,
          returned: selected.length,
          offset: start,
          order: 'oldest',
          has_older: start > 0,
          has_newer: start + selected.length < rows.length,
          first_cursor: selected[0]?.id,
          last_cursor: selected.at(-1)?.id,
          leading_prompt_row_id: start > 0 ? 1 : null
        }
      }
    })

    const mounted = mount()
    const live = mounted.view.$messages.get()
    await act(async () => {
      await mounted.window.revealRow(1, new AbortController().signal)
    })
    const promptId = mounted.window.currentMessages[0].id
    const results = new Set<string>()

    for (let i = 0; i < 5; i += 1) {
      await act(async () => {
        expect(await mounted.window.revealNewer()).toBe(true)
      })

      const calls = mounted.window.currentMessages
        .flatMap(message => message.parts)
        .filter(part => part.type === 'tool-call')

      expect(mounted.window.expectedRuntimeIds).toBe(
        mounted.runtime.thread.getState().messages.map(message => message.id).join('\n')
      )

      // Raw retention is three pages even though hundreds of rows hydrate into
      // one bubble; no cumulative full-transcript hydration.
      expect(calls.length).toBeLessThanOrEqual(181)

      for (const part of calls) {
        if (part.result !== undefined && part.toolCallId) {
          results.add(part.toolCallId)
        }
      }

      if (i === 0) {
        const seam = calls.find(part => part.toolCallId === 'call-59')!
        expect(seam.result).toBeDefined()
        expect(calls.filter(part => part.toolCallId === 'call-59')).toHaveLength(1)
        expect(mounted.window.currentMessages[0].id).toBe(promptId)
      }
    }

    expect(results.size).toBe(300)
    expect(mounted.window.newerAvailable).toBe(false)
    expect(mounted.window.leadingRowId).toBe(1)
    expect(mounted.window.currentMessages.at(-1)?.rowId).toBe(602)

    for (let i = 0; i < 3; i += 1) {
      await act(async () => {
        expect(await mounted.window.expandWindow()).toBe(true)
      })
    }

    expect(mounted.window.currentMessages[0].id).toBe(promptId)
    expect(mounted.window.olderAvailable).toBe(false)
    expect(mounted.window.newerAvailable).toBe(true)
    expect(mounted.view.$messages.get()).toBe(live)
    expect(api.mock.calls.every(([request]) => request.path.includes('/messages/around?'))).toBe(true)
    act(() => mounted.window.returnToLatest())
    expect(mounted.runtime.thread.getState().messages.at(-1)?.id).toBe('live-10119')
  })

  it.each(['older', 'newer'] as const)(
    'captures %s changes only on commit and ignores a response superseded by a rail jump',
    async direction => {
      const api = vi.spyOn(window.hermesDesktop, 'api').mockResolvedValueOnce(page(4000))
      const mounted = mount()
      await act(async () => {
        await mounted.window.revealRow(4000, new AbortController().signal)
      })
      let finish!: (value: ReturnType<typeof page>) => void
      api.mockImplementationOnce(
        () =>
          new Promise(resolve => {
            finish = resolve
          })
      )
      const before = vi.fn()
      let pending!: Promise<boolean | void>
      act(() => {
        pending = Promise.resolve(
          direction === 'older' ? mounted.window.expandWindow(before) : mounted.window.revealNewer(before)
        )
      })
      expect(before).not.toHaveBeenCalled()
      api.mockResolvedValueOnce(page(8000))
      await act(async () => {
        await mounted.window.revealRow(8000, new AbortController().signal)
      })
      expect(await pending).toBe(false)
      await act(async () => {
        finish(page(direction === 'older' ? 3880 : 4120))
        await Promise.resolve()
      })
      expect(before).not.toHaveBeenCalled()
      expect(mounted.window.currentMessages[0].rowId).toBe(8000)
    }
  )

  it('preserves the selected content on empty, invalid or failed continuations and reports older backend capability', async () => {
    const api = vi.spyOn(window.hermesDesktop, 'api').mockResolvedValue(page(40))
    const mounted = mount()
    await act(async () => {
      await mounted.window.revealRow(40, new AbortController().signal)
    })
    const selected = mounted.window.currentMessages
    const before = vi.fn()
    await act(async () => {
      expect(await mounted.window.revealNewer(before)).toBe(false)
    })
    expect(mounted.window.historyError).toBe('failed')
    expect(mounted.window.currentMessages).toBe(selected)
    api.mockResolvedValueOnce({
      ...page(160),
      messages: [],
      pagination: { ...page(160).pagination, returned: 0, has_newer: false }
    })
    await act(async () => {
      expect(await mounted.window.revealNewer(before)).toBe(false)
    })
    expect(mounted.window.currentMessages).toBe(selected)
    expect(mounted.window.newerAvailable).toBe(false)
    expect(before).not.toHaveBeenCalled()
    api.mockResolvedValueOnce({
      ...page(400),
      pagination: { ...page(400).pagination, first_cursor: undefined, last_cursor: undefined }
    })
    await act(async () => {
      await mounted.window.revealRow(400, new AbortController().signal)
    })
    await act(async () => {
      expect(await mounted.window.revealNewer()).toBe(false)
    })
    expect(mounted.window.historyError).toBe('unavailable')
    expect(mounted.window.currentMessages[0].rowId).toBe(400)
  })
})
