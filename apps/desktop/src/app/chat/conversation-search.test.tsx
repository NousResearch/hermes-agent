import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { StrictMode, useRef } from 'react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import { useTimelineReveal } from '@/components/assistant-ui/thread/use-timeline-reveal'
import { en } from '@/i18n/en'
import type { ChatMessage } from '@/lib/chat-messages'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessions } from '@/store/session'

import { ChatHeader } from './chat-header'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'

import { ChatRuntimeBoundary } from '.'

stubThreadEnvironment()
const copy = en.conversationSearch
const live: ChatMessage[] = [{ id: 'live', rowId: 10000, role: 'user', parts: [{ type: 'text', text: 'recent only' }] }]

beforeEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  $sessions.set([])
  Element.prototype.scrollIntoView = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
})

function VisibleWindow() {
  const history = useTranscriptWindow()
  const viewport = useRef<HTMLDivElement>(null)
  const messages = history.currentMessages ?? []
  useTimelineReveal({
    viewport,
    groups: messages.map(message => ({ id: message.id, weight: 1 })),
    hiddenCount: 0,
    renderBudget: 1000,
    olderAvailable: false,
    revealBudget: () => {},
    expandWindow: () => {},
    prepare: () => {}
  })

  return (
    <div data-slot="aui_thread-viewport" ref={viewport}>
      {messages.map(message => (
        <div data-message-id={message.id} key={message.id}>
          {message.id}
        </div>
      ))}
    </div>
  )
}

function mount() {
  const view = {
    ...PRIMARY_SESSION_VIEW,
    $messages: atom(live),
    $runtimeId: atom<string | null>('runtime'),
    $storedId: atom<string | null>('search-stored')
  }

  const ui = render(
    <StrictMode>
      <MemoryRouter>
        <SessionViewProvider value={view}>
          <div data-chat-surface="">
            <ChatRuntimeBoundary
              busy={false}
              onCancel={() => {}}
              onEdit={async () => {}}
              onReload={async () => {}}
              onThreadMessagesChange={() => {}}
              suppressMessages={false}
            >
              <ChatHeader
                activeSessionId="runtime"
                isRoutedSessionView={false}
                onDeleteSelectedSession={() => {}}
                onToggleSelectedPin={() => {}}
                selectedSessionId="search-stored"
              />
              <VisibleWindow />
            </ChatRuntimeBoundary>
          </div>
        </SessionViewProvider>
      </MemoryRouter>
    </StrictMode>
  )

  fireEvent.click(screen.getByRole('button', { name: copy.open }))

  return { view, ...ui }
}

function page(offset = 0, empty = false) {
  return {
    session_id: 'search-stored',
    profile: 'default',
    results: empty
      ? []
      : offset === 0
        ? [
            { row_id: 40, role: 'user', snippet: 'needle archived', timestamp: 40 },
            { row_id: 41, role: 'assistant', snippet: 'needle old answer', timestamp: 41 }
          ]
        : [{ row_id: 90, role: 'assistant', snippet: 'needle next page', timestamp: 90 }],
    pagination: { limit: 50, offset, has_more: offset === 0 && !empty, next_offset: offset === 0 && !empty ? 50 : null }
  }
}

function windowPage(rowId: number, role: string) {
  return {
    session_id: 'search-stored',
    profile: 'default',
    messages: [
      {
        id: rowId,
        role,
        content: 'needle match',
        timestamp: rowId,
        tool_name: 'terminal',
        tool_call_id: `call-${rowId}`
      }
    ],
    pagination: {
      row_id: rowId,
      limit: 120,
      offset: rowId,
      returned: 1,
      order: 'oldest',
      has_older: true,
      has_newer: true
    }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

const requests = (api: ReturnType<typeof vi.fn>, suffix: string) =>
  api.mock.calls.map(([request]) => new URL(request.path, 'http://test')).filter(url => url.pathname.endsWith(suffix))

describe('stored conversation search', () => {
  it.each(['user', 'assistant', 'tool'])(
    'reveals unloaded %s matches and navigates bounded result pages without changing live history',
    async role => {
      const pending = deferred<ReturnType<typeof page>>()

      const api = vi.spyOn(window.hermesDesktop, 'api').mockImplementation(async request => {
        const url = new URL(request.path, 'http://test')

        if (url.pathname.endsWith('/messages/match')) {
          return windowPage(Number(url.searchParams.get('row_id')), role)
        }

        const offset = Number(url.searchParams.get('offset'))

        if (requests(api, '/messages/search').length === 1) {
          return pending.promise
        }

        return page(offset)
      })

      const mounted = mount()
      const original = mounted.view.$messages.get()
      fireEvent.change(screen.getByRole('textbox', { name: copy.open }), { target: { value: 'needle' } })
      await waitFor(() => expect(requests(api, '/messages/search')).toHaveLength(1), { timeout: 10000 })
      expect(screen.getByText(copy.searching)).toBeTruthy()
      expect(screen.queryByText(copy.empty)).toBeNull()
      await act(async () => pending.resolve(page()))
      fireEvent.click(await screen.findByText('needle archived'))
      await waitFor(() => expect(mounted.container.querySelector('[data-conversation-match]')).toBeTruthy(), {
        timeout: 10000
      })
      expect(requests(api, '/messages/match').at(-1)?.searchParams.get('row_id')).toBe('40')
      fireEvent.click(screen.getByRole('button', { name: en.findInPage.next }))
      await waitFor(() => expect(requests(api, '/messages/match').at(-1)?.searchParams.get('row_id')).toBe('41'))
      fireEvent.click(screen.getByRole('button', { name: en.findInPage.next }))
      await waitFor(() => expect(requests(api, '/messages/match').at(-1)?.searchParams.get('row_id')).toBe('90'))
      expect(requests(api, '/messages/search').at(-1)?.searchParams.get('offset')).toBe('50')
      fireEvent.click(screen.getByRole('button', { name: en.findInPage.previous }))
      await waitFor(() => expect(requests(api, '/messages/match').at(-1)?.searchParams.get('row_id')).toBe('41'))

      for (const url of requests(api, '/messages/search')) {
        expect(url.pathname).toBe('/api/sessions/search-stored/messages/search')
        expect(url.searchParams.get('profile')).toBe('default')
        expect(url.searchParams.get('q')).toBe('needle')
        expect(url.searchParams.get('limit')).toBe('50')
      }

      for (const url of requests(api, '/messages/match')) {
        expect(url.searchParams.get('limit')).toBe('120')
      }

      expect(mounted.view.$messages.get()).toBe(original)
      expect(mounted.container.querySelectorAll('[data-conversation-match]')).toHaveLength(1)
      fireEvent.click(screen.getByRole('button', { name: copy.close }))
      await waitFor(() => expect(mounted.container.querySelector('[data-conversation-match]')).toBeNull())
    }
  )

  it.each(['query', 'profile', 'close', 'failure', 'unsupported'])(
    'keeps pending %s responses from becoming false negatives or stale results',
    async scenario => {
      const pending = deferred<ReturnType<typeof page>>()

      const api = vi.spyOn(window.hermesDesktop, 'api').mockImplementation(async () => {
        if (requests(api, '/messages/search').length === 1) {
          return pending.promise
        }

        return page(0, true)
      })

      const mounted = mount()
      fireEvent.change(screen.getByRole('textbox', { name: copy.open }), { target: { value: 'needle' } })
      await waitFor(() => expect(requests(api, '/messages/search')).toHaveLength(1), { timeout: 10000 })
      expect(screen.queryByText(copy.empty)).toBeNull()

      const transitions: Record<string, () => Promise<void>> = {
        query: async () => {
          fireEvent.change(screen.getByRole('textbox', { name: copy.open }), { target: { value: 'replacement' } })
          await screen.findByText(copy.empty, {}, { timeout: 10000 })
          await act(async () => pending.resolve(page()))
          expect(screen.getByRole<HTMLInputElement>('textbox', { name: copy.open }).value).toBe('replacement')
        },
        profile: async () => {
          await act(async () => {
            $activeGatewayProfile.set('other')
            mounted.view.$storedId.set('other-stored')
          })
          await act(async () => pending.resolve(page()))
          expect(screen.queryByRole('textbox', { name: copy.open })).toBeNull()
          fireEvent.click(screen.getByRole('button', { name: copy.open }))
          fireEvent.change(screen.getByRole('textbox', { name: copy.open }), { target: { value: 'new owner' } })
          await screen.findByText(copy.empty, {}, { timeout: 10000 })
          expect(requests(api, '/messages/search').at(-1)?.searchParams.get('profile')).toBe('other')
        },
        close: async () => {
          fireEvent.click(screen.getByRole('button', { name: copy.close }))
          await act(async () => pending.resolve(page()))
          expect(screen.queryByRole('textbox', { name: copy.open })).toBeNull()
        },
        failure: async () => {
          await act(async () => pending.reject(new Error('connection lost')))
          expect(await screen.findByText(copy.searchFailed)).toBeTruthy()
          expect(screen.queryByText(copy.empty)).toBeNull()
          fireEvent.click(screen.getByRole('button', { name: copy.retry }))
          expect(await screen.findByText(copy.empty)).toBeTruthy()
        },
        unsupported: async () => {
          await act(async () => pending.reject(new Error('No such API endpoint')))
          expect(await screen.findByText(copy.unavailable)).toBeTruthy()
          expect(screen.queryByText(copy.empty)).toBeNull()
        }
      }

      await transitions[scenario]()
      expect(screen.queryByText('needle archived')).toBeNull()
      expect(requests(api, '/messages/match')).toHaveLength(0)
      expect(mounted.container.querySelector('[data-conversation-match]')).toBeNull()
      expect(mounted.view.$messages.get()).toBe(live)
    }
  )
})
