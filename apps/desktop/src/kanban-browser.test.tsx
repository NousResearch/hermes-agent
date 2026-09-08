import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getApiRequestConnection,
  getApiRequestProfile,
  setApiRequestConnection,
  setApiRequestProfile
} from '@/api/client'
import { createPluginContext } from '@/contrib/plugin'
import { createClientSessionState } from '@/lib/chat-runtime'
import { host } from '@/sdk'
import { $composerDraft } from '@/store/composer'
import { setActiveSessionId } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { BoardBrowser } from './plugins/kanban/browser'
import { browserKeys, createBoardBrowserReads } from './plugins/kanban/browser-api'
import { KANBAN_LOCALES } from './plugins/kanban/i18n'
import type { KanbanTaskDetail } from './plugins/kanban/types'

const source = (id: string) => ({
  id,
  label: `Synthetic ${id}`,
  kind: 'ssh' as const,
  remoteProfile: 'worker',
  tokenSet: false,
  tokenPreview: null
})

const detail = (label: string): KanbanTaskDetail => ({
  task: {
    id: 'same-task',
    title: label,
    status: 'blocked',
    assignee: 'worker',
    body: 'Synthetic task description',
    diagnostics: [
      {
        kind: 'blocked',
        severity: 'warning',
        title: 'Synthetic blocker',
        detail: 'Waiting for fixture approval',
        actions: [{ kind: 'reclaim', label: 'Reclaim' }],
        count: 1,
        last_seen_at: 1,
        data: {}
      }
    ]
  },
  runs: [{ id: 1, status: 'blocked', summary: 'Synthetic run summary', profile: 'worker' }],
  links: { parents: ['dependency'], children: [] },
  comments: [],
  events: [],
  attachments: []
})

let qc: QueryClient
const disposers: Array<() => void> = []

beforeEach(() => {
  qc = new QueryClient()
  createPluginContext('kanban', dispose => disposers.push(dispose)).i18n.register(KANBAN_LOCALES)
})
afterEach(() => {
  cleanup()
  qc.clear()
  disposers.splice(0).forEach(dispose => dispose())
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  setActiveSessionId(null)
  clearAllSessionStates()
  $composerDraft.set('')
})

describe('read-only host board browser (synthetic transport)', () => {
  it('isolates colliding board/task identities, ignores late replies and preserves foreground chat', async () => {
    let finishA: (value: KanbanTaskDetail) => void = () => undefined
    let rows = [source('a'), source('b')]

    const api = vi.fn(async (request: { path: string; connectionId: string; profile: string; method: string }) => {
      expect(request.method).toBe('GET')
      expect(request.profile).toBe('worker')

      if (request.path === '/api/plugins/kanban/boards') {
        return { boards: [{ slug: 'shared', name: 'Shared' }], current: 'shared' }
      }

      if (request.path === '/api/plugins/kanban/board?board=shared') {
        return { columns: [{ name: 'blocked', tasks: [detail(`Card ${request.connectionId}`).task] }], now: 1 }
      }

      if (request.path === '/api/plugins/kanban/tasks/same-task?board=shared') {
        if (request.connectionId === 'a') {
          return new Promise<KanbanTaskDetail>(resolve => {
            finishA = resolve
          })
        }

        return detail('Detail b')
      }

      throw new Error(`Unexpected request: ${request.path}`)
    })

    vi.stubGlobal('hermesDesktop', { api, connections: { list: async () => ({ connections: rows, primary: 'a' }) } })
    setApiRequestConnection('chat-host')
    setApiRequestProfile('chat-profile')
    setActiveSessionId('bot-chat-runtime')
    $composerDraft.set('Synthetic unsent Bot Chat draft')
    publishSessionState('bot-chat-runtime', {
      ...createClientSessionState('bot-chat-stored'),
      busy: true,
      awaitingResponse: true
    })
    const reads = createBoardBrowserReads(createPluginContext('kanban').rest)
    const view = render(
      <QueryClientProvider client={qc}>
        <BoardBrowser reads={reads} />
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Synthetic a / worker / Shared' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Card a' }))
    await waitFor(() => expect(api.mock.calls.some(([r]) => r.path.includes('/tasks/'))).toBe(true))
    fireEvent.click(screen.getByRole('button', { name: 'Synthetic b / worker / Shared' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Card b' }))
    expect(await screen.findByRole('heading', { name: 'Detail b' })).toBeTruthy()
    await act(async () => finishA(detail('Late detail a')))
    expect(screen.queryByText('Late detail a')).toBeNull()
    expect(screen.getByText('Synthetic run summary')).toBeTruthy()
    expect(screen.getByText('Waiting for fixture approval')).toBeTruthy()
    expect(
      screen.queryByRole('button', { name: /Reclaim|New task|Edit description|Send|Delete|Orchestration/i })
    ).toBeNull()
    expect(view.container.querySelector('[draggable="true"],textarea,input[type="file"]')).toBeNull()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'Card b' }))
    fireEvent.dragStart(screen.getByRole('button', { name: 'Card b' }))
    expect(screen.queryByRole('menu')).toBeNull()
    expect(getApiRequestConnection()).toBe('chat-host')
    expect(getApiRequestProfile()).toBe('chat-profile')
    expect(host.state.activeSessionId.get()).toBe('bot-chat-runtime')
    expect(host.state.busy.get()).toBe(true)
    expect(host.state.awaitingResponse.get()).toBe(true)
    expect($composerDraft.get()).toBe('Synthetic unsent Bot Chat draft')
    expect(
      qc.getQueryData(browserKeys.board({ connectionId: 'a', profile: 'worker', label: '' }, 'shared'))
    ).toBeTruthy()
    expect(
      qc.getQueryData(browserKeys.board({ connectionId: 'b', profile: 'worker', label: '' }, 'shared'))
    ).toBeTruthy()

    rows = [source('a')]
    await act(async () => {
      await qc.invalidateQueries({ queryKey: browserKeys.inventory })
    })
    expect(await screen.findByText('The selected source or board is no longer registered.')).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Detail b' })).toBeNull()
    const calls = api.mock.calls.length
    view.unmount()
    await act(async () => {
      await qc.invalidateQueries()
    })
    expect(api).toHaveBeenCalledTimes(calls)
  })

  it('keeps cached data explicitly stale on denial, retries manually, and never calls hidden sources', async () => {
    let denied = false

    const api = vi.fn(async (request: { path: string; connectionId: string; method: string }) => {
      expect(request.method).toBe('GET')

      if (denied) {
        throw new Error('403: forbidden')
      }

      return { boards: [{ slug: 'shared', name: 'Shared' }], current: 'shared' }
    })

    vi.stubGlobal('hermesDesktop', {
      api,
      connections: { list: async () => ({ connections: [source('a')], primary: 'a' }) }
    })
    const reads = createBoardBrowserReads(createPluginContext('kanban').rest)
    render(
      <QueryClientProvider client={qc}>
        <BoardBrowser reads={reads} />
      </QueryClientProvider>
    )
    const pick = await screen.findByRole('button', { name: 'Synthetic a / worker / Shared' })
    denied = true
    await act(async () => {
      await qc.invalidateQueries({ queryKey: browserKeys.boards({ connectionId: 'a', profile: 'worker', label: '' }) })
    })
    expect(await screen.findByText('Access denied. Check this connection’s authentication.')).toBeTruthy()
    expect(screen.getByText('Stale: last successful snapshot, not current state.')).toBeTruthy()
    expect(pick).toBeTruthy()
    denied = false
    fireEvent.click(
      within(screen.getByRole('region', { name: 'Synthetic a / worker' })).getByRole('button', { name: 'Retry' })
    )
    await waitFor(() => expect(screen.queryByText('Access denied. Check this connection’s authentication.')).toBeNull())
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    fireEvent(document, new Event('visibilitychange'))
    const calls = api.mock.calls.length
    await act(async () => {
      await qc.invalidateQueries()
    })
    expect(api).toHaveBeenCalledTimes(calls)
  })
})
