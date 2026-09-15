import { queryClient } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import * as api from './plugins/kanban/api'
import { KanbanBoardPage } from './plugins/kanban/board'

const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries')

let stored: Record<string, unknown> = {}

const storage = {
  get: <T,>(key: string, fallback: T): T => (key in stored ? (stored[key] as T) : fallback),
  set: vi.fn((key: string, value: unknown) => {
    stored[key] = value
  })
}

beforeEach(() => {
  queryClient.clear()
  invalidateQueries.mockClear()
  storage.set.mockClear()
  stored = {
    boardSlug: 'legacy',
    boardSlugsByScope: { 'local::admin': 'operations', 'local::default': 'shipping' }
  }
  $connection.set({ connectionId: 'local', mode: 'local' } as never)
  $activeGatewayProfile.set('default')
  api.$boardSlug.set('')
})

afterEach(cleanup)

describe('Kanban server scope', () => {
  it('isolates cache identity and restores each server scope\'s selected board', () => {
    const closes: Array<ReturnType<typeof vi.fn>> = []
    const frames: Array<(data: unknown) => void> = []

    const socket = vi.fn((_path: string, onMessage: (data: unknown) => void) => {
      const close = vi.fn()
      closes.push(close)
      frames.push(onMessage)

      return close
    })

    const dispose = api.bindApi(vi.fn(async () => ({})) as never, storage as never, socket)

    const defaultKey = api.boardKey('shipping', false)
    expect(socket).toHaveBeenLastCalledWith('/events?board=shipping', expect.any(Function))

    $activeGatewayProfile.set('admin')

    expect(closes[0]).toHaveBeenCalledOnce()
    expect(socket).toHaveBeenCalledTimes(2)
    expect(socket).toHaveBeenLastCalledWith('/events?board=operations', expect.any(Function))
    expect(api.boardKey('shipping', false)).not.toEqual(defaultKey)
    frames[0]({ events: [{ id: 1, kind: 'created', task_id: 'task-1' }] })
    frames[1]({ events: [{ id: 2, kind: 'spawned', task_id: 'task-2' }] })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['kanban', 'board', 'local::default', 'shipping']
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['kanban', 'board', 'local::admin', 'operations']
    })

    api.$boardSlug.set('incidents')
    expect(socket).toHaveBeenCalledTimes(3)
    expect(socket).toHaveBeenLastCalledWith('/events?board=incidents', expect.any(Function))

    $activeGatewayProfile.set('default')
    expect(api.$boardSlug.get()).toBe('shipping')

    $activeGatewayProfile.set('admin')
    expect(api.$boardSlug.get()).toBe('incidents')

    $connection.set({ connectionId: 'remote-gateway', mode: 'remote' } as never)
    expect(socket).toHaveBeenLastCalledWith('/events', expect.any(Function))
    expect(api.boardKey('', false)).toContain('remote-gateway::admin')

    dispose()
  })

  it('manually refetches the current route and discards its response after a profile switch', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const emptyBoard = { assignees: [], columns: [], latest_event_id: 0, now: Date.now() / 1000, tenants: [] }
    let resolveAdminBoard!: (board: typeof emptyBoard) => void

    const rest = vi.fn(async (path: string) => {
      if (path === '/board?board=operations') {
        return new Promise<typeof emptyBoard>(resolve => {
          resolveAdminBoard = resolve
        })
      }

      if (path.startsWith('/board?')) {
        return emptyBoard
      }

      if (path === '/boards') {
        return { boards: [{ name: 'Shipping', project_id: null, slug: 'shipping', total: 0 }], current: 'shipping' }
      }

      if (path === '/profiles') {
        return { profiles: [] }
      }

      if (path === '/orchestration') {
        return { default_assignee: '', resolved_default_assignee: '', resolved_orchestrator_profile: '' }
      }

      if (path === '/projects') {
        return { projects: [] }
      }

      throw new Error(`unexpected rest call: ${path}`)
    })

    const dispose = api.bindApi(rest as never, storage as never, () => vi.fn())
    api.$introDismissed.set(true)

    render(
      <QueryClientProvider client={client}>
        <KanbanBoardPage />
      </QueryClientProvider>
    )

    const refresh = await screen.findByRole('button', { name: /refresh/i })
    await waitFor(() => expect(rest.mock.calls.filter(([path]) => String(path).startsWith('/board?'))).toHaveLength(1))

    fireEvent.click(refresh)

    await waitFor(() => expect(rest.mock.calls.filter(([path]) => String(path).startsWith('/board?'))).toHaveLength(2))
    expect(rest).toHaveBeenLastCalledWith('/board?board=shipping', undefined)

    $activeGatewayProfile.set('admin')
    await waitFor(() => expect(resolveAdminBoard).toBeTypeOf('function'))
    $activeGatewayProfile.set('default')
    resolveAdminBoard({ ...emptyBoard, latest_event_id: 9 })

    const adminKey = api.boardKey('operations', false, 'local::admin')
    await waitFor(() => expect(client.getQueryState(adminKey)?.fetchStatus).toBe('idle'))
    expect(client.getQueryData(adminKey)).toBeUndefined()

    dispose()
  })
})
