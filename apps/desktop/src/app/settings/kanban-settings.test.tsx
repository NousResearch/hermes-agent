import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getKanbanBoard, listKanbanBoards, setApiRequestProfile } from '@/hermes'

import { KanbanSettings } from './kanban-settings'

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

describe('Kanban settings API helpers', () => {
  const api = vi.fn(async (request: { path: string; profile?: string }) => ({ request }) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockReset()
    api.mockImplementation(async request => ({ request }) as never)
    setApiRequestProfile('writer')
  })

  afterEach(() => {
    cleanup()
    setApiRequestProfile(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('routes board discovery through the active profile', async () => {
    await listKanbanBoards()

    expect(api).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/boards',
      profile: 'writer'
    })
  })

  it('routes a selected board through the active profile and encodes its slug', async () => {
    await getKanbanBoard({ board: 'release / train', includeArchived: true })

    expect(api).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/board?board=release+%2F+train&include_archived=true',
      profile: 'writer'
    })
  })

  it('can target the local gateway independently of the ambient API profile', async () => {
    await listKanbanBoards({ gatewayId: 'local', profile: null })
    await getKanbanBoard({ board: 'local-board', gatewayId: 'local', profile: null })

    expect(api).toHaveBeenNthCalledWith(1, {
      gatewayId: 'local',
      path: '/api/plugins/kanban/boards'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      gatewayId: 'local',
      path: '/api/plugins/kanban/board?board=local-board'
    })
  })

  it('threads an explicit gateway target through the Kanban settings surface', async () => {
    api.mockResolvedValue({ boards: [], current: null } as never)

    render(<KanbanSettings gatewayId="local" />)

    expect(await screen.findByText('No Kanban boards')).toBeTruthy()
    expect(api).toHaveBeenCalledWith({
      gatewayId: 'local',
      path: '/api/plugins/kanban/boards',
      profile: 'writer'
    })
  })

  it('shows an explicit unavailable state when the plugin endpoint is disabled', async () => {
    api.mockRejectedValue(new Error('404: {"detail":"Plugin not found"}'))

    render(<KanbanSettings />)

    expect(await screen.findByText('Kanban is unavailable')).toBeTruthy()
    expect(screen.queryByText('No Kanban boards')).toBeNull()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('clears the previous board when a newly selected board fails', async () => {
    api.mockImplementation(async request => {
      if (request.path === '/api/plugins/kanban/boards') {
        return {
          boards: [
            { name: 'Alpha', slug: 'alpha', is_current: true },
            { name: 'Beta', slug: 'beta' }
          ],
          current: 'alpha'
        } as never
      }

      if (request.path.endsWith('board=alpha')) {
        return { columns: [{ name: 'Todo', tasks: [{ title: 'Alpha task' }] }] } as never
      }

      if (request.path.endsWith('board=beta')) {
        throw new Error('503: board offline')
      }

      throw new Error(`Unexpected request: ${request.path}`)
    })

    render(<KanbanSettings />)
    expect(await screen.findByText('Alpha task')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Beta/ }))

    expect(await screen.findByText('Board unavailable')).toBeTruthy()
    expect(screen.queryByText('Alpha task')).toBeNull()
  })
})
