import { host } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { KanbanCount } from './plugin'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => ({ columns: [{ name: 'running', tasks: [{ id: 'task-1' }] }] }))
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.location.hash = '#/'
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <KanbanCount />
    </QueryClientProvider>
  )

describe('Kanban statusbar toggle', () => {
  it('returns to the route and query string from which the board was opened', async () => {
    window.location.hash = '#/session-1?view=workspace'
    mount()
    const count = await screen.findByRole('button')

    fireEvent.click(count)
    expect(window.location.hash).toBe('#/kanban')
    fireEvent.click(count)
    expect(window.location.hash).toBe('#/session-1?view=workspace')

    window.location.hash = '#/skills'
    fireEvent.click(count)
    fireEvent.click(count)
    expect(window.location.hash).toBe('#/skills')
  })

  it('returns to the workspace when mounted on the board without a prior route', async () => {
    window.location.hash = '#/kanban?board=shipping'
    mount()

    fireEvent.click(await screen.findByRole('button'))

    expect(window.location.hash).toBe('#/')
  })

  it.each(['profile', 'connectionId'] as const)('does not restore a route after the %s changes', async key => {
    const scope = vi.spyOn(host.state[key], 'get').mockReturnValue('first')
    window.location.hash = '#/session-1'
    mount()
    const count = await screen.findByRole('button')
    fireEvent.click(count)
    scope.mockReturnValue('second')

    fireEvent.click(count)

    expect(window.location.hash).toBe('#/')
  })
})
