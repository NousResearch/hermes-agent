import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { Card } from './board'
import { KANBAN_LOCALES } from './i18n'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/orchestration') {return { default_assignee: '' }}

  if (path === '/tasks/t_worker/context') {
    return {
      available: true,
      context_used: 41_818,
      context_max: 200_000,
      estimated: true,
      source: 'provider_usage_plus_estimate'
    }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.useRealTimers()
  vi.clearAllMocks()
})

it('fetches worker context only when the running card is hovered', async () => {
  render(
    <QueryClientProvider client={client}>
      <Card
        columns={['running']}
        onDelete={vi.fn()}
        onMove={vi.fn()}
        onOpen={vi.fn()}
        onToggleSelect={vi.fn()}
        selected={false}
        task={{ id: 't_worker', title: 'Worker task', status: 'running', current_run_id: 17, worker_pid: 4321 }}
      />
    </QueryClientProvider>
  )

  await waitFor(() => expect(rest).toHaveBeenCalledWith('/orchestration', undefined))
  expect(rest).not.toHaveBeenCalledWith('/tasks/t_worker/context', undefined)

  const card = screen.getByText('Worker task').closest('[draggable="true"]')
  expect(card).not.toBeNull()
  fireEvent.pointerEnter(card!)

  await waitFor(() => expect(rest).toHaveBeenCalledWith('/tasks/t_worker/context', undefined))
  await waitFor(() =>
    expect(client.getQueryData(['kanban', 'task-context', '', 't_worker', 17, 4321])).toMatchObject({
      context_used: 41_818,
      context_max: 200_000,
      estimated: true
    })
  )

  fireEvent.pointerLeave(card!)
  vi.useFakeTimers()
  act(() => {
    fireEvent.pointerMove(card!, { pointerType: 'mouse' })
    vi.advanceTimersByTime(300)
  })
  expect(screen.getByRole('tooltip').textContent).toContain('Context 42k / 200k tokens (estimated)')
})
