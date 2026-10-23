import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, fetchOrchestration, saveOrchestration } from './api'
import { KANBAN_LOCALES } from './i18n'
import { OrchestrationPanel } from './orchestration'

// Mutable host facts so each test can set the machine's current board.
const hostage = vi.hoisted(() => ({ current: 'tsa-mgmt' }))

const ORCHESTRATION_PAYLOAD = {
  orchestrator_profile: 'planner',
  default_assignee: 'worker',
  auto_decompose: true,
  resolved_orchestrator_profile: 'planner',
  resolved_default_assignee: 'worker',
  board: 'tsa-mgmt',
  board_orchestrator_profile: 'planner',
  board_default_assignee: ''
}

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [{ name: 'Tsa', project_id: null, slug: 'tsa-mgmt', total: 0 }],
    current: hostage.current
  })),
  fetchOrchestration: vi.fn(async () => ({ ...ORCHESTRATION_PAYLOAD })),
  fetchProfiles: vi.fn(async () => ({
    profiles: [{ name: 'planner', is_default: false, description: 'plans', description_auto: false }]
  })),
  saveOrchestration: vi.fn(async () => ({ ...ORCHESTRATION_PAYLOAD }))
}))

let disposeLocales: () => void = () => undefined

beforeEach(() => {
  vi.clearAllMocks()
  hostage.current = 'tsa-mgmt'
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  $boardSlug.set('')
})

const mount = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <OrchestrationPanel />
    </QueryClientProvider>
  )

describe('orchestration panel (board scope)', () => {
  it('scopes to the effective board when no explicit selection is made', async () => {
    // $boardSlug '' means "follow boards.current": the panel must scope the
    // request there rather than silently reading the global.
    $boardSlug.set('')
    mount()

    await waitFor(() => expect(fetchOrchestration).toHaveBeenCalledWith('tsa-mgmt'))
    // Board scope is active: the override/inherit tags render.
    expect(await screen.findByText(/Orchestrator profile .* overridden here/)).toBeTruthy()
    expect(screen.getByText(/Default assignee .* inherited from global/)).toBeTruthy()
  })

  it('tags each knob overridden vs inherited on this board', async () => {
    $boardSlug.set('tsa-mgmt')
    mount()

    // The board override wins for the orchestrator, so it is tagged as such;
    // the assignee has no board key, so it reads as inherited from global.
    expect(await screen.findByText(/Orchestrator profile .* overridden here/)).toBeTruthy()
    expect(screen.getByText(/Default assignee .* inherited from global/)).toBeTruthy()
  })

  it('clears both board overrides when the clear button is pressed', async () => {
    $boardSlug.set('tsa-mgmt')
    mount()

    fireEvent.click(await screen.findByRole('button', { name: 'Clear board override' }))
    await waitFor(() =>
      expect(saveOrchestration).toHaveBeenCalledWith('tsa-mgmt', { orchestrator_profile: '', default_assignee: '' })
    )
  })

  it('switches to the global scope: no tags, no clear, edits go global', async () => {
    $boardSlug.set('tsa-mgmt')
    mount()

    fireEvent.click(await screen.findByRole('button', { name: 'Global defaults' }))

    // Global scope has no board tags and no clear control.
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Clear board override' })).toBeNull())
    expect(screen.queryByText(/overridden here/)).toBeNull()
    expect(screen.queryByText(/inherited from global/)).toBeNull()
    // The read and the write both drop the board param.
    await waitFor(() => expect(fetchOrchestration).toHaveBeenCalledWith(''))
    fireEvent.click(screen.getByRole('switch', { name: 'Auto-decompose triage tasks' }))
    await waitFor(() => expect(saveOrchestration).toHaveBeenCalledWith('', { auto_decompose: false }))
  })

  it('offers no board controls without a board in view', async () => {
    hostage.current = ''
    $boardSlug.set('')
    mount()

    await waitFor(() => expect(fetchOrchestration).toHaveBeenCalledWith(''))
    expect(screen.queryByRole('button', { name: 'Clear board override' })).toBeNull()
    expect(screen.queryByText('This board')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Global defaults' })).toBeNull()
  })
})
