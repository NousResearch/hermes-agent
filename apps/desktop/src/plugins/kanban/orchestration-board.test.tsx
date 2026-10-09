import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug, saveOrchestration } from './api'
import { KANBAN_LOCALES } from './i18n'
import { OrchestrationPanel } from './orchestration'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchOrchestration: vi.fn(async () => ({
    orchestrator_profile: 'planner',
    default_assignee: 'worker',
    auto_decompose: true,
    resolved_orchestrator_profile: 'planner',
    resolved_default_assignee: 'worker',
    board: 'tsa-mgmt',
    board_orchestrator_profile: 'planner',
    board_default_assignee: ''
  })),
  fetchProfiles: vi.fn(async () => ({
    profiles: [{ name: 'planner', is_default: false, description: 'plans', description_auto: false }]
  })),
  saveOrchestration: vi.fn(async () => ({
    orchestrator_profile: 'planner',
    default_assignee: 'worker',
    auto_decompose: true,
    resolved_orchestrator_profile: 'planner',
    resolved_default_assignee: 'worker'
  }))
}))

let disposeLocales: () => void = () => undefined

beforeEach(() => {
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
      expect(saveOrchestration).toHaveBeenCalledWith({ orchestrator_profile: '', default_assignee: '' })
    )
  })

  it('offers no board controls without a board in view', async () => {
    mount()

    expect(screen.queryByRole('button', { name: 'Clear board override' })).toBeNull()
    expect(screen.queryByText('This board')).toBeNull()
  })
})
