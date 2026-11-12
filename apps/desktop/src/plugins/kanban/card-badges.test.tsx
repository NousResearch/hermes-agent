import { KANBAN_CARD_AREAS, type KanbanCardBadgeContribution } from '@hermes/plugin-sdk'
/**
 * The board paints plugin card badges (#140026): the REAL board page, the REAL
 * plugin context and registry, with only the kanban REST layer mocked. A
 * plugin never touches the card's DOM; it returns text and a palette tone and
 * the board renders it in the card's own badge row.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { createPluginContext } from '@/contrib/plugin'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import type * as KanbanApi from './api'
import { $boardSlug } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoards: vi.fn(async () => ({
    boards: [{ name: 'Main', project_id: null, slug: 'main', total: 2 }],
    current: 'main'
  })),
  fetchBoard: vi.fn(async () => ({
    assignees: [],
    columns: [
      {
        name: 'ready',
        tasks: [
          { id: 't_1a2b3c', status: 'ready', tenant: 'acme', title: 'Ship the acme report' },
          { id: 't_4d5e6f', status: 'ready', tenant: null, title: 'Untenanted chore' }
        ]
      }
    ],
    latest_event_id: 0,
    now: 0,
    tenants: ['acme', 'globex']
  })),
  fetchOrchestration: vi.fn(async () => ({ default_assignee: '' })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] }))
}))

let disposeLocales: () => void = () => undefined
let unloadPlugin: () => void = () => undefined
let seenTenants: string[] = []

beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)

  const disposers: Array<() => void> = []
  const ctx = createPluginContext('tenant-labels', dispose => disposers.push(dispose))

  ctx.register({
    area: KANBAN_CARD_AREAS.badges,
    data: {
      decorate: ({ tenant }, { tenants }) => {
        seenTenants = [...tenants]

        return tenant
          ? [
              { label: tenant, tone: 'blue', tooltip: `Tenant ${tenant} of ${tenants.length}` },
              { label: '<b>bold</b>' }
            ]
          : null
      }
    } satisfies KanbanCardBadgeContribution,
    id: 'tenant'
  })
  unloadPlugin = () => disposers.splice(0).forEach(dispose => dispose())
})

afterEach(() => {
  cleanup()
  unloadPlugin()
  disposeLocales()
  $boardSlug.set('')
})

const renderBoard = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

const tileOf = (title: string) => screen.getByText(title).closest<HTMLElement>('[draggable="true"]')!

describe('kanban board: plugin card badges', () => {
  it('paints a plugin badge as plain text on the card it was decorated for, and nowhere else', async () => {
    renderBoard()
    await screen.findByText('Ship the acme report')

    const acme = within(tileOf('Ship the acme report'))
    const badge = acme.getByText('acme')

    expect(badge.closest('[data-slot="kanban-card-badge"]')?.getAttribute('data-tone')).toBe('blue')
    // The board's tenant list reaches the decorator (collision-free colors need it).
    expect(seenTenants).toEqual(['acme', 'globex'])
    // Markup-looking text stays text: no element is created from a label.
    expect(acme.getByText('<b>bold</b>').querySelector('b')).toBeNull()
    expect(tileOf('Untenanted chore').querySelector('[data-slot="kanban-card-badges"]')).toBeNull()
  })

  it('unloading the plugin removes its badges from the live board', async () => {
    const { container } = renderBoard()
    await screen.findByText('acme')

    act(() => unloadPlugin())

    expect(screen.queryByText('acme')).toBeNull()
    expect(container.querySelector('[data-slot="kanban-card-badges"]')).toBeNull()
  })
})
