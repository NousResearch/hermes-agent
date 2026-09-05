/**
 * Coverage for the tooltip sweep of t_da65e35c: every status icon and
 * free-form field on a board card gets an explanatory hover tooltip, reusing
 * the same `<Tip>` mechanism the dependency chip already used (see the audit
 * at kanban-card-icon-audit.md). This file renders the REAL board page
 * against mocked API calls and drives real pointer events, so a regression
 * that silently drops a `<Tip>` wrapper (reverting to a bare `<span>`) fails
 * here — not just in an isolated unit test of the label strings.
 */

import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { translateKanban } from './i18n-test-helper'
import type { KanbanBoard } from './types'

const queryClient = new (await import('@tanstack/react-query')).QueryClient({
  defaultOptions: { queries: { retry: false } }
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<Record<string, unknown>>()

  return { ...sdk, usePluginI18n: () => translateKanban }
})

const testBoard: KanbanBoard = {
  assignees: ['alice'],
  columns: [
    {
      name: 'done',
      tasks: [
        {
          assignee: 'alice',
          comment_count: 2,
          id: 't_test123456',
          link_counts: { children: 3, parents: 0 },
          priority: 5,
          progress: { done: 1, total: 4 },
          status: 'done',
          title: 'Card with every field',
          warnings: { count: 2, highest_severity: 'critical' }
        }
      ]
    }
  ],
  latest_event_id: 1,
  now: 0,
  tenants: []
}

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchBoard: vi.fn(async () => testBoard),
  fetchBoards: vi.fn(async () => ({ boards: [], current: '' })),
  fetchOrchestration: vi.fn(async () => ({
    auto_decompose: false,
    default_assignee: '',
    orchestrator_profile: '',
    resolved_default_assignee: '',
    resolved_orchestrator_profile: ''
  })),
  fetchProfiles: vi.fn(async () => ({ profiles: [] }))
}))

beforeAll(() => {
  // Radix Tooltip/DropdownMenu call these on interaction; jsdom lacks them.
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

/** Hover a trigger and settle the tip's open delay (Tip's own TooltipProvider,
 *  since these tests don't mount the app-root RootTooltipProvider). */
async function hoverTip(trigger: Element) {
  vi.useFakeTimers()
  act(() => {
    fireEvent.pointerMove(trigger, { pointerType: 'mouse' })
    vi.advanceTimersByTime(300)
  })
}

async function renderBoard() {
  const { KanbanBoardPage } = await import('./board')

  render(
    <QueryClientProvider client={queryClient}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

  await screen.findByText('Card with every field')
}

describe('board card tooltips', () => {
  it('shows a tooltip on the priority badge', async () => {
    await renderBoard()

    await hoverTip(screen.getByText('5').closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('Priority 5')
  })

  it('shows a tooltip on the child-progress checklist', async () => {
    await renderBoard()

    await hoverTip(screen.getByText('1/4').closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('1 of 4 child tasks done')
  })

  it('shows a tooltip on the comment count', async () => {
    await renderBoard()

    const trigger = document.querySelector('.codicon-comment')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    expect(screen.getByRole('tooltip').textContent).toContain('2 comments on this task')
  })

  it('shows a tooltip on the "blocks N" references count, distinct from the dependency-remaining wording', async () => {
    await renderBoard()

    const trigger = document.querySelector('.codicon-references')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    const text = screen.getByRole('tooltip').textContent
    expect(text).toContain('Blocks 3 other tasks')
    expect(text).not.toContain('blocking this card')
  })

  it('shows a tooltip on the warnings badge including the highest severity', async () => {
    await renderBoard()

    const trigger = document.querySelector('.codicon-warning')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    expect(screen.getByRole('tooltip').textContent).toContain('critical')
  })

  it('shows a tooltip on the short task id revealing the full id', async () => {
    await renderBoard()

    await hoverTip(screen.getByText('test12').closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('t_test123456')
  })

  it('shows a tooltip on the plain assignee avatar (native title= converted to Tip)', async () => {
    await renderBoard()

    const avatar = screen.getByTitle('alice')
    expect(avatar.closest('[data-slot="tooltip-trigger"]')).toBeTruthy()

    await hoverTip(avatar.closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('Assigned to alice')
  })

  it('still shows the existing wontRun tooltip unregressed (pre-existing Tip usage)', async () => {
    const unassignedBoard: KanbanBoard = {
      ...testBoard,
      columns: [
        {
          name: 'ready',
          tasks: [{ id: 't_unassigned1', status: 'ready', title: 'Ready, unassigned card' }]
        }
      ]
    }
    const api = await import('./api')
    ;(api.fetchBoard as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(unassignedBoard)

    await renderBoard()
    await screen.findByText('Ready, unassigned card')

    const disconnectIcon = await vi.waitFor(() => {
      const el = document.querySelector('.codicon-debug-disconnect')
      if (!el) throw new Error('not yet rendered')
      return el
    })

    await hoverTip(disconnectIcon.closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('Ready cards only run once a profile is assigned')
  })
})
