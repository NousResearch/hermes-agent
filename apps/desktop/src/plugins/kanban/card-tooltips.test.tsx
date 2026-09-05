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

async function renderBoard(waitForTitle = 'Card with every field') {
  const { KanbanBoardPage } = await import('./board')

  render(
    <QueryClientProvider client={queryClient}>
      <KanbanBoardPage />
    </QueryClientProvider>
  )

  await screen.findByText(waitForTitle)
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

  it('shows a tooltip on the links count with a direction-neutral claim matching the badge total (children-only fixture)', async () => {
    await renderBoard()

    const trigger = document.querySelector('.codicon-references')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    const text = screen.getByRole('tooltip').textContent
    // Badge shows parents+children = 0+3 = 3; copy must match that total and
    // break it down, not make a directional "Blocks" claim.
    expect(text).toContain('Linked to 3 other tasks')
    expect(text).toContain('0 parents')
    expect(text).toContain('3 children')
    expect(text).not.toContain('Blocks')
    expect(text).not.toContain('blocking this card')
  })

  it('shows the links tooltip matching the total for a parents-only fixture', async () => {
    const parentsOnlyBoard: KanbanBoard = {
      ...testBoard,
      columns: [
        {
          name: 'done',
          tasks: [
            {
              ...testBoard.columns[0].tasks[0],
              id: 't_parentsonly',
              link_counts: { children: 0, parents: 2 },
              title: 'Parents-only links card'
            }
          ]
        }
      ]
    }
    const api = await import('./api')
    ;(api.fetchBoard as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(parentsOnlyBoard)

    await renderBoard('Parents-only links card')

    const trigger = document.querySelector('.codicon-references')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    const text = screen.getByRole('tooltip').textContent
    // Badge total is 2+0 = 2. A children-only binding (the M6 mutation) would
    // render "0" here instead — this fixture is the one that kills it.
    expect(text).toContain('Linked to 2 other tasks')
    expect(text).toContain('2 parents')
    expect(text).toContain('0 children')
  })

  it('shows the links tooltip matching the combined total for a mixed parents+children fixture', async () => {
    const mixedBoard: KanbanBoard = {
      ...testBoard,
      columns: [
        {
          name: 'done',
          tasks: [
            {
              ...testBoard.columns[0].tasks[0],
              id: 't_mixedlinks1',
              link_counts: { children: 3, parents: 1 },
              title: 'Mixed links card'
            }
          ]
        }
      ]
    }
    const api = await import('./api')
    ;(api.fetchBoard as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mixedBoard)

    await renderBoard('Mixed links card')

    const trigger = document.querySelector('.codicon-references')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    const text = screen.getByRole('tooltip').textContent
    // Badge total is 1+3 = 4, not 3 (children-only) and not directional.
    expect(text).toContain('Linked to 4 other tasks')
    expect(text).toContain('1 parent,')
    expect(text).toContain('3 children')
  })

  it('shows a tooltip on the warnings badge including the highest severity', async () => {
    await renderBoard()

    const trigger = document.querySelector('.codicon-warning')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    expect(screen.getByRole('tooltip').textContent).toContain('critical')
  })

  it('omits the severity clause entirely when highest_severity is null', async () => {
    const nullSeverityBoard: KanbanBoard = {
      ...testBoard,
      columns: [
        {
          name: 'done',
          tasks: [
            {
              ...testBoard.columns[0].tasks[0],
              id: 't_nullsev0001',
              title: 'Null severity card',
              warnings: { count: 1, highest_severity: null }
            }
          ]
        }
      ]
    }
    const api = await import('./api')
    ;(api.fetchBoard as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(nullSeverityBoard)

    await renderBoard('Null severity card')

    const trigger = document.querySelector('.codicon-warning')!.closest('[data-slot="tooltip-trigger"]')!
    await hoverTip(trigger)

    const text = screen.getByRole('tooltip').textContent
    expect(text).toContain('1 warning. Open the card for details.')
    expect(text).not.toContain('highest severity')
    expect(text).not.toContain('highest severity: .')
  })

  it('shows a tooltip on the short task id revealing the full id', async () => {
    await renderBoard()

    await hoverTip(screen.getByText('test12').closest('[data-slot="tooltip-trigger"]')!)

    expect(screen.getByRole('tooltip').textContent).toContain('t_test123456')
  })

  it('shows a tooltip on the plain assignee avatar (native title= converted to Tip)', async () => {
    await renderBoard()

    const avatarInitials = screen.getByText('A')
    expect(avatarInitials.closest('span')?.hasAttribute('title')).toBe(false)
    const trigger = avatarInitials.closest('[data-slot="tooltip-trigger"]')!
    expect(trigger).toBeTruthy()

    await hoverTip(trigger)

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
