import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $lanesByProfile } from './api'
import { Column } from './board'
import type { KanbanTask } from './types'

const kanbanText = {
  collapse: (label: string) => `Collapse ${label}`,
  empty: 'Empty',
  expand: (label: string) => `Expand ${label}`,
  newTaskIn: (label: string) => `New task in ${label}`
}

const { emptyAtom, lanesByProfile } = vi.hoisted(() => ({
  emptyAtom: { get: () => null, set: () => undefined },
  lanesByProfile: { get: () => false, set: () => undefined }
}))

vi.mock('@hermes/plugin-sdk', () => ({
  Codicon: ({
    name,
    size: _size,
    spinning: _spinning,
    ...props
  }: {
    name: string
    size?: string
    spinning?: boolean
  }) => <span data-codicon={name} {...props} />,
  Tip: ({ children, label }: { children: ReactNode; label: string }) => <span data-tip={label}>{children}</span>,
  cn: (...values: unknown[]) => values.filter(value => typeof value === 'string' && value.length > 0).join(' '),
  useValue: (store: { get: () => unknown }) => store.get()
}))

vi.mock('./api', () => ({
  $boardSlug: emptyAtom,
  $collapsedLanes: emptyAtom,
  $introDismissed: emptyAtom,
  $lanesByProfile: lanesByProfile,
  BOARDS_KEY: [],
  PROFILES_KEY: [],
  boardKey: vi.fn(),
  bulkTasks: vi.fn(),
  createTask: vi.fn(),
  deleteTask: vi.fn(),
  estimateNew: vi.fn(),
  fetchBoard: vi.fn(),
  fetchBoards: vi.fn(),
  fetchProfiles: vi.fn(),
  patchTask: vi.fn()
}))

vi.mock('./ui', () => ({
  columnHelp: vi.fn(() => 'Ready lane help'),
  columnLabel: vi.fn(() => 'Ready'),
  isLockedTarget: vi.fn(() => false),
  useKanban: () => kanbanText
}))

vi.mock('./board-switcher', () => ({ BoardSwitcher: () => null }))
vi.mock('./drawer', () => ({ TaskDrawer: () => null }))
vi.mock('./model-override', () => ({
  EMPTY_OVERRIDE: {},
  ModelOverrideField: () => null,
  overrideCreateFields: () => ({})
}))
vi.mock('./orchestration', () => ({ OrchestrationPanel: () => null }))

afterEach(() => {
  cleanup()
  $lanesByProfile.set(false)
  vi.clearAllMocks()
})

const column = { name: 'ready', tasks: [] as KanbanTask[] }

function renderColumn(collapsed: boolean, onToggle = vi.fn()) {
  return {
    onToggle,
    ...render(
      <Column
        collapsed={collapsed}
        column={column}
        columns={['ready']}
        onAdd={vi.fn()}
        onDelete={vi.fn()}
        onDropTask={vi.fn()}
        onMove={vi.fn()}
        onOpen={vi.fn()}
        onToggle={onToggle}
        onToggleSelect={vi.fn()}
        selected={new Set()}
      />
    )
  }
}

describe('kanban column lane controls', () => {
  it('renders one full-width native collapse button for an expanded lane header', () => {
    const { container, onToggle } = renderColumn(false)
    const collapse = screen.getByRole('button', { name: 'Collapse Ready' })

    expect(collapse.getAttribute('type')).toBe('button')
    expect(collapse.classList.contains('h-5')).toBe(true)
    expect(collapse.classList.contains('w-full')).toBe(true)
    expect(container.querySelector('header')).toBeNull()
    expect(collapse.querySelector('button')).toBeNull()
    expect(collapse.contains(screen.getByText('Ready'))).toBe(true)
    expect(collapse.textContent).toContain('0')
    expect(collapse.querySelector('[data-codicon="chevron-left"]')).toBeTruthy()
    expect(screen.getByText('Ready').closest('[data-tip]')?.getAttribute('data-tip')).toBe('Ready lane help')

    fireEvent.click(collapse)

    expect(onToggle).toHaveBeenCalledTimes(1)
  })

  it('leaves the collapsed rail as a full-height native expand button', () => {
    const { onToggle } = renderColumn(true)
    const expand = screen.getByRole('button', { name: 'Expand Ready' })

    expect(expand.getAttribute('type')).toBe('button')
    expect(expand.classList.contains('h-full')).toBe(true)

    fireEvent.click(expand)

    expect(onToggle).toHaveBeenCalledTimes(1)
  })

  it('collapsed rail button has keyboard focus-visible highlight matching the project pattern', () => {
    renderColumn(true)
    const expand = screen.getByRole('button', { name: 'Expand Ready' })
    const cls = expand.className

    expect(cls).toContain('focus-visible:bg-(--chrome-action-hover)')
    expect(cls).toContain('focus-visible:text-foreground')
    expect(cls).toContain('focus-visible:outline-none')
    expect(cls).toContain('focus-visible:ring-2')
    expect(cls).toContain('focus-visible:ring-ring/40')
  })

  it('activates natively on Enter and Space via keyboard', async () => {
    const user = userEvent.setup()
    const { onToggle } = renderColumn(false)
    const collapse = screen.getByRole('button', { name: 'Collapse Ready' })

    collapse.focus()
    await user.keyboard('{Enter}')
    expect(onToggle).toHaveBeenCalledTimes(1)

    await user.keyboard(' ')
    expect(onToggle).toHaveBeenCalledTimes(2)
  })
})
