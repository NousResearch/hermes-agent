import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesRepoStatus } from '@/global'
import { I18nProvider } from '@/i18n'
import { $repoStatus, $repoWorktrees } from '@/store/coding-status'
import { $newWorktreeRequest } from '@/store/projects'

import { CodingStatusRow } from './coding-row'

const worktreeDialog = vi.fn((props: { initialMode: string; open: boolean }) => (
  <div data-mode={props.initialMode} data-open={props.open ? '1' : '0'} data-testid="worktree-dialog" />
))

vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children }: { children: ReactNode }) => <>{children}</>,
  DropdownMenuContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
    onSelect
  }: {
    children: ReactNode
    onClick?: () => void
    onSelect?: () => void
  }) => <button onClick={() => (onSelect || onClick)?.()}>{children}</button>,
  DropdownMenuLabel: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DropdownMenuSeparator: () => <div />,
  DropdownMenuTrigger: ({ children }: { children: ReactNode }) => <>{children}</>
}))

vi.mock('@/app/chat/sidebar/projects/worktree-dialog', () => ({
  WorktreeDialog: (props: { initialMode: string; open: boolean }) => {
    worktreeDialog(props)

    return <div data-testid="worktree-dialog" />
  }
}))

beforeAll(() => {
  const proto = Element.prototype as unknown as Record<string, () => unknown>

  const stubs = {
    hasPointerCapture: () => false,
    releasePointerCapture: () => undefined,
    scrollIntoView: () => undefined,
    setPointerCapture: () => undefined
  }

  for (const [name, fn] of Object.entries(stubs)) {
    proto[name] ??= fn
  }
})

function setActiveRepoState(overrides: Partial<HermesRepoStatus> = {}) {
  $repoStatus.set({
    added: 0,
    ahead: 0,
    branch: 'main',
    changed: 0,
    conflicted: 0,
    defaultBranch: 'main',
    detached: false,
    files: [],
    removed: 0,
    staged: 0,
    unstaged: 0,
    behind: 0,
    untracked: 0,
    ...overrides
  })
}

function renderRow(onOpenWorktree = vi.fn()) {
  worktreeDialog.mockClear()

  render(
    <I18nProvider configClient={null}>
      <CodingStatusRow
        onBranchOff={() => Promise.resolve()}
        onConvertBranch={() => Promise.resolve()}
        onOpen={() => undefined}
        onOpenWorktree={onOpenWorktree}
        repoPath="/repo"
      />
    </I18nProvider>
  )

  return { onOpenWorktree }
}

function latestDialogProps() {
  const props = worktreeDialog.mock.lastCall?.[0] as undefined | { initialMode: string; open: boolean }

  if (!props) {
    throw new Error('WorktreeDialog was not rendered')
  }

  return props
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $repoStatus.set(null)
  $repoWorktrees.set([])
  $newWorktreeRequest.set(0)
})

beforeEach(() => {
  setActiveRepoState()
})

describe('CodingStatusRow worktree dialog mode', () => {
  it('opens Start Work in create mode', async () => {
    renderRow()

    fireEvent.click(screen.getByRole('button', { name: 'New worktree' }))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'create', open: true })
    })
  })

  it('opens Convert Branch in convert mode', async () => {
    renderRow()

    fireEvent.click(screen.getByRole('button', { name: 'Convert a branch…' }))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'convert', open: true })
    })
  })

  it('opens the global new-worktree request in create mode', async () => {
    renderRow()
    act(() => $newWorktreeRequest.set(1))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'create', open: true })
    })
  })
})
