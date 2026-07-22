import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitBranch } from '@/global'
import { listRepoBranches } from '@/store/projects'

import { WorktreeDialog } from './worktree-dialog'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      sidebar: {
        projects: {
          branchCreateWorktree: 'Create worktree',
          branchOpenExisting: 'Open existing worktree',
          branchPlaceholder: 'Branch name',
          branchSwitchHome: 'Switch primary checkout',
          branchesLoading: 'Loading branches',
          convertBranchDesc: 'Choose an existing branch',
          convertBranchInstead: 'Convert a branch instead',
          convertBranchPlaceholder: 'Find a branch',
          convertBranchTitle: 'Convert branch',
          newWorktreeDesc: 'Create a worktree',
          newWorktreeTitle: 'New worktree',
          noBranches: 'No branches',
          startWork: 'Start work',
          startWorkFailed: 'Could not start worktree'
        }
      }
    }
  })
}))

vi.mock('@/store/projects', () => ({
  listRepoBranches: vi.fn(),
  startWorkInRepo: vi.fn(),
  switchBranchInRepo: vi.fn()
}))

vi.mock('./base-branch-picker', () => ({
  BaseBranchPicker: ({ value }: { value: string }) => <output data-testid="base-branch">{value}</output>
}))

beforeEach(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      disconnect() {}
      observe() {}
      unobserve() {}
    }
  )
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

describe('WorktreeDialog', () => {
  it('drops a late branch list after reopening in a new repository context', async () => {
    let resolveBranchesA: (branches: HermesGitBranch[]) => void = () => undefined
    let resolveBranchesB: (branches: HermesGitBranch[]) => void = () => undefined

    const branchesA = new Promise<HermesGitBranch[]>(resolve => {
      resolveBranchesA = resolve
    })

    const branchesB = new Promise<HermesGitBranch[]>(resolve => {
      resolveBranchesB = resolve
    })

    const onOpenChange = vi.fn()

    vi.mocked(listRepoBranches).mockReturnValueOnce(branchesA).mockReturnValueOnce(branchesB)

    const view = render(
      <WorktreeDialog initialBase="main" onOpenChange={onOpenChange} onStarted={vi.fn()} open repoPath="/repo-a" />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Convert a branch instead' }))
    expect(listRepoBranches).toHaveBeenCalledWith('/repo-a')

    view.rerender(
      <WorktreeDialog initialBase="main" onOpenChange={onOpenChange} onStarted={vi.fn()} open repoPath="/repo-b" />
    )
    fireEvent.click(screen.getByRole('button', { name: 'Convert a branch instead' }))
    expect(listRepoBranches).toHaveBeenLastCalledWith('/repo-b')

    resolveBranchesA([{ checkedOut: false, isDefault: false, name: 'feature/a', worktreePath: null }])

    await waitFor(() => expect(screen.queryByText('feature/a')).toBeNull())

    resolveBranchesB([{ checkedOut: false, isDefault: false, name: 'feature/b', worktreePath: null }])

    await screen.findByText('feature/b')
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
