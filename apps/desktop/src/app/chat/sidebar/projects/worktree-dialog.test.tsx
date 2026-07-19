import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitBaseBranch, HermesGitBranch } from '@/global'
import { I18nProvider } from '@/i18n'

import { WorktreeDialog } from './worktree-dialog'

vi.mock('./base-branch-picker', () => ({
  BaseBranchPicker: () => <div data-testid="base-branch-picker" />
}))

const { listBaseBranches, listRepoBranches, startWorkInRepo, switchBranchInRepo, $worktreeRefreshToken } = vi.hoisted(
  () => ({
    listBaseBranches: vi.fn<(repoPath: string) => Promise<HermesGitBaseBranch[]>>(),
    listRepoBranches: vi.fn<(repoPath: string) => Promise<HermesGitBranch[]>>(),
    startWorkInRepo: vi.fn(),
    switchBranchInRepo: vi.fn(),
    $worktreeRefreshToken: {
      subscribe: vi.fn(() => vi.fn())
    }
  })
)

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken,
  listBaseBranches: (repoPath: string) => listBaseBranches(repoPath),
  listRepoBranches: (repoPath: string) => listRepoBranches(repoPath),
  startWorkInRepo: (...args: unknown[]) => startWorkInRepo(...args),
  switchBranchInRepo: (...args: unknown[]) => switchBranchInRepo(...args)
}))

beforeAll(() => {
  globalThis.ResizeObserver = class {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
    constructor() {}
  } as any
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  listBaseBranches.mockResolvedValue([] as HermesGitBaseBranch[])
  listRepoBranches.mockResolvedValue([] as HermesGitBranch[])
  startWorkInRepo.mockReset()
  switchBranchInRepo.mockReset()
})

describe('WorktreeDialog mode initialization', () => {
  it('shows create content when opened in create mode', () => {
    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog
          initialBase="main"
          initialMode="create"
          onOpenChange={() => undefined}
          onStarted={() => undefined}
          open
          repoPath="/repo"
        />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()
    expect(screen.getByPlaceholderText('e.g. my-feature')).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Convert a branch' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Convert an existing branch' })).toBeTruthy()
    expect(screen.queryByPlaceholderText('Search branches…')).toBeNull()
    expect(listRepoBranches).not.toHaveBeenCalled()
  })

  it('loads and shows branch conversion content when opened in convert mode', async () => {
    listRepoBranches.mockResolvedValue([
      { checkedOut: false, isDefault: true, name: 'main', worktreePath: null },
      { checkedOut: false, isDefault: false, name: 'feature/quick-fix', worktreePath: null }
    ])

    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog
          initialMode="convert"
          onOpenChange={() => undefined}
          onStarted={() => undefined}
          open
          repoPath="/repo"
        />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'Convert a branch' })).toBeTruthy()
    expect(screen.getByPlaceholderText('Search branches…')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('feature/quick-fix')).toBeTruthy())
    expect(screen.queryByPlaceholderText('e.g. my-feature')).toBeNull()
    expect(listRepoBranches).toHaveBeenCalledWith('/repo')
  })
})
