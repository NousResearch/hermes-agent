import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitBaseBranch, HermesGitBranch } from '@/global'
import { I18nProvider } from '@/i18n'

import { WorktreeDialog } from './worktree-dialog'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(next => {
    resolve = next
  })

  return { promise, resolve }
}

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

  it('remounts each opening with the requested mode and focus target', async () => {
    const props = {
      onOpenChange: () => undefined,
      onStarted: () => undefined,
      repoPath: '/repo'
    }

    const view = render(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} initialMode="create" open />
      </I18nProvider>
    )

    const branchName = screen.getByPlaceholderText('e.g. my-feature')
    await waitFor(() => expect(globalThis.document.activeElement).toBe(branchName))

    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} initialMode="create" open={false} />
      </I18nProvider>
    )
    expect(screen.queryByRole('heading', { name: 'New worktree' })).toBeNull()

    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} initialMode="convert" open />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'Convert a branch' })).toBeTruthy()
    const branchSearch = screen.getByPlaceholderText('Search branches…')
    await waitFor(() => expect(globalThis.document.activeElement).toBe(branchSearch))

    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} initialMode="convert" open={false} />
      </I18nProvider>
    )
    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} initialMode="create" open />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()
    const reopenedBranchName = screen.getByPlaceholderText('e.g. my-feature')
    await waitFor(() => expect(globalThis.document.activeElement).toBe(reopenedBranchName))
  })

  it('ignores a stale branch response after reopening for another repository', async () => {
    const repoA = deferred<HermesGitBranch[]>()
    const repoB = deferred<HermesGitBranch[]>()
    listRepoBranches.mockImplementationOnce(() => repoA.promise).mockImplementationOnce(() => repoB.promise)

    const props = {
      initialMode: 'convert' as const,
      onOpenChange: () => undefined,
      onStarted: () => undefined
    }

    const view = render(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} open repoPath="/repo-a" />
      </I18nProvider>
    )

    await waitFor(() => expect(listRepoBranches).toHaveBeenCalledWith('/repo-a'))
    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} open={false} repoPath="/repo-a" />
      </I18nProvider>
    )
    view.rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog {...props} open repoPath="/repo-b" />
      </I18nProvider>
    )
    await waitFor(() => expect(listRepoBranches).toHaveBeenCalledWith('/repo-b'))

    await act(async () => {
      repoB.resolve([{ checkedOut: false, isDefault: false, name: 'repo-b-branch', worktreePath: null }])
      await repoB.promise
    })
    await waitFor(() => expect(screen.getByText('repo-b-branch')).toBeTruthy())

    await act(async () => {
      repoA.resolve([{ checkedOut: false, isDefault: false, name: 'repo-a-branch', worktreePath: null }])
      await repoA.promise
    })

    expect(screen.queryByText('repo-a-branch')).toBeNull()
    expect(screen.getByText('repo-b-branch')).toBeTruthy()
  })
})
