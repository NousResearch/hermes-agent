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

// Minimal nanostores-compatible store mock that supports useStore from @nanostores/react.
// Defined inside vi.hoisted so the class is available where the store instances are created.
const { $worktreeDialog, $projectTree, listBaseBranches, listRepoBranches, startWorkInRepo, switchBranchInRepo } =
  vi.hoisted(() => {
    class MockStore<T> {
      private value: T
      private listeners: Set<(value: T) => void> = new Set()

      constructor(initial: T) {
        this.value = initial
      }

      get() {
        return this.value
      }

      set(next: T) {
        this.value = next

        for (const cb of this.listeners) {
          cb(this.value)
        }
      }

      subscribe(cb: (value: T) => void) {
        this.listeners.add(cb)
        cb(this.value)

        return () => {
          this.listeners.delete(cb)
        }
      }

      listen(cb: (value: T) => void) {
        this.listeners.add(cb)
        cb(this.value)

        return () => {
          this.listeners.delete(cb)
        }
      }

      notify() {}
    }

    return {
      $worktreeDialog: new MockStore<null | { base?: string; mode?: 'create' | 'convert'; repoPath: string }>(null),
      $projectTree: new MockStore([]),
      listBaseBranches: vi.fn<(repoPath: string) => Promise<HermesGitBaseBranch[]>>(),
      listRepoBranches: vi.fn<(repoPath: string) => Promise<HermesGitBranch[]>>(),
      startWorkInRepo: vi.fn(),
      switchBranchInRepo: vi.fn()
    }
  })

vi.mock('@/store/projects', () => ({
  $projectTree,
  $worktreeDialog,
  closeWorktreeDialog: vi.fn(),
  listBaseBranches: (repoPath: string) => listBaseBranches(repoPath),
  listRepoBranches: (repoPath: string) => listRepoBranches(repoPath),
  projectIdForCwd: vi.fn(() => null),
  projectRootCwd: vi.fn(() => null),
  requestStartWorkSession: vi.fn(),
  startWorkInRepo: (...args: unknown[]) => startWorkInRepo(...args),
  switchBranchInRepo: (...args: unknown[]) => switchBranchInRepo(...args)
}))

vi.mock('@/store/coding-status', () => ({
  registerRepoStatusCwd: () => undefined,
  repoStatusForCwd: () => vi.fn(),
  repoWorktreesForCwd: () => vi.fn(),
  resolveWorktreeRepoPath: vi.fn().mockResolvedValue('/repo')
}))

vi.mock('@/lib/sanitize', () => ({
  gitRef: (s: string) => s
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

vi.mock('./base-branch-picker', () => ({
  BaseBranchPicker: () => <div data-testid="base-branch-picker" />
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
  $worktreeDialog.set(null)
})

beforeEach(() => {
  listBaseBranches.mockResolvedValue([] as HermesGitBaseBranch[])
  listRepoBranches.mockResolvedValue([] as HermesGitBranch[])
  startWorkInRepo.mockReset()
  switchBranchInRepo.mockReset()
})

describe('WorktreeDialog mode initialization', () => {
  it('shows create content when opened in create mode', () => {
    $worktreeDialog.set({ repoPath: '/repo', mode: 'create' })

    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()
    expect(screen.getByPlaceholderText('e.g. my-feature')).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Convert a branch' })).toBeNull()
    expect(listRepoBranches).not.toHaveBeenCalled()
  })

  it('shows create content by default when mode is omitted', () => {
    $worktreeDialog.set({ repoPath: '/repo' })

    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()
  })

  it('loads and shows branch conversion content when opened in convert mode', async () => {
    listRepoBranches.mockResolvedValue([
      { checkedOut: false, isDefault: true, isRemote: false, name: 'main', worktreePath: null },
      { checkedOut: false, isDefault: false, isRemote: false, name: 'feature/quick-fix', worktreePath: null }
    ])

    $worktreeDialog.set({ repoPath: '/repo', mode: 'convert' })

    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'Convert a branch' })).toBeTruthy()
    expect(screen.getByPlaceholderText('Search branches…')).toBeTruthy()
    await waitFor(() => expect(screen.getByText('feature/quick-fix')).toBeTruthy())
    expect(screen.queryByPlaceholderText('e.g. my-feature')).toBeNull()
    expect(listRepoBranches).toHaveBeenCalledWith('/repo')
  })

  it('remounts with the requested mode when the dialog reopens after close', async () => {
    $worktreeDialog.set({ repoPath: '/repo', mode: 'create' })

    const { rerender } = render(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()

    act(() => {
      $worktreeDialog.set(null)
    })
    rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )
    expect(screen.queryByRole('heading', { name: 'New worktree' })).toBeNull()

    act(() => {
      $worktreeDialog.set({ repoPath: '/repo', mode: 'convert' })
    })
    rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'Convert a branch' })).toBeTruthy()

    act(() => {
      $worktreeDialog.set(null)
    })
    rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )
    act(() => {
      $worktreeDialog.set({ repoPath: '/repo', mode: 'create' })
    })
    rerender(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    expect(screen.getByRole('heading', { name: 'New worktree' })).toBeTruthy()
  })

  it('ignores a stale branch response after reopening for another repository', async () => {
    const repoA = deferred<HermesGitBranch[]>()
    const repoB = deferred<HermesGitBranch[]>()
    listRepoBranches.mockImplementation(repoPath => {
      if (repoPath === '/repo-a') {
        return repoA.promise
      }

      if (repoPath === '/repo-b') {
        return repoB.promise
      }

      return Promise.resolve([])
    })

    $worktreeDialog.set({ repoPath: '/repo-a', mode: 'convert' })

    render(
      <I18nProvider configClient={null}>
        <WorktreeDialog />
      </I18nProvider>
    )

    await waitFor(() => expect(listRepoBranches).toHaveBeenCalledWith('/repo-a'))

    // Reopen for a different repo — invalidates the first load
    act(() => {
      $worktreeDialog.set(null)
    })
    act(() => {
      $worktreeDialog.set({ repoPath: '/repo-b', mode: 'convert' })
    })

    await waitFor(() => expect(listRepoBranches).toHaveBeenCalledWith('/repo-b'))

    await act(async () => {
      repoB.resolve([
        { checkedOut: false, isDefault: false, isRemote: false, name: 'repo-b-branch', worktreePath: null }
      ])
      await repoB.promise
    })
    await waitFor(() => expect(screen.getByText('repo-b-branch')).toBeTruthy())

    // The stale repoA response must not appear
    await act(async () => {
      repoA.resolve([
        { checkedOut: false, isDefault: false, isRemote: false, name: 'repo-a-branch', worktreePath: null }
      ])
      await repoA.promise
    })

    expect(screen.queryByText('repo-a-branch')).toBeNull()
    expect(screen.getByText('repo-b-branch')).toBeTruthy()
  })
})
