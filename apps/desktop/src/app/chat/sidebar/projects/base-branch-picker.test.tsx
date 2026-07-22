import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitBaseBranch } from '@/global'
import type * as ProjectsStore from '@/store/projects'
import { listBaseBranches } from '@/store/projects'

import { BaseBranchPicker } from './base-branch-picker'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        projects: {
          baseBranchNone: 'No base branches',
          baseBranchPlaceholder: 'Find a base branch',
          branchOff: () => ({ after: '', before: 'Branch off ' })
        }
      }
    }
  })
}))

vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal<typeof ProjectsStore>()

  return { ...actual, listBaseBranches: vi.fn() }
})

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

describe('BaseBranchPicker', () => {
  it('drops an old repository result and loads the new repository base', async () => {
    let resolveBranchesA: (branches: HermesGitBaseBranch[]) => void = () => undefined

    const branchesA = new Promise<HermesGitBaseBranch[]>(resolve => {
      resolveBranchesA = resolve
    })

    const onValueChange = vi.fn()

    vi.mocked(listBaseBranches).mockImplementation(repoPath =>
      repoPath === '/repo-a'
        ? branchesA
        : Promise.resolve([{ isDefault: true, isRemote: false, name: 'main-b' }])
    )

    const view = render(<BaseBranchPicker onValueChange={onValueChange} repoPath="/repo-a" value="" />)
    expect(listBaseBranches).toHaveBeenCalledWith('/repo-a')

    view.rerender(<BaseBranchPicker onValueChange={onValueChange} repoPath="/repo-b" value="" />)
    resolveBranchesA([{ isDefault: true, isRemote: false, name: 'main-a' }])

    await waitFor(() => expect(listBaseBranches).toHaveBeenCalledWith('/repo-b'))
    await waitFor(() => expect(onValueChange).toHaveBeenLastCalledWith('main-b'))
    expect(onValueChange).not.toHaveBeenCalledWith('main-a')
  })
})
