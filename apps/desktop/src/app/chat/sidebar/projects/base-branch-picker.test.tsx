import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitBaseBranch } from '@/global'
import { $worktreeDialog } from '@/store/projects'

import { BaseBranchPicker } from './base-branch-picker'
import { WorktreeDialog } from './worktree-dialog'

type ActGlobal = typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }

const MAIN: HermesGitBaseBranch = { isDefault: true, isRemote: true, name: 'origin/main' }
const FEAT: HermesGitBaseBranch = { isDefault: false, isRemote: false, name: 'feat-x' }

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

// The bridge answers after an IPC round trip, as the Electron one does.
const listing = (answer: () => HermesGitBaseBranch[]) =>
  vi.fn(async (_repoPath: string) => {
    await sleep(5)

    return answer()
  })

const worktreeAddSpy = () =>
  vi.fn(async (_repoPath: string, options: { branch: string }) => ({
    branch: options.branch,
    path: `/repo/.worktrees/${options.branch}`
  }))

function installGit(git: Record<string, unknown>) {
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { git }
}

async function submitWorktree(name: string) {
  fireEvent.change(await screen.findByPlaceholderText('e.g. my-feature'), { target: { value: name } })
  fireEvent.click(screen.getByRole('button', { name: 'New worktree' }))
}

// act() holds every update until its scope exits, so a load loop gets one pass
// per act() and looks finite. Production renders on the real scheduler, and so
// do these tests.
beforeEach(() => {
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = false
})

afterEach(() => {
  cleanup()
  $worktreeDialog.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  ;(globalThis as ActGlobal).IS_REACT_ACT_ENVIRONMENT = true
})

describe('BaseBranchPicker loading', () => {
  it.each([
    ['an empty list (a folder git cannot list, an unborn HEAD)', () => []],
    [
      'a failed listing (a backend without the endpoint)',
      () => {
        throw new Error('404 Not Found')
      }
    ]
  ])('lists the repo once on %s', async (_case, answer: () => HermesGitBaseBranch[]) => {
    const baseBranchList = listing(answer)
    installGit({ baseBranchList })

    render(<BaseBranchPicker onValueChange={() => {}} repoPath="/folder" value="" />)
    await sleep(150)

    expect(baseBranchList).toHaveBeenCalledTimes(1)
  })

  it('a list from the previous repo does not set the base of the next', async () => {
    const land: Record<string, (list: HermesGitBaseBranch[]) => void> = {}

    const baseBranchList = vi.fn(
      (repoPath: string) =>
        new Promise<HermesGitBaseBranch[]>(resolve => {
          land[repoPath] = resolve
        })
    )

    installGit({ baseBranchList })
    const onValueChange = vi.fn()

    // The dialog remounts the picker per repo, keyed on the path.
    const view = render(<BaseBranchPicker key="/a" onValueChange={onValueChange} repoPath="/a" value="" />)
    await waitFor(() => expect(baseBranchList).toHaveBeenCalledWith('/a'))
    view.rerender(<BaseBranchPicker key="/b" onValueChange={onValueChange} repoPath="/b" value="" />)
    await waitFor(() => expect(baseBranchList).toHaveBeenCalledWith('/b'))

    land['/a']([{ ...MAIN, name: 'origin/main-a' }])
    land['/b']([{ ...MAIN, name: 'origin/main-b' }])

    await waitFor(() => expect(onValueChange).toHaveBeenCalledWith('origin/main-b'))
    expect(onValueChange).not.toHaveBeenCalledWith('origin/main-a')
  })
})

describe('WorktreeDialog base branch', () => {
  it('cuts the worktree from the base the caller chose', async () => {
    const baseBranchList = listing(() => [MAIN, FEAT])
    const worktreeAdd = worktreeAddSpy()
    installGit({ baseBranchList, worktreeAdd })

    render(<WorktreeDialog />)
    // What the coding row's "Branch off from feat-x" publishes.
    $worktreeDialog.set({ base: 'feat-x', repoPath: '/repo' })
    await waitFor(() => expect(baseBranchList).toHaveBeenCalled())
    await sleep(50)
    await submitWorktree('my-work')

    await waitFor(() => expect(worktreeAdd).toHaveBeenCalledTimes(1))
    expect(worktreeAdd.mock.calls[0][1]).toMatchObject({ base: 'feat-x' })
  })

  it('cuts the worktree from the default branch when the caller names no base', async () => {
    const baseBranchList = listing(() => [FEAT, MAIN])
    const worktreeAdd = worktreeAddSpy()
    installGit({ baseBranchList, worktreeAdd })

    render(<WorktreeDialog />)
    $worktreeDialog.set({ repoPath: '/repo' })
    await screen.findByText('origin/main')
    await submitWorktree('my-work')

    await waitFor(() => expect(worktreeAdd).toHaveBeenCalledTimes(1))
    expect(worktreeAdd.mock.calls[0][1]).toMatchObject({ base: 'origin/main' })
  })
})
