import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitWorktree } from '@/global'
import { $notifications, clearNotifications } from '@/store/notifications'

const mockCodingStatus = vi.hoisted(() => ({ worktrees: [] as HermesGitWorktree[] }))

vi.mock('@/store/coding-status', () => ({
  registerRepoStatusCwd: () => undefined,
  repoStatusForCwd: () =>
    atom({
      added: 12,
      ahead: 0,
      behind: 0,
      branch: 'bb/hitbox',
      defaultBranch: 'main',
      detached: false,
      removed: 3,
      untracked: 0
    }),
  repoWorktreesForCwd: () => atom(mockCodingStatus.worktrees)
}))

const { CodingStatusRow } = await import('./coding-row')

function openBranchMenu() {
  fireEvent.pointerDown(screen.getByRole('button', { name: 'New branch' }), {
    button: 0,
    ctrlKey: false,
    pointerType: 'mouse'
  })
}

describe('CodingStatusRow', () => {
  afterEach(() => {
    cleanup()
    mockCodingStatus.worktrees = []
  })

  it('opens the review pane from the branch and the diff counts, never the bar itself', () => {
    const onOpen = vi.fn()

    const { container } = render(<CodingStatusRow onOpen={onOpen} repoPath="/repo" />)

    const bar = container.querySelector<HTMLElement>('.coding-status-bar')

    expect(bar).not.toBeNull()

    fireEvent.click(bar!)
    expect(onOpen).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText('bb/hitbox'))
    expect(onOpen).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByText('12'))
    expect(onOpen).toHaveBeenCalledTimes(2)
  })

  it('copies the absolute cwd inline — checkmark feedback, no toast', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    clearNotifications()

    render(<CodingStatusRow onOpen={() => undefined} repoPath="/Users/someone/www/repo" />)

    // Painted tildified, copied raw.
    expect(screen.getByText('~/www/repo')).toBeTruthy()

    const copy = screen.getByRole('button', { name: 'Copy path' })

    fireEvent.click(copy)

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('/Users/someone/www/repo'))
    // Confirmation is the button turning into a checkmark, not a notification.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied' })).toBeTruthy())
    expect($notifications.get()).toHaveLength(0)
  })

  it('opens the default worktree instead of requesting an impossible branch switch', () => {
    mockCodingStatus.worktrees = [
      { branch: 'bb/hitbox', detached: false, isMain: false, locked: false, path: '/repo/.worktrees/hitbox' },
      { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' }
    ]
    const onOpenWorktree = vi.fn()
    const onSwitchBranch = vi.fn(async () => undefined)

    render(
      <CodingStatusRow
        onBranchOff={vi.fn(async () => undefined)}
        onOpenWorktree={onOpenWorktree}
        onSwitchBranch={onSwitchBranch}
        repoPath="/repo/.worktrees/hitbox"
      />
    )

    openBranchMenu()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Switch to main' }))

    expect(onOpenWorktree).toHaveBeenCalledWith('/repo')
    expect(onSwitchBranch).not.toHaveBeenCalled()
  })

  it('switches normally when no worktree owns the default branch', () => {
    const onOpenWorktree = vi.fn()
    const onSwitchBranch = vi.fn(async () => undefined)

    render(
      <CodingStatusRow
        onBranchOff={vi.fn(async () => undefined)}
        onOpenWorktree={onOpenWorktree}
        onSwitchBranch={onSwitchBranch}
        repoPath="/repo/.worktrees/hitbox"
      />
    )

    openBranchMenu()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Switch to main' }))

    expect(onSwitchBranch).toHaveBeenCalledWith('main')
    expect(onOpenWorktree).not.toHaveBeenCalled()
  })
})
