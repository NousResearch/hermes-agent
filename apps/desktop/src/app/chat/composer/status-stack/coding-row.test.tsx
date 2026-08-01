import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

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
  repoWorktreesForCwd: () => atom([])
}))

const { CodingStatusRow } = await import('./coding-row')

describe('CodingStatusRow', () => {
  afterEach(() => {
    cleanup()
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

    fireEvent.click(screen.getByText('12').closest('button')!)
    expect(onOpen).toHaveBeenCalledTimes(2)
  })
})
