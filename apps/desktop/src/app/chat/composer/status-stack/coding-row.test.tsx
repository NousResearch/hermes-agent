import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $newWorktreeRequest } from '@/store/projects'

const worktreeDialog = vi.fn()

vi.mock('@/app/chat/sidebar/projects/worktree-dialog', () => ({
  WorktreeDialog: (props: { initialMode: string; open: boolean }) => {
    worktreeDialog(props)

    return <div data-testid="worktree-dialog" />
  }
}))

vi.mock('@/components/ui/actions-menu', () => ({
  ActionsContextMenu: ({ children }: { children: ReactNode }) => <>{children}</>,
  ActionsMenu: ({ children, items }: { children: ReactNode; items: (kit: unknown) => ReactNode }) => (
    <>
      {children}
      {items({
        Label: ({ children: label }: { children: ReactNode }) => <div>{label}</div>,
        Separator: () => <div />
      })}
    </>
  ),
  renderActionItem: (_kit: unknown, item: { key: string; label: ReactNode; onSelect: () => void }) => (
    <button key={item.key} onClick={item.onSelect} type="button">
      {item.label}
    </button>
  )
}))

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

    fireEvent.click(screen.getByText('12'))
    expect(onOpen).toHaveBeenCalledTimes(2)
  })

  it('wraps the click targets without adding a layout box', () => {
    const { container } = render(<CodingStatusRow onOpen={() => undefined} repoPath="/repo" />)

    // `display: contents` is what keeps the branch label and the counts direct
    // flex children of the row — the hit areas cost nothing visually.
    expect(screen.getByText('bb/hitbox').parentElement?.classList.contains('contents')).toBe(true)
    expect(screen.getByText('12').closest('button')?.classList.contains('contents')).toBe(true)
    // The glyph button fills the row's existing 3.5 leading slot exactly.
    expect(container.querySelector('button[class~="size-3.5"]')).not.toBeNull()
  })

  it('parks the copy glyph against the end of the path, not the end of the row', () => {
    render(<CodingStatusRow onOpen={() => undefined} repoPath="/Users/someone/www/repo" />)

    const path = screen.getByText('~/www/repo')

    // The path sizes to its content and the glyph is its immediate sibling, so
    // the pair reads as one unit. `flex-1` belongs to the wrapper (which holds
    // the row's slack open) — on the label it stretched the text and pushed the
    // glyph out to the kebab.
    expect(path.classList.contains('flex-1')).toBe(false)
    expect(path.parentElement?.classList.contains('flex-1')).toBe(true)
    expect(path.nextElementSibling?.tagName).toBe('BUTTON')
  })

  it('copies the absolute cwd inline — checkmark feedback, no toast', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    clearNotifications()

    render(<CodingStatusRow onOpen={() => undefined} repoPath="/Users/someone/www/repo" />)

    // Painted tildified, copied raw.
    expect(screen.getByText('~/www/repo')).toBeTruthy()

    const copy = screen.getByRole('button', { name: 'Copy Path' })

    fireEvent.click(copy)

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('/Users/someone/www/repo'))
    // Confirmation is the button turning into a checkmark, not a notification.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied' })).toBeTruthy())
    expect($notifications.get()).toHaveLength(0)
  })
})

function renderWorktreeRow() {
  worktreeDialog.mockClear()

  render(
    <CodingStatusRow
      onBranchOff={() => Promise.resolve()}
      onConvertBranch={() => Promise.resolve()}
      onOpen={() => undefined}
      onOpenWorktree={() => undefined}
      repoPath="/repo"
    />
  )
}

function latestDialogProps() {
  const props = worktreeDialog.mock.lastCall?.[0] as undefined | { initialMode: string; open: boolean }

  if (!props) {
    throw new Error('WorktreeDialog was not rendered')
  }

  return props
}

describe('CodingStatusRow worktree dialog mode', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    $newWorktreeRequest.set(0)
  })

  it('opens Start Work in create mode', async () => {
    renderWorktreeRow()

    fireEvent.click(screen.getByRole('button', { name: 'New worktree' }))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'create', open: true })
    })
  })

  it('opens Convert Branch in convert mode', async () => {
    renderWorktreeRow()

    fireEvent.click(screen.getByRole('button', { name: 'Convert a branch…' }))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'convert', open: true })
    })
  })

  it('opens the global new-worktree request in create mode', async () => {
    renderWorktreeRow()
    act(() => $newWorktreeRequest.set(1))

    await waitFor(() => {
      expect(latestDialogProps()).toMatchObject({ initialMode: 'create', open: true })
    })
  })
})
