import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StartWorkButton, WorkspaceAddButton, WorkspaceMenu, WorkspaceShowMoreButton } from './workspace-header'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        projects: {
          copyPath: 'Copy path',
          menu: 'Actions',
          removeWorktree: 'Remove worktree',
          reveal: 'Reveal in file manager',
          startWork: 'New worktree'
        },
        showMoreIn: (n: number, label: string) => `Show ${n} more in ${label}`
      }
    }
  })
}))

vi.mock('@/store/projects', () => ({
  copyPath: vi.fn(),
  revealPath: vi.fn()
}))

// StartWorkButton no longer renders a dialog. It publishes an intent to the one
// WorktreeDialog that is mounted in the sidebar. Stub the store action, so this
// test keeps the button separate from the git probes of the resolver.
vi.mock('@/store/coding-status', () => ({
  openWorktreeDialog: vi.fn()
}))

const tipTrigger = (button: HTMLElement) => button.closest('[data-slot="tooltip-trigger"]')

describe('WorkspaceAddButton', () => {
  it('wraps the "+" button in a Tip', () => {
    render(<WorkspaceAddButton label="New session in Test D" onClick={vi.fn()} />)

    const button = screen.getByRole('button', { name: 'New session in Test D' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('still fires onClick', () => {
    const onClick = vi.fn()
    render(<WorkspaceAddButton label="New session in Test D" onClick={onClick} />)

    fireEvent.click(screen.getByRole('button', { name: 'New session in Test D' }))
    expect(onClick).toHaveBeenCalledOnce()
  })
})

describe('WorkspaceShowMoreButton', () => {
  it('wraps the ellipsis button in a Tip with the composed label', () => {
    render(<WorkspaceShowMoreButton count={5} label="Test D" onClick={vi.fn()} />)

    const button = screen.getByRole('button', { name: 'Show 5 more in Test D' })
    expect(tipTrigger(button)).toBeTruthy()
  })
})

describe('WorkspaceMenu', () => {
  it('does not wrap the kebab trigger in a Tip', () => {
    render(<WorkspaceMenu onRemove={vi.fn()} path="/repo/lane" />)

    const button = screen.getByRole('button', { name: 'Actions' })
    expect(tipTrigger(button)).toBeNull()
  })
})

describe('StartWorkButton', () => {
  it('wraps the git-branch trigger in a Tip', () => {
    render(<StartWorkButton repoPath="/repo" />)

    const button = screen.getByRole('button', { name: 'New worktree' })
    expect(tipTrigger(button)).toBeTruthy()
  })
})
