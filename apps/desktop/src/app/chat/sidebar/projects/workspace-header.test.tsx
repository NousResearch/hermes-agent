import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { WorkspaceAddButton } from './workspace-header'

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

vi.mock('@/store/coding-status', () => ({
  openWorktreeDialog: vi.fn()
}))

describe('WorkspaceAddButton', () => {
  it('still fires onClick', () => {
    const onClick = vi.fn()
    render(<WorkspaceAddButton label="New session in Test D" onClick={onClick} />)

    fireEvent.click(screen.getByRole('button', { name: 'New session in Test D' }))
    expect(onClick).toHaveBeenCalledOnce()
  })
})
