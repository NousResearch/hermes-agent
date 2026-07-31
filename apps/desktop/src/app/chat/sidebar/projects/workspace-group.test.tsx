import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as ProjectsStore from '@/store/projects'

import type * as Model from './model'
import { SidebarWorkspaceGroup } from './workspace-group'
import type { SidebarSessionGroup } from './workspace-groups'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        noSessions: 'No sessions yet',
        newSessionIn: (label: string) => `New session in ${label}`,
        projects: {
          copyPath: 'Copy path',
          menu: 'Actions',
          removeWorktree: 'Remove worktree',
          reveal: 'Reveal in file manager',
          startWork: 'New worktree'
        },
        showMoreIn: (n: number, label: string) => `Show ${n} more in ${label}`
      },
      statusStack: { coding: { switchFailed: (branch: string) => `Could not switch to ${branch}` } }
    }
  })
}))

vi.mock('@/store/layout', () => ({ setWorkspaceNodeOpen: vi.fn() }))
vi.mock('@/store/profile', () => ({ newSessionInProfile: vi.fn() }))
// Partial mocks: `@/store/projects` and `./model` pull in the coding-status
// store's subscriptions, so replacing either module wholesale breaks the import
// graph rather than the behavior under test.
vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectsStore>()),
  copyPath: vi.fn(),
  revealPath: vi.fn(),
  switchBranchInRepo: vi.fn(() => Promise.resolve())
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('./model', async importOriginal => ({
  ...(await importOriginal<typeof Model>()),
  useWorkspaceNodeOpen: () => [true, vi.fn()] as const
}))

const { notifyError } = await import('@/store/notifications')
const { switchBranchInRepo } = await import('@/store/projects')

const lane = (over: Partial<SidebarSessionGroup> & Pick<SidebarSessionGroup, 'id' | 'label'>): SidebarSessionGroup => ({
  path: null,
  sessions: [],
  ...over
})

const renderGroup = (group: SidebarSessionGroup, onNewSession = vi.fn()) => {
  render(<SidebarWorkspaceGroup group={group} onNewSession={onNewSession} renderRows={() => null} />)

  return onNewSession
}

const clickAdd = (label: string) => fireEvent.click(screen.getByRole('button', { name: `New session in ${label}` }))

describe('SidebarWorkspaceGroup "+"', () => {
  it('starts a session directly in a folder lane, without touching git', async () => {
    const onNewSession = renderGroup(
      lane({
        id: '/docs/notes::folder',
        label: 'notes',
        isFolder: true,
        isHome: true,
        isMain: true,
        path: '/docs/notes'
      })
    )

    clickAdd('notes')

    // A plain folder has no branch behind the label; switching would fail on
    // `git switch` and take the new session down with it.
    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/docs/notes'))
    expect(switchBranchInRepo).not.toHaveBeenCalled()
  })

  it('switches to the lane branch first in a real repo', async () => {
    const onNewSession = renderGroup(
      lane({ id: '/repo::branch::main', label: 'main', isHome: true, isMain: true, path: '/repo' })
    )

    clickAdd('main')

    await waitFor(() => expect(switchBranchInRepo).toHaveBeenCalledWith('/repo', 'main'))
    expect(onNewSession).toHaveBeenCalledWith('/repo')
  })

  it('blocks creation and reports when a real switch fails', async () => {
    vi.mocked(switchBranchInRepo).mockRejectedValueOnce(new Error('dirty tree'))

    const onNewSession = renderGroup(
      lane({ id: '/repo::branch::main', label: 'main', isHome: true, isMain: true, path: '/repo' })
    )

    clickAdd('main')

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Could not switch to main'))
    expect(onNewSession).not.toHaveBeenCalled()
  })
})
