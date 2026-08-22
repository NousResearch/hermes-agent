import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { EnteredProjectContent } from './entered-content'
import type { SidebarProjectTree } from './workspace-groups'

const startWorkInRepo = vi.fn()
const openWorktreeDialog = vi.fn()
const requestStartWorkSession = vi.fn()

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken: {
    subscribe: vi.fn(() => vi.fn())
  },
  startWorkInRepo: (...args: unknown[]) => startWorkInRepo(...args),
  requestStartWorkSession: (...args: unknown[]) => requestStartWorkSession(...args)
}))

vi.mock('@/store/coding-status', () => ({
  openWorktreeDialog: (...args: unknown[]) => openWorktreeDialog(...args),
  registerRepoStatusCwd: () => undefined,
  repoStatusForCwd: () => vi.fn(),
  repoWorktreesForCwd: () => vi.fn()
}))

vi.mock('./workspace-header', () => ({
  StartWorkButton: ({ repoPath }: { repoPath: string }) => (
    <button onClick={() => openWorktreeDialog({ mode: 'create', repoPath })} type="button">
      Start work for {repoPath}
    </button>
  ),
  WorkspaceAddButton: ({ label, onClick }: { label: string; onClick: () => void }) => (
    <button onClick={onClick} type="button">
      {label}
    </button>
  ),
  WorkspaceHeader: ({ action, children }: { action?: React.ReactNode; children: React.ReactNode }) => (
    <div>
      {action}
      {children}
    </div>
  )
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  startWorkInRepo.mockReset()
  openWorktreeDialog.mockReset()
  requestStartWorkSession.mockReset()
})

describe('EnteredProjectContent', () => {
  it('starts work from the selected repo via the shared worktree dialog', async () => {
    const repoTwoPath = '/work/repo-two'

    const project: SidebarProjectTree = {
      id: 'p_multi_repo',
      label: 'Multi Repo Project',
      path: '/project/root',
      repos: [
        {
          id: 'r_one',
          label: 'Repo One',
          path: '/work/repo-one',
          groups: [],
          sessionCount: 0
        },
        {
          id: 'r_two',
          label: 'Repo Two',
          path: repoTwoPath,
          groups: [],
          sessionCount: 0
        }
      ],
      sessionCount: 0
    }

    render(
      <I18nProvider configClient={null}>
        <EnteredProjectContent onNewSession={vi.fn()} project={project} renderRows={() => null} />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Start work for /work/repo-two' }))

    await waitFor(() => {
      expect(openWorktreeDialog).toHaveBeenCalledWith({ mode: 'create', repoPath: repoTwoPath })
    })
  })
})
