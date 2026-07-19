import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { EnteredProjectContent } from './entered-content'
import type { SidebarProjectTree } from './workspace-groups'

const { startWorkInRepo } = vi.hoisted(() => ({
  startWorkInRepo: vi.fn()
}))

vi.mock('./worktree-dialog', async () => {
  const React = await import('react')

  return {
    WorktreeDialog: ({
      open,
      onOpenChange,
      onStarted,
      repoPath
    }: {
      open: boolean
      onOpenChange: (next: boolean) => void
      onStarted: (path: string) => void
      repoPath: string
    }) => {
      React.useEffect(() => {
        if (!open) {
          return
        }
        void (async () => {
          const result = await startWorkInRepo(repoPath, { branch: 'feature-start', name: 'feature-start' })

          if (result && typeof result === 'object' && 'path' in result) {
            onStarted(result.path as string)
          }

          onOpenChange(false)
        })()
      }, [open, onOpenChange, onStarted, repoPath])

      return null
    }
  }
})

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken: {
    subscribe: vi.fn(() => vi.fn())
  },
  startWorkInRepo: (...args: unknown[]) => startWorkInRepo(...args)
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  startWorkInRepo.mockReset()
})

describe('EnteredProjectContent', () => {
  it('starts work from the selected repo and opens the returned worktree path', async () => {
    const createdWorktreePath = '/work/repo-two/.worktrees/feature-start'
    const repoOnePath = '/work/repo-one'
    const repoTwoPath = '/work/repo-two'

    const project: SidebarProjectTree = {
      id: 'p_multi_repo',
      label: 'Multi Repo Project',
      path: '/project/root',
      repos: [
        {
          id: 'r_one',
          label: 'Repo One',
          path: repoOnePath,
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

    const onNewSession = vi.fn()

    startWorkInRepo.mockResolvedValue({ branch: 'feature-start', path: createdWorktreePath })

    render(
      <I18nProvider configClient={null}>
        <EnteredProjectContent onNewSession={onNewSession} project={project} renderRows={() => null} />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'New worktree: Repo Two' }))

    await waitFor(() => expect(startWorkInRepo).toHaveBeenCalledTimes(1))
    expect(startWorkInRepo).toHaveBeenCalledWith(
      repoTwoPath,
      expect.objectContaining({ branch: 'feature-start', name: 'feature-start' })
    )
    expect(onNewSession).toHaveBeenCalledWith(createdWorktreePath)
  })
})
