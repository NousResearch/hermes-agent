import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarWorkspaceGroup } from './workspace-group'
import type { SidebarSessionGroup } from './workspace-groups'

const mockSwitchBranchInRepo = vi.fn()

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        noSessions: 'No sessions yet',
        projects: {
          menu: 'Project actions'
        }
      },
      statusStack: {
        coding: {
          switchFailed: (branch: string) => `Could not switch to ${branch}`
        }
      }
    }
  })
}))

vi.mock('@/store/projects', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/store/projects')>()

  return {
    ...actual,
    switchBranchInRepo: (...args: unknown[]) => mockSwitchBranchInRepo(...args)
  }
})

vi.mock('@/store/profile', () => ({
  newSessionInProfile: vi.fn()
}))

vi.mock('./model', () => ({
  SIDEBAR_GROUP_PAGE: 10,
  useWorkspaceNodeOpen: () => [true, vi.fn()]
}))

describe('SidebarWorkspaceGroup', () => {
  it('calls onNewSession even if switchBranchInRepo fails (best-effort branch switch)', async () => {
    mockSwitchBranchInRepo.mockRejectedValue(new Error('fatal: not a git repository'))
    const onNewSession = vi.fn()

    const group: SidebarSessionGroup = {
      id: '/repo::branch::main',
      isMain: true,
      label: 'main',
      path: '/repo',
      sessions: []
    }

    render(
      <SidebarWorkspaceGroup
        group={group}
        onNewSession={onNewSession}
        renderRows={() => null}
      />
    )

    const addButton = screen.getByRole('button', { name: 'New session in main' })
    fireEvent.click(addButton)

    await waitFor(() => {
      expect(mockSwitchBranchInRepo).toHaveBeenCalledWith('/repo', 'main')
      expect(onNewSession).toHaveBeenCalledWith('/repo')
    })
  })

  it('calls onNewSession when switchBranchInRepo succeeds', async () => {
    mockSwitchBranchInRepo.mockResolvedValue(undefined)
    const onNewSession = vi.fn()

    const group: SidebarSessionGroup = {
      id: '/repo::branch::main',
      isMain: true,
      label: 'main',
      path: '/repo',
      sessions: []
    }

    render(
      <SidebarWorkspaceGroup
        group={group}
        onNewSession={onNewSession}
        renderRows={() => null}
      />
    )

    const addButton = screen.getByRole('button', { name: 'New session in main' })
    fireEvent.click(addButton)

    await waitFor(() => {
      expect(mockSwitchBranchInRepo).toHaveBeenCalledWith('/repo', 'main')
      expect(onNewSession).toHaveBeenCalledWith('/repo')
    })
  })
})
