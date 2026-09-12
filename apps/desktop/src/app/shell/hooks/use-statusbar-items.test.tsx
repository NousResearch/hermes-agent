import { cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $currentCwd, setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'

import { useStatusbarItems } from './use-statusbar-items'

vi.mock('@/app/shell/approval-mode-menu', () => ({
  useApprovalModeStatusbarItem: () => ({ id: 'approval-mode', label: 'Approvals', variant: 'menu' })
}))
vi.mock('@/app/shell/hooks/use-context-breakdown', () => ({
  useContextBreakdown: () => ({ breakdown: null, loading: false })
}))

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

afterEach(() => {
  cleanup()
  $currentCwd.set('')
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
})

describe('useStatusbarItems workspace ownership', () => {
  it('does not paint the previous workspace path while a fresh draft is resolving', () => {
    $currentCwd.set('/repo/previous-worktree')

    const requestGateway = vi.fn(async () => ({})) as unknown as <T = unknown>(
      method: string,
      params?: Record<string, unknown>
    ) => Promise<T>

    const { result } = renderHook(
      () =>
        useStatusbarItems({
          agentsOpen: false,
          chatOpen: true,
          commandCenterOpen: false,
          extraLeftItems: [],
          extraRightItems: [],
          freshDraftReady: true,
          gatewayState: 'open',
          inferenceStatus: null,
          openAgents: vi.fn(),
          openCommandCenterSection: vi.fn(),
          requestGateway,
          statusSnapshot: null,
          toggleCommandCenter: vi.fn()
        }),
      { wrapper }
    )

    expect(result.current.leftStatusbarItems.find(item => item.id === 'workspace-cwd')?.hidden).toBe(true)
  })
})
