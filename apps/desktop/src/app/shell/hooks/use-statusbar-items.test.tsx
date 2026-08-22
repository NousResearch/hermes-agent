import { cleanup, renderHook } from '@testing-library/react'
import { type ReactElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId, $sessions, $sessionStartedAt } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useStatusbarItems } from './use-statusbar-items'

vi.mock('@/app/shell/approval-mode-menu', () => ({
  useApprovalModeStatusbarItem: () => ({ id: 'approval-mode', label: 'Approvals', variant: 'menu' })
}))

async function requestGateway<T = unknown>(_method: string, _params?: Record<string, unknown>): Promise<T> {
  return {} as T
}

const options = () => ({
  agentsOpen: false,
  chatOpen: true,
  commandCenterOpen: false,
  extraLeftItems: [],
  extraRightItems: [],
  freshDraftReady: false,
  gatewayState: 'open',
  inferenceStatus: null,
  openAgents: vi.fn(),
  openCommandCenterSection: vi.fn(),
  requestGateway,
  statusSnapshot: null,
  toggleCommandCenter: vi.fn()
})

const sessionRow = (id: string, startedAt: number, parentSessionId: string | null = null) =>
  ({
    id,
    parent_session_id: parentSessionId,
    started_at: startedAt
  }) as SessionInfo

afterEach(() => {
  cleanup()
  $activeTreeGroup.set(null)
  $layoutTree.set(null)
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $sessionStartedAt.set(null)
  $sessionStates.set({})
  $sessionTiles.set([])
})

describe('useStatusbarItems session timer', () => {
  it("anchors a focused branch tile to its runtime cache instead of the parent's stored age", () => {
    const parentRowStartedAt = 1_600_000_000
    const branchRuntimeStartedAt = 1_800_000_000_000

    $activeSessionId.set('parent-runtime')
    $selectedStoredSessionId.set('parent-stored')
    $sessionStartedAt.set(1_700_000_000_000)
    $sessions.set([
      sessionRow('parent-stored', parentRowStartedAt),
      sessionRow('branch-stored', parentRowStartedAt, 'parent-stored')
    ])
    $sessionTiles.set([{ runtimeId: 'branch-runtime', storedSessionId: 'branch-stored' }])
    $sessionStates.set({
      'branch-runtime': {
        ...createClientSessionState('branch-stored'),
        runtimeStartedAt: branchRuntimeStartedAt
      }
    })
    $layoutTree.set(
      group(['workspace', 'session-tile:branch-stored'], {
        active: 'session-tile:branch-stored',
        id: 'main'
      })
    )
    $activeTreeGroup.set('main')

    const { result } = renderHook(() => useStatusbarItems(options()))
    const sessionTimer = result.current.statusbarItems.find(item => item.id === 'session-timer')
    const duration = sessionTimer?.detail as ReactElement<{ since: number | null }> | undefined

    expect(duration?.props.since).toBe(branchRuntimeStartedAt)
    expect(duration?.props.since).not.toBe(parentRowStartedAt * 1000)
  })
})
