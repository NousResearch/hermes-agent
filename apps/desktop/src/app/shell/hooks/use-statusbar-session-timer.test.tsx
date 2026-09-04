import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import {
  setSelectedStoredSessionId,
  setSessions,
  setSessionStartedAt
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { useStatusbarItems } from './use-statusbar-items'

vi.mock('@/app/chat/sidebar/connection-switcher', () => ({ ConnectionSwitcher: () => null }))
vi.mock('@/app/shell/approval-mode-menu', () => ({
  useApprovalModeStatusbarItem: () => ({ id: 'approval-mode' })
}))
vi.mock('@/app/shell/system-resources-statusbar', () => ({
  useSystemResourcesStatusbarItem: () => ({ id: 'system-resources' })
}))

const requestGateway = vi.fn(async () => ({}))

const STARTED_AT = Date.now() / 1000 - 23 * 3600 - 3 * 60

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'sess-a',
    input_tokens: 0,
    is_active: true,
    is_default_profile: true,
    last_active: STARTED_AT,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'tui',
    started_at: STARTED_AT,
    title: null,
    tool_call_count: 0,
    cwd: '/repo',
    ...overrides
  }
}

/** The "Session" item's LiveDuration `since` value under the current focus. */
function sessionTimerSince(): number | null {
  const { result } = renderHook(() =>
    useStatusbarItems({
      agentsOpen: false,
      chatOpen: true,
      commandCenterOpen: false,
      extraLeftItems: [],
      extraRightItems: [],
      freshDraftReady: false,
      gatewayState: 'open',
      inferenceStatus: null,
      openAgents: () => undefined,
      openCommandCenterSection: () => undefined,
      requestGateway: requestGateway as unknown as Parameters<typeof useStatusbarItems>[0]['requestGateway'],
      statusSnapshot: null,
      toggleCommandCenter: () => undefined
    })
  )

  const item = result.current.statusbarItems.find(candidate => candidate.id === 'session-timer')
  const detail = item && 'detail' in item ? (item.detail as { props: { since: number | null } }) : null

  return detail ? detail.props.since : null
}

/** Move the "interacted zone" focus onto a session tile, as the layout tree does. */
function focusTile(storedId: string): void {
  const tree = group(['tile-pane', `session-tile:${storedId}`], { active: `session-tile:${storedId}` })

  act(() => {
    $layoutTree.set(tree)
    $activeTreeGroup.set(tree.id)
  })
}

beforeEach(() => {
  $activeTreeGroup.set(null)
  $layoutTree.set(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setSessionStartedAt(null)
})

afterEach(() => {
  cleanup()
  $activeTreeGroup.set(null)
  $layoutTree.set(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setSessionStartedAt(null)
})

describe('statusbar session timer semantics', () => {
  it('reads the same row age whether the primary pane or a tile has focus', () => {
    act(() => {
      setSessions([makeSession()])
      setSelectedStoredSessionId('sess-a')
      setSessionStartedAt(Date.now())
    })

    expect(sessionTimerSince()).toBe(STARTED_AT * 1000)

    focusTile('sess-a')

    expect(sessionTimerSince()).toBe(STARTED_AT * 1000)
  })

  it('falls back to the create-time stamp only while the row is missing', () => {
    act(() => {
      setSelectedStoredSessionId('sess-a')
      setSessionStartedAt(123_450_000)
    })

    expect(sessionTimerSince()).toBe(123_450_000)

    act(() => {
      setSessions([makeSession()])
    })

    expect(sessionTimerSince()).toBe(STARTED_AT * 1000)
  })

  it('hides the timer for a focused tile whose row cannot be resolved', () => {
    act(() => {
      setSessions([makeSession()])
      setSelectedStoredSessionId('sess-a')
      setSessionStartedAt(Date.now())
    })

    focusTile('sess-b')

    expect(sessionTimerSince()).toBeNull()
  })
})
