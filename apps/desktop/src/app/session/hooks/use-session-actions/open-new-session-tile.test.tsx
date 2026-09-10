// Regression: openNewSessionTile in Bot Mode was setting the persistent
// hidden: true flag, making user-initiated side sessions permanently
// invisible in the Sessions list.
//
// This test verifies the fix: openNewSessionTile must NOT pass hidden: true
// in the session.create params, regardless of workspaceMode.

import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSession } from '@/hermes'
import { ensureGatewayProfile } from '@/store/profile'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import {
  requestGateway,
  requestGatewayForAgent,
} from '@/store/gateway'
import {
  $activeSessionStoredIdRotation,
  $messages,
  $sessions,
  sessionPinId,
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'
import type { ClientSessionState } from '@/store/session'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn(),
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  openGatewayForAgent: vi.fn(),
  openGatewayForProfile: vi.fn(),
  requestGateway: vi.fn().mockResolvedValue({ ok: true, stored_session_id: 'stored-1' }),
  requestGatewayForAgent: vi.fn().mockResolvedValue({ ok: true, stored_session_id: 'stored-1' }),
  retainGatewayForAgent: vi.fn().mockResolvedValue(() => undefined),
}))

type SessionCreateParams = Record<string, unknown>

function extractSessionCreateParams(): SessionCreateParams | undefined {
  const calls = vi.mocked(requestGatewayForAgent).mock.calls
  const createCall = calls.find(call => call[2] === 'session.create')
  return createCall?.[3] as SessionCreateParams | undefined
}

type Handle = Pick<ReturnType<typeof useSessionActions>, 'openNewSessionTile'>

interface HarnessProps {
  onReady: (handle: Handle) => void
}

function Harness({ onReady }: HarnessProps) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: vi.fn().mockResolvedValue(undefined),
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState,
  })

  useEffect(() => {
    onReady({ openNewSessionTile: actions.openNewSessionTile })
  }, [actions, onReady])

  return null
}

async function mountHarness(): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={h => (handle = h)} />)
  await waitFor(() => expect(handle).toBeDefined())
  return handle as Handle
}

describe('openNewSessionTile × Bot Mode hidden flag', () => {
  beforeEach(() => {
    vi.mocked(requestGateway).mockReset()
    vi.mocked(requestGatewayForAgent).mockReset()
    vi.mocked(requestGateway).mockResolvedValue({ ok: true, stored_session_id: 'stored-1' })
    vi.mocked(requestGatewayForAgent).mockResolvedValue({ ok: true, stored_session_id: 'stored-1' })
    $activeGatewayProfile.set('default')
    $newChatProfile.set('')
  })

  afterEach(() => {
    cleanup()
    $activeGatewayProfile.set('default')
    $newChatProfile.set('')
  })

  it('does NOT pass hidden: true when workspaceMode is "bots"', async () => {
    const handle = await mountHarness()

    await act(() =>
      handle.openNewSessionTile('right', {
        workspaceScope: { workspaceMode: 'bots' },
      }),
    )

    const params = extractSessionCreateParams()
    expect(params).toBeDefined()
    expect(params?.hidden).toBeUndefined()
    expect(params?.hidden).not.toBe(true)
  })

  it('does NOT pass hidden: true in normal sessions mode', async () => {
    const handle = await mountHarness()

    await act(() =>
      handle.openNewSessionTile('right', {
        workspaceScope: { workspaceMode: 'sessions' },
      }),
    )

    const params = extractSessionCreateParams()
    expect(params).toBeDefined()
    expect(params?.hidden).toBeUndefined()
  })

  it('does NOT pass hidden: true even when desktopSessionCreateParams could set it', async () => {
    const handle = await mountHarness()

    await act(() =>
      handle.openNewSessionTile('right', {
        workspaceScope: { workspaceMode: 'bots' },
      }),
    )

    const params = extractSessionCreateParams()
    expect(params).toBeDefined()

    // Verify the complete params object does not contain hidden
    const keys = Object.keys(params!)
    expect(keys).not.toContain('hidden')

    // Verify other expected params are still present
    expect(params?.source).toBe('desktop')
    expect(params?.cols).toBe(96)
  })
})
