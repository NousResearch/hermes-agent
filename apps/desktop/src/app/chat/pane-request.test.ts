import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $paneStates } from '@/store/panes'
import { setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $activePaneId, $splitFreshDraft, $splitPaneSession, SPLIT_PANE_ID } from '@/store/split'

import { createMainPaneRequest, createPaneRequest } from './pane-request'
import { MAIN_PANE_VIEW } from './pane-view'

// The main pane's selection, as the wrapper reads it: the UNMIRRORED cache
// ref (while the split is focused the $selectedStoredSessionId singleton
// mirrors the SPLIT's session). The atom is set alongside to mirror how
// use-session-actions writes both in lockstep.
function setMainSelection(storedId: null | string) {
  MAIN_PANE_VIEW.selectedStoredSessionIdRef.current = storedId
  setSelectedStoredSessionId(storedId)
}

// Profile pinning (design §4): the split's request wrapper closes the
// Enter-beats-swap race when active and pins to the profile's own socket when
// inactive; the main pane's wrapper is a pure pass-through until the split
// opens.

const ensureGatewayProfile = vi.fn<(profile: string | null | undefined) => Promise<void>>(async () => undefined)

const ensureProfileSocketOpen = vi.fn<
  (profile: string | null | undefined) => Promise<null | { request: typeof socketRequest }>
>(async () => ({ request: socketRequest }))

const socketRequest = vi.fn(async () => 'socket-result' as never)

vi.mock('@/store/profile', () => ({
  ensureGatewayProfile: (profile: string | null | undefined) => ensureGatewayProfile(profile)
}))

vi.mock('@/store/gateway', () => ({
  ensureProfileSocketOpen: (profile: string | null | undefined) => ensureProfileSocketOpen(profile)
}))

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

afterEach(() => {
  ensureGatewayProfile.mockClear()
  ensureProfileSocketOpen.mockClear()
  socketRequest.mockClear()
  $paneStates.set({})
  $splitPaneSession.set(null)
  $splitFreshDraft.set(false)
  $activePaneId.set('main')
  setMainSelection(null)
  setSessions([])
})

describe('createPaneRequest (split pane)', () => {
  it('active pane: awaits the in-flight profile swap BEFORE the shared request goes out', async () => {
    let releaseSwap!: () => void
    ensureGatewayProfile.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          releaseSwap = resolve
        })
    )

    const order: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      order.push(`request:${method}`)

      return 'active-result' as never
    })

    const paneRequest = createPaneRequest({
      getProfile: () => 'coder',
      isActive: () => true,
      requestGateway
    })

    const pending = paneRequest('prompt.submit', { text: 'hi' })
    await Promise.resolve()

    // The swap hasn't settled — the submit must still be parked.
    expect(ensureGatewayProfile).toHaveBeenCalledWith('coder')
    expect(requestGateway).not.toHaveBeenCalled()

    order.push('swap:settled')
    releaseSwap()

    await expect(pending).resolves.toBe('active-result')
    expect(order).toEqual(['swap:settled', 'request:prompt.submit'])
    expect(ensureProfileSocketOpen).not.toHaveBeenCalled()
  })

  it("inactive pane: pins to the profile's own socket and never touches the active gateway", async () => {
    const requestGateway = vi.fn(async () => 'active-result' as never)

    const paneRequest = createPaneRequest({
      getProfile: () => 'coder',
      isActive: () => false,
      requestGateway
    })

    await expect(paneRequest('session.resume', { session_id: 's-1' }, 5_000)).resolves.toBe('socket-result')

    expect(ensureProfileSocketOpen).toHaveBeenCalledWith('coder')
    expect(socketRequest).toHaveBeenCalledWith('session.resume', { session_id: 's-1' }, 5_000, undefined)
    expect(requestGateway).not.toHaveBeenCalled()
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it('inactive pane: surfaces a gateway-unavailable error when no socket can be opened', async () => {
    ensureProfileSocketOpen.mockResolvedValueOnce(null)

    const paneRequest = createPaneRequest({
      getProfile: () => 'coder',
      isActive: () => false,
      requestGateway: vi.fn(async () => 'unused' as never)
    })

    await expect(paneRequest('session.resume')).rejects.toThrow('Hermes gateway unavailable')
  })
})

describe('createMainPaneRequest', () => {
  it('split closed: pure pass-through — no profile machinery at all', async () => {
    const requestGateway = vi.fn(async () => 'main-result' as never)
    const mainRequest = createMainPaneRequest({ requestGateway })

    await expect(mainRequest('prompt.submit', { text: 'hi' })).resolves.toBe('main-result')

    expect(requestGateway).toHaveBeenCalledWith('prompt.submit', { text: 'hi' }, undefined, undefined)
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it("split open, main pane ACTIVE: settles the swap onto the main session's profile before sending", async () => {
    $paneStates.set({ [SPLIT_PANE_ID]: { open: true } })
    $splitPaneSession.set({ profile: 'coder', storedId: 'stored-split' })
    setMainSelection('stored-main')
    setSessions([storedSession({ id: 'stored-main', profile: 'analyst' })])

    const requestGateway = vi.fn(async () => 'main-result' as never)
    const mainRequest = createMainPaneRequest({ requestGateway })

    await expect(mainRequest('prompt.submit')).resolves.toBe('main-result')

    expect(ensureGatewayProfile).toHaveBeenCalledWith('analyst')
    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(ensureProfileSocketOpen).not.toHaveBeenCalled()
  })

  it('split open with a fresh main draft: settles any in-flight swap with a null profile', async () => {
    $paneStates.set({ [SPLIT_PANE_ID]: { open: true } })
    $splitPaneSession.set({ profile: 'coder', storedId: 'stored-split' })
    setMainSelection(null)

    const requestGateway = vi.fn(async () => 'main-result' as never)
    const mainRequest = createMainPaneRequest({ requestGateway })

    await mainRequest('prompt.submit')

    expect(ensureGatewayProfile).toHaveBeenCalledWith(null)
  })

  it("split FOCUSED (main pane background): pins to the main session's own profile socket", async () => {
    // The background main-pane dispatchers (reconnect route-resume, bounded
    // auto-retry, queue auto-drain) fire while the split is focused. The
    // active gateway belongs to the split then, so their RPCs must ride the
    // main session's pinned socket — and the profile must come from the
    // UNMIRRORED selection ref, not the mirrored singleton.
    $paneStates.set({ [SPLIT_PANE_ID]: { open: true } })
    $splitPaneSession.set({ profile: 'coder', storedId: 'stored-split' })
    $activePaneId.set('split')
    MAIN_PANE_VIEW.selectedStoredSessionIdRef.current = 'stored-main'
    // The mirrored singleton points at the SPLIT's session — it must be ignored.
    setSelectedStoredSessionId('stored-split')
    setSessions([
      storedSession({ id: 'stored-main', profile: 'analyst' }),
      storedSession({ id: 'stored-split', profile: 'coder' })
    ])

    const requestGateway = vi.fn(async () => 'active-result' as never)
    const mainRequest = createMainPaneRequest({ requestGateway })

    await expect(mainRequest('session.resume', { session_id: 'stored-main' })).resolves.toBe('socket-result')

    expect(ensureProfileSocketOpen).toHaveBeenCalledWith('analyst')
    expect(socketRequest).toHaveBeenCalledWith('session.resume', { session_id: 'stored-main' }, undefined, undefined)
    expect(requestGateway).not.toHaveBeenCalled()
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it('split FOCUSED with a profile-less main pane: stays on the shared path', async () => {
    $paneStates.set({ [SPLIT_PANE_ID]: { open: true } })
    $splitPaneSession.set({ profile: 'coder', storedId: 'stored-split' })
    $activePaneId.set('split')
    MAIN_PANE_VIEW.selectedStoredSessionIdRef.current = null

    const requestGateway = vi.fn(async () => 'active-result' as never)
    const mainRequest = createMainPaneRequest({ requestGateway })

    await expect(mainRequest('config.get')).resolves.toBe('active-result')

    expect(ensureProfileSocketOpen).not.toHaveBeenCalled()
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
  })
})
