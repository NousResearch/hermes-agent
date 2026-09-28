/**
 * `host.openSession({ refreshInPlace })` — the BACKGROUND re-resume contract
 * (issue 121874).
 *
 * A `session.reclaimed` (or roster-activity) wake re-resolves the open Bot
 * Chat so the user's next send doesn't eat a stale runtime id. It used to go
 * through the full navigating open: `intent: 'in-place'` → the route flipped
 * to the chat, replacing whatever the user was reading (the Kanban board).
 * Background events must never navigate — "offer; don't hijack"
 * (apps/desktop/AGENTS.md).
 *
 * The refresh-only open may dial the owner backend and re-resume, but it must
 * not: call the core open (navigation / tile minting), publish the bots
 * workspace scope, or flip the all-profiles view. It refreshes through the
 * shared cache's `resumeTile(refreshTranscript)` for either a tile or the
 * primary chat. Queuing a route resume is not proof of a successful read.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

vi.mock('@/app/chat/session-view', async () => {
  const { atom } = await import('nanostores')

  return { PRIMARY_SESSION_VIEW: { $awaitingResponse: atom(false), $busy: atom(false) } }
})
vi.mock('@/app/open-session', () => ({ openSession: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', async () => {
  const { atom } = await import('nanostores')

  return { $narrowViewport: atom(false) }
})
vi.mock('@/contrib/events', () => ({ onGatewayEvent: vi.fn() }))
vi.mock('@/hermes', () => ({ deleteProfile: vi.fn(), getLogs: vi.fn(), getStatus: vi.fn(), hermesApi: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/system-actions', () => ({ runGatewayRestart: vi.fn() }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  type LineageRow = { _lineage_root_id?: null | string; id: string }

  return {
    $activeSessionId: atom(null),
    $connection: atom(null),
    $cronSessions: atom([]),
    $currentCwd: atom(''),
    $currentModel: atom(''),
    $gatewayState: atom('open'),
    $messages: atom([]),
    $messagingSessions: atom([]),
    $selectedStoredSessionId: atom(null),
    $sessions: atom([]),
    $unreadFinishedSessionIds: atom([]),
    lineageAliases: (storedId: string) => [storedId],
    rememberedSessionProfile: (_sessions: unknown, _sessionId: null | string, activeProfile: null | string) =>
      (activeProfile ?? '').trim() || 'default',
    requestSessionResume: vi.fn(),
    sessionMatchesStoredId: (session: LineageRow, storedSessionId: string) =>
      session.id === storedSessionId || session._lineage_root_id === storedSessionId,
    sessionPinId: (session: LineageRow) => session._lineage_root_id ?? session.id,
    setSessionOwnerHint: vi.fn(),
    setResumeExhaustedSessionId: vi.fn()
  }
})
vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return {
    $attentionSessionIds: atom([]),
    $draftSessionIds: atom([]),
    $focusedRuntimeId: atom(null),
    $focusedSessionState: atom(null),
    $focusedStoredSessionId: atom(null),
    $sessionTiles: atom([]),
    $sessionStates: atom({}),
    $stalledSessionIds: atom([]),
    $workingSessionIds: atom([]),
    dropTilesForProfile: vi.fn(),
    sessionTileDelegate: vi.fn(() => null)
  }
})
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  const profiles = atom([
    {
      has_env: false,
      is_default: false,
      model: null,
      name: 'cached-only',
      path: '/profiles/cached-only',
      provider: null,
      skill_count: 0
    }
  ])

  return {
    $activeGatewayProfile: atom('ops'),
    $gatewaySwapTarget: atom(null),
    $hydrationSyncProfile: atom(null),
    $profiles: profiles,
    $showAllProfiles: atom(false),
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn(),
    newSessionInAgent: vi.fn(),
    newSessionInProfile: vi.fn(),
    normalizeProfileKey: (value: null | string | undefined) => (value ?? '').trim() || 'default',
    refreshProfiles: vi.fn(async () => profiles.get()),
    selectProfile: vi.fn(),
    setActiveProfile: vi.fn(),
    setShowAllProfiles: vi.fn()
  }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayRoute: atom('default'),
    $gateway: atom(null),
    activeGateway: vi.fn(() => null),
    activeGatewayConnectionId: vi.fn(() => 'local'),
    ensureGatewayForAgent: vi.fn(),
    openGatewayForAgent: vi.fn(),
    openGatewayForProfile: vi.fn(),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => vi.fn()),
    retireLocalProfileGateways: vi.fn()
  }
})

const { host } = await import('./index')

const { openSession: openSessionCore } = await import('@/app/open-session')

const { openGatewayForProfile } = await import('@/store/gateway')

const { $activeGatewayProfile, setShowAllProfiles } = await import('@/store/profile')

const { $sessionTiles, sessionTileDelegate } = await import('@/store/session-states')

const { requestSessionResume, setSessionOwnerHint, $selectedStoredSessionId } = await import('@/store/session')

const { setWorkspaceScope, $workspaceMode } = await import('@/components/pane-shell/workspace-scope')

const setMockAtom = <T>(store: unknown, value: T) => (store as { set(next: T): void }).set(value)

const profile = (name: string): ProfileInfo => ({
  has_env: false,
  is_default: name === 'default',
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

afterEach(() => {
  vi.clearAllMocks()
  vi.mocked(sessionTileDelegate).mockReturnValue(null)
  setMockAtom($sessionTiles, [])
  setMockAtom($selectedStoredSessionId, null)
  $activeGatewayProfile.set('ops')
  setWorkspaceScope('sessions')
})

describe('host.openSession refreshInPlace — background re-resume never navigates (issue 121874)', () => {
  it('refreshes the open tile transcript without calling the core open', async () => {
    const resumeTile = vi.fn(async () => 'runtime-ops')

    vi.mocked(sessionTileDelegate).mockReturnValue({ resumeTile } as never)
    setMockAtom($sessionTiles, [{ storedSessionId: 'bot-chat-ops' }] as never)
    $activeGatewayProfile.set('ops')

    await host.openSession('bot-chat-ops', {
      profile: 'ops',
      refreshInPlace: true,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:ops'
    })

    // The one thing this wake exists for: the tile re-pulls its transcript.
    expect(resumeTile).toHaveBeenCalledWith('bot-chat-ops', { refreshTranscript: true })
    // And none of the navigating-open side effects.
    expect(openSessionCore).not.toHaveBeenCalled()
    expect($workspaceMode.get()).toBe('sessions')
    expect(setShowAllProfiles).not.toHaveBeenCalled()
  })

  it('awaits the main chat transcript refresh and propagates failure without navigating', async () => {
    const { $selectedStoredSessionId } = await import('@/store/session')
    const resumeTile = vi.fn().mockRejectedValueOnce(new Error('stored read failed'))
    vi.mocked(sessionTileDelegate).mockReturnValue({ resumeTile } as never)
    setMockAtom($selectedStoredSessionId, 'bot-chat-ops')

    await expect(host.openSession('bot-chat-ops', { profile: 'ops', refreshInPlace: true }))
      .rejects.toThrow('stored read failed')

    let complete!: (runtime: string) => void
    resumeTile.mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
    let settled = false

    const refresh = host.openSession('bot-chat-ops', { profile: 'ops', refreshInPlace: true })
      .then(() => { settled = true })

    await vi.waitFor(() => expect(resumeTile).toHaveBeenCalledTimes(2))
    expect(settled).toBe(false)
    complete('runtime-ops')
    await refresh
    expect(settled).toBe(true)
    expect(resumeTile).toHaveBeenLastCalledWith('bot-chat-ops', { refreshTranscript: true })
    expect(requestSessionResume).not.toHaveBeenCalled()
    expect(openSessionCore).not.toHaveBeenCalled()
  })

  it('resolves without re-opening anything when the session is not on screen', async () => {
    // No tile, route elsewhere: a background wake must not mint a tab, load
    // the chat into main, or dial the backend into a workspace switch.
    await host.openSession('bot-chat-ops', { profile: 'ops', refreshInPlace: true })

    expect(requestSessionResume).not.toHaveBeenCalled()
    expect(openSessionCore).not.toHaveBeenCalled()
    expect(openGatewayForProfile).not.toHaveBeenCalled()
  })

  it('stamps the owner hint so the resume routes to the owning backend', async () => {
    await host.openSession('bot-chat-ops', { profile: 'ops', refreshInPlace: true })

    expect(setSessionOwnerHint).toHaveBeenCalledWith('bot-chat-ops', expect.objectContaining({ profile: 'ops' }))
  })
})
