import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const fixture = vi.hoisted(() => ({
  activeProfile: 'default',
  activeConnection: null as null | string,
  ensureProfile: vi.fn<(profile: string) => Promise<void>>(),
  ensureAgent: vi.fn<(connectionId: string, profile: string) => Promise<boolean>>(),
  requestAgent: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  $gateway: atom({ connectionState: 'open' }),
  activeGatewayConnectionId: () => fixture.activeConnection,
  activeGatewayProfileKey: () => fixture.activeProfile,
  ensureGatewayForProfile: fixture.ensureProfile,
  ensureGatewayForAgent: fixture.ensureAgent,
  requestGatewayForAgent: fixture.requestAgent,
  retainGatewayForSessionTurn: vi.fn(async () => vi.fn())
}))
vi.mock('@/store/notifications', async original => ({
  ...(await original<Record<string, unknown>>()),
  notifyError: fixture.notifyError
}))
vi.mock('@/lib/query-client', async original => ({
  ...(await original<Record<string, unknown>>()),
  invalidateProfileScopedQueries: vi.fn()
}))
vi.mock('@/store/starmap', async original => ({
  ...(await original<Record<string, unknown>>()),
  reloadOpenStarmapGraph: vi.fn(),
  resetStarmapGraph: vi.fn()
}))
vi.mock('@/components/onboarding/first-run/assembly', () => ({ useOnboardingChatActive: () => false }))

import { ChatSwitchIndicator, SWITCH_STALLED_MS } from '@/app/chat/chat-switch-indicator'
import { SendTargetHint, useSwitchSendTarget } from '@/app/chat/composer/send-target-hint'
import { useFreshSessionRequest } from '@/app/contrib/wiring-effects'
import { useSessionActions } from '@/app/session/hooks/use-session-actions'
import {
  createGatewaySession,
  desktopSessionCreateParams
} from '@/app/session/hooks/use-session-actions/session-create-request'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $freshDraftKey, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $defaultProfileRoute } from '@/store/default-profile'
import {
  $activeGatewayProfile,
  $freshSessionRequest,
  $gatewaySwapTarget,
  $newChatProfile,
  $newChatRoute,
  captureNewChatSource,
  ensureGatewayProfile,
  isLegacyNewChatProfile,
  newSessionInProfile,
  normalizeProfileKey,
  pinNewChatRoute,
  resolveNewChatOwnerRoute,
  selectProfile
} from '@/store/profile'
import { $profileSwitchFailure, $profileSwitchPending, invalidateProfileSwitch } from '@/store/profile-switch-state'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import {
  _resetSessionOwnerHintsForTests,
  applyConfiguredDefaultProjectDir,
  setActiveSessionId,
  setConnection,
  setSelectedStoredSessionId,
  setSessionOwnerHint,
  setSessions
} from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

const originalBridge = window.hermesDesktop
const created = { session_id: 'created-runtime', stored_session_id: 'created-stored', info: {} }

function connection(profile: string, connectionId?: string) {
  return {
    baseUrl: 'http://profile.invalid',
    connectionId,
    isFullscreen: false,
    logs: [],
    mode: 'remote' as const,
    nativeOverlayWidth: 0,
    profile,
    token: 'test',
    windowButtonPosition: null,
    wsUrl: 'ws://profile.invalid/ws'
  }
}

function mountActions() {
  const state = createClientSessionState()

  const deps = {
    activeSessionId: null,
    activeSessionIdRef: { current: null as string | null },
    busyRef: { current: false },
    creatingSessionRef: { current: false },
    ensureSessionState: () => state,
    getRouteToken: () => 'draft-route',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn(),
    requestGateway: vi.fn(async () => created as never),
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: { current: new Map() },
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: { current: null as string | null },
    sessionStateByRuntimeIdRef: { current: new Map() },
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => state
  }

  return renderHook(() => {
    const actions = useSessionActions(deps)
    useFreshSessionRequest(actions.startFreshSessionDraft)

    return actions
  })
}

// Both production consumers use this same destination hook.
function SwitchChrome() {
  const target = useSwitchSendTarget()
  const failure = useStore($profileSwitchFailure)

  return (
    <>
      <SendTargetHint />
      <ChatSwitchIndicator failure={failure} syncing={null} target={target} />
    </>
  )
}

async function sendDraft(request: Parameters<typeof createGatewaySession>[2]) {
  const route = resolveNewChatOwnerRoute()
  const profile = $newChatProfile.get() || normalizeProfileKey($activeGatewayProfile.get())
  const params = await desktopSessionCreateParams('', route, profile, isLegacyNewChatProfile(profile))
  await createGatewaySession(route, params, request)

  return { route, params }
}

beforeEach(() => {
  fixture.activeProfile = 'default'
  fixture.activeConnection = null
  fixture.ensureProfile.mockReset().mockImplementation(async profile => {
    fixture.activeProfile = profile
    fixture.activeConnection = null
  })
  fixture.ensureAgent.mockReset().mockImplementation(async (connectionId, profile) => {
    fixture.activeProfile = profile
    fixture.activeConnection = connectionId

    return true
  })
  fixture.requestAgent.mockReset().mockResolvedValue(created)
  fixture.notifyError.mockClear()
  _resetSessionOwnerHintsForTests()
  invalidateProfileSwitch()
  $gatewaySwapTarget.set(null)
  $defaultProfileRoute.set(null)
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  captureNewChatSource(null)
  $projectScope.set(ALL_PROJECTS)
  $activeGatewayProfile.set('default')
  setSessions([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setConnection(connection('default') as never)
  window.history.replaceState(null, '', '/')
  window.hermesDesktop = {
    getConnection: vi.fn(async (profile: string) => connection(profile)),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => connection(profile, connectionId)),
    getConnectionConfig: vi.fn(async () => ({ mode: 'remote' }))
  } as never
})

afterEach(async () => {
  vi.useRealTimers()
  await ensureGatewayProfile(null)
  cleanup()
  applyConfiguredDefaultProjectDir('')
  $defaultProfileRoute.set(null)
  invalidateProfileSwitch()
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  captureNewChatSource(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setConnection(null)
  setSessions([])
  window.hermesDesktop = originalBridge
})

describe('switch assurances follow the send owner', () => {
  it('names beta in both assurances while alpha holds activation, and creates on beta', async () => {
    const blocked = Promise.withResolvers<void>()
    fixture.ensureProfile.mockImplementationOnce(async profile => {
      await blocked.promise
      fixture.activeProfile = profile
    })
    const view = render(<SwitchChrome />)
    const request = vi.fn(async () => created as never)
    let sending: Promise<{ params: Record<string, unknown> }> | undefined

    try {
      act(() => selectProfile('alpha'))
      await vi.waitFor(() => expect(fixture.ensureProfile).toHaveBeenCalledWith('alpha'))
      vi.useFakeTimers()
      act(() => selectProfile('beta'))
      act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))
      expect($gatewaySwapTarget.get()).toBe('alpha')
      expect(view.container.querySelector('[data-slot="composer-send-target"]')?.textContent).toContain('beta')
      const strip = view.container.querySelector('[data-slot="chat-switch-indicator"]')
      expect(strip?.getAttribute('data-switch-state')).toBe('stalled')
      expect(strip?.textContent).toContain('beta')
      expect(strip?.textContent).not.toContain('alpha')
      sending = sendDraft(request)
      expect(request).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
      await act(async () => {
        blocked.resolve()
        await ensureGatewayProfile(null)
        await sending
      })
    }

    expect((await sending)?.params.profile).toBe('beta')
    expect(request).toHaveBeenCalledWith('session.create', expect.objectContaining({ profile: 'beta' }))
    await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
    expect($activeGatewayProfile.get()).toBe('beta')
  })

  it('uses an exact registry draft and an existing session owner instead of either switch target', async () => {
    const route = { connectionId: 'homelab', profile: 'gateway', targetProfile: 'writer' }
    pinNewChatRoute(route)
    $gatewaySwapTarget.set('alpha')
    $profileSwitchPending.set('beta')
    vi.useFakeTimers()
    const view = render(<SwitchChrome />)
    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))
    expect(view.container.querySelector('[data-slot="composer-send-target"]')?.textContent).toContain('writer')
    expect(view.container.querySelector('[data-slot="chat-switch-indicator"]')?.textContent).toContain('writer')
    vi.useRealTimers()
    const ambient = vi.fn(async () => created as never)
    await act(async () => { await sendDraft(ambient) })
    expect(fixture.requestAgent).toHaveBeenCalledWith(
      'homelab', 'gateway', 'session.create', expect.objectContaining({ profile: 'writer' }),
      undefined, undefined, { spawnPriority: 'foreground' }
    )
    expect(ambient).not.toHaveBeenCalled()

    const owner = { connectionId: 'other-host', profile: 'shared', targetProfile: 'ops' }
    setSessionOwnerHint('stored-existing', owner)
    vi.useFakeTimers()
    act(() => {
      setSelectedStoredSessionId('stored-existing')
      $gatewaySwapTarget.set('alpha')
    })
    expect(view.container.querySelector('[data-slot="composer-send-target"]')?.textContent).toContain('ops')
    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))
    expect(view.container.querySelector('[data-slot="chat-switch-indicator"]')?.textContent).toContain('ops')
    vi.useRealTimers()
    await requestForSessionProfile(knownOwnerForSession('stored-existing'), ambient, 'prompt.submit', { session_id: 'runtime', profile: 'shared' })
    expect(fixture.requestAgent).toHaveBeenLastCalledWith(
      'other-host', 'shared', 'prompt.submit', expect.objectContaining({ profile: 'ops' })
    )
  })

  it('serializes a later pick of the already-active profile behind the slow outgoing activation', async () => {
    const blocked = Promise.withResolvers<void>()
    fixture.ensureProfile.mockImplementationOnce(async profile => {
      await blocked.promise
      fixture.activeProfile = profile
    })

    try {
      selectProfile('alpha')
      await vi.waitFor(() => expect(fixture.ensureProfile).toHaveBeenCalledWith('alpha'))
      selectProfile('default')
    } finally {
      blocked.resolve()
      await ensureGatewayProfile(null)
    }

    await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
    expect($activeGatewayProfile.get()).toBe('default')
    expect(fixture.activeProfile).toBe('default')
  })
})

describe('retry belongs to its fresh-draft intent', () => {
  it('invalidates a failed ops retry when generic New Session selects already-active legacy personal', async () => {
    fixture.activeProfile = 'personal'
    $activeGatewayProfile.set('personal')
    setConnection(connection('personal') as never)
    $defaultProfileRoute.set({ connectionId: null, profile: 'personal' })
    const actions = mountActions()
    render(<SwitchChrome />)
    fixture.ensureProfile.mockRejectedValueOnce(new Error('ops unavailable'))
    act(() => selectProfile('ops'))
    await screen.findByRole('button', { name: 'Retry' })
    const staleRetry = $profileSwitchFailure.get()!.retry
    const calls = fixture.ensureProfile.mock.calls.length

    act(() => actions.result.current.selectSidebarItem({ action: 'new-session' } as never))
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect($newChatProfile.get()).toBe('personal')
    expect(isLegacyNewChatProfile('personal')).toBe(true)
    act(staleRetry)
    expect(fixture.ensureProfile).toHaveBeenCalledTimes(calls)
    expect($profileSwitchFailure.get()).toBeNull()
    const request = vi.fn(async () => created as never)
    await act(async () => { await sendDraft(request) })
    expect(request).toHaveBeenCalledWith('session.create', expect.objectContaining({ profile: 'personal' }))
    expect($activeGatewayProfile.get()).toBe('personal')
  })

  it.each(['same-profile draft', 'source change', 'stored conversation'] as const)('revokes a saved retry after a %s', async replacement => {
    const actions = mountActions()
    fixture.ensureProfile.mockRejectedValueOnce(new Error('ops unavailable'))
    act(() => selectProfile('ops'))
    await vi.waitFor(() => expect($profileSwitchFailure.get()).not.toBeNull())
    const retry = $profileSwitchFailure.get()!.retry
    const calls = fixture.ensureProfile.mock.calls.length

    act(() => {
      if (replacement === 'same-profile draft') {
        actions.result.current.startFreshSessionDraft()
      } else if (replacement === 'source change') {
        pinNewChatRoute({ connectionId: 'new-host', profile: 'ops' })
      } else {
        setSelectedStoredSessionId('stored-existing')
      }

      retry()
    })
    expect($profileSwitchFailure.get()).toBeNull()
    expect(fixture.ensureProfile).toHaveBeenCalledTimes(calls)
    expect(fixture.ensureAgent).not.toHaveBeenCalled()
  })

  it('keeps a legitimate retry on the captured registry source after passive gateway activation', async () => {
    fixture.activeConnection = 'homelab'
    fixture.ensureAgent.mockRejectedValueOnce(new Error('ops unavailable'))
    act(() => selectProfile('ops'))
    await vi.waitFor(() => expect($profileSwitchFailure.get()).not.toBeNull())
    fixture.activeConnection = 'other-host'
    act(() => $profileSwitchFailure.get()!.retry())
    await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
    expect(fixture.ensureAgent).toHaveBeenLastCalledWith('homelab', 'ops')
    expect(fixture.activeConnection).toBe('homelab')
  })

  it.each(['same-profile draft', 'source change', 'direct fresh draft'] as const)('does not resurrect a late failure after a %s', async replacement => {
    const blocked = Promise.withResolvers<void>()
    fixture.ensureProfile.mockReturnValueOnce(blocked.promise)
    const actions = mountActions()
    render(<SwitchChrome />)

    try {
      act(() => selectProfile('ops'))
      await vi.waitFor(() => expect(fixture.ensureProfile).toHaveBeenCalledWith('ops'))
      act(() => {
        if (replacement === 'same-profile draft') {
          actions.result.current.selectSidebarItem({ action: 'new-session' } as never)
        } else if (replacement === 'direct fresh draft') {
          actions.result.current.startFreshSessionDraft()
        } else {
          pinNewChatRoute({ connectionId: 'new-host', profile: 'ops' })
        }
      })
    } finally {
      await act(async () => {
        blocked.reject(new Error('late ops failure'))
        await ensureGatewayProfile(null)
      })
    }

    expect($profileSwitchFailure.get()).toBeNull()
    expect($profileSwitchPending.get()).toBeNull()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('retries the same intent with a fresh grace period, without losing words or requesting another draft', async () => {
    fixture.ensureProfile.mockRejectedValueOnce(new Error('temporarily unavailable'))
    mountActions()
    const view = render(<SwitchChrome />)
    act(() => newSessionInProfile('ops'))
    await screen.findByRole('button', { name: 'Retry' })
    const drafts = $freshSessionRequest.get()
    const key = $freshDraftKey.get()
    stashSessionDraft(key, 'unsent words', [])
    const blocked = Promise.withResolvers<void>()
    fixture.ensureProfile.mockImplementationOnce(async profile => {
      await blocked.promise
      fixture.activeProfile = profile
    })

    try {
      vi.useFakeTimers()
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
      const strip = view.container.querySelector('[data-slot="chat-switch-indicator"]')
      expect(strip?.getAttribute('data-switch-state')).toBe('pending')
      expect(strip?.textContent).toBe('')
      act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))
      expect(strip?.getAttribute('data-switch-state')).toBe('stalled')
    } finally {
      vi.useRealTimers()
      await act(async () => {
        blocked.resolve()
        await ensureGatewayProfile(null)
      })
    }

    await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
    expect($profileSwitchFailure.get()).toBeNull()
    expect($activeGatewayProfile.get()).toBe('ops')
    expect($freshSessionRequest.get()).toBe(drafts)
    expect($freshDraftKey.get()).toBe(key)
    expect(takeSessionDraft(key).text).toBe('unsent words')
  })
})
