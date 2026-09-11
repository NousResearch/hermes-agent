import { JsonRpcGatewayError } from '@hermes/shared'
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  notify: vi.fn(),
  watch: vi.fn(),
  tile: vi.fn(),
  connectionId: 'source-a',
  ensure: vi.fn(async () => undefined)
}))

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => mocks.connectionId,
  requestGatewayForAgent: mocks.request,
  requestGatewayForProfile: (profile: string, ...args: unknown[]) => mocks.request('source-a', profile, ...args)
}))
vi.mock('@/components/onboarding-chat/assembly', async () => {
  const { atom } = await import('nanostores')

  return { $chatOnboardingThreadIds: atom([]), pickOnboardingGreeting: vi.fn(), startChatOnboardingSolo: vi.fn() }
})
vi.mock('@/components/onboarding-chat/first-build', async () => {
  const { atom } = await import('nanostores')

  return { $setupCheckIn: atom(null), watchFirstBuild: mocks.watch }
})
vi.mock('@/components/onboarding-chat/signpost', () => ({
  declinedLookAround: () => true,
  showProfileSignpost: vi.fn()
}))
vi.mock('@/components/pane-shell/tree/store', async () => ({
  $layoutTree: (await import('nanostores')).atom(null),
  activateTreePane: vi.fn()
}))
vi.mock('@/store/session-states', () => ({ patchSessionTile: mocks.tile }))
vi.mock('@/store/notifications', () => ({ notify: mocks.notify, dismissNotification: vi.fn() }))
vi.mock('@/store/machine', () => ({ loadMachineProfile: vi.fn(), machineDescription: () => '' }))
vi.mock('@/store/onboarding-script', () => ({
  PLAIN_SPEECH: '',
  buildChatOnboardingSeedMessages: vi.fn(() => [])
}))
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  const activeGatewayProfile = atom('hermes-setup')

  return {
    $activeGatewayProfile: activeGatewayProfile,
    $newChatProfile: atom('hermes-setup'),
    $newChatRoute: atom(null),
    ensureGatewayProfile: async (profile: string) => {
      await mocks.ensure()
      activeGatewayProfile.set(profile)
    },
    ensureGatewayAgent: mocks.ensure,
    normalizeProfileKey: (profile: string) => profile || 'default'
  }
})
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')
  const active = atom<string | null>('guide-runtime')
  const selected = atom<string | null>('guide-stored')

  return {
    $activeSessionId: active,
    $selectedStoredSessionId: selected,
    $messages: atom([]),
    setActiveSessionId: (id: string) => active.set(id),
    setAwaitingResponse: vi.fn(),
    setBusy: vi.fn(),
    setMessages: vi.fn(),
    setSessionOwnerHint: vi.fn(),
    forgetSessionOwnerHintsForSession: vi.fn(),
    getSessionOwnerHint: () => ({ connectionId: 'source-a', profile: 'default' })
  }
})

import type { SessionCreateOverrides } from '@/app/session/hooks/use-session-actions/create-overrides'
import type { ClientSessionState } from '@/app/types'
import { startChatOnboardingSolo } from '@/components/onboarding-chat/assembly'
import {
  $setupHandoff,
  $setupSession,
  hasCompletedSetupHandoff,
  resetSetupHandoffForTests
} from '@/components/onboarding-chat/setup-profile'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $onboardingAnswers, DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { $onboardingGate, devResetOnboardingFlow } from '@/store/onboarding-gate'
import { buildChatOnboardingSeedMessages } from '@/store/onboarding-script'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { retrySetupHandoff } from './handoff-receipt'
import { type OnboardingHandoffOptions, useOnboardingHandoff } from './onboarding-handoff'

const task = { task: 'Tracker', brief: 'Build my tracker', plan: 'build' as const }

function harness() {
  const states = new Map<string, ClientSessionState>()

  const options: OnboardingHandoffOptions = {
    activeSessionIdRef: { current: 'guide-runtime' },
    createBackendSessionForSend: vi.fn(async () => {
      $activeSessionId.set('build-runtime')
      $selectedStoredSessionId.set('build-stored')
      options.activeSessionIdRef.current = 'build-runtime'

      return 'build-runtime'
    }),
    ensureSessionState: (id, storedId) => {
      if (!states.has(id)) {
        states.set(id, createClientSessionState(storedId ?? null))
      }

      return states.get(id)!
    },
    updateSessionState: (id, updater, storedId) => {
      const next = updater(options.ensureSessionState(id, storedId))
      states.set(id, next)

      return next
    },
    requestGateway: vi.fn(),
    resumeSession: vi.fn(async () => undefined),
    runCreatePinnedTo: async (_profile, create) => create()
  }

  const hook = renderHook(() => useOnboardingHandoff(options))

  return { ...hook, options, states }
}

afterEach(() => vi.unstubAllGlobals())

beforeEach(() => {
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: true })
  devResetOnboardingFlow()
  $onboardingGate.set({ phase: 'guided', guideQueued: false })
  $onboardingAnswers.set({ ...DEFAULT_ANSWERS, name: 'Ada', context: 'Garden tracker', connectors: ['Calendar'] })
  localStorage.clear()
  resetSetupHandoffForTests()
  vi.clearAllMocks()
  mocks.connectionId = 'source-a'
  $activeGatewayProfile.set('hermes-setup')
  $newChatProfile.set('hermes-setup')
  $setupSession.set({ profile: 'hermes-setup', runtimeId: 'guide-runtime', storedId: 'guide-stored' })
  $activeSessionId.set('guide-runtime')
  $selectedStoredSessionId.set('guide-stored')
  mocks.request.mockImplementation(async (_connection, profile, method) => {
    if (method === 'setup.status') {
      return { ready: true, provider_configured: true, free_tier: false }
    }

    if (method === 'profiles.remember_onboarding') {
      return { saved: true, profile: 'default', target: 'user' }
    }

    if (method === 'session.resume') {
      return { session_id: 'build-runtime-2', session_key: 'build-stored', running: false, messages: [] }
    }

    if (method === 'prompt.submit' && profile === 'default') {
      throw new JsonRpcGatewayError('Provider unavailable', { code: 4090 })
    }

    return { status: 'streaming' }
  })
})

describe('the real onboarding handoff effect', () => {
  it('leaves the classic onboarding profiles unchanged when the guide is not ready', async () => {
    $newChatProfile.set('default')
    $activeGatewayProfile.set('default')
    const originalNewChatProfile = $newChatProfile.get()
    const originalActiveGatewayProfile = $activeGatewayProfile.get()
    const h = harness()
    mocks.request.mockResolvedValue({ ready: false, provider_configured: true })

    await act(async () => expect(await h.result.current()).toBe(false))

    expect($newChatProfile.get()).toBe(originalNewChatProfile)
    expect($activeGatewayProfile.get()).toBe(originalActiveGatewayProfile)
    expect(mocks.ensure).not.toHaveBeenCalled()
    expect(h.options.createBackendSessionForSend).not.toHaveBeenCalled()
  })

  it.each([
    { record: { provider_configured: true }, starts: false },
    { record: { ready: false, provider_configured: true }, starts: false },
    { record: { ready: true, provider_configured: false }, starts: false },
    { record: { ready: true, provider_configured: true, free_tier: false }, starts: true },
    { record: { ready: true, provider_configured: true, free_tier: true }, starts: true }
  ])('starts only after the guide backend confirms readiness: $record', async ({ record, starts }) => {
    const h = harness()
    mocks.request.mockImplementation(async (_connection, _profile, method) =>
      method === 'setup.status' ? record : { sessions: [] }
    )

    await act(async () => expect(await h.result.current()).toBe(starts))
    expect(mocks.request.mock.calls.filter(([, , method]) => method === 'setup.status')).toEqual([
      ['source-a', 'hermes-setup', 'setup.status', {}]
    ])
    expect(mocks.ensure).toHaveBeenCalledTimes(starts ? 1 : 0)
    expect(startChatOnboardingSolo).toHaveBeenCalledTimes(starts ? 1 : 0)
    expect(h.options.createBackendSessionForSend).toHaveBeenCalledTimes(starts ? 1 : 0)

    if (starts) {
      expect(mocks.request.mock.invocationCallOrder[0]).toBeLessThan(mocks.ensure.mock.invocationCallOrder[0])
      expect(buildChatOnboardingSeedMessages).toHaveBeenCalledWith(undefined, record.free_tier !== true)
      const createOverrides: SessionCreateOverrides = { title: 'Welcome to Hermes' }

      if (record.free_tier) {
        createOverrides.reasoningEffort = 'minimal'
      }

      expect(h.options.createBackendSessionForSend).toHaveBeenCalledWith(null, [], createOverrides)
    }

    expect(mocks.request.mock.calls.some(([, , method]) => method === 'config.set')).toBe(false)
  })

  it('adopts the titled guide and changes only free-tier reasoning on that backend', async () => {
    const h = harness()
    mocks.request.mockImplementation(async (_connection, _profile, method) =>
      method === 'setup.status'
        ? { ready: true, provider_configured: true, free_tier: true }
        : { sessions: [{ id: 'guide-stored', resolved_id: 'guide-tip' }] }
    )
    await act(async () => expect(await h.result.current()).toBe(true))
    expect(h.options.resumeSession).toHaveBeenCalledWith('guide-tip', true)
    expect(h.options.createBackendSessionForSend).not.toHaveBeenCalled()
    expect(mocks.request.mock.calls.filter(([, , method]) => method === 'config.set')).toEqual([
      [
        'source-a',
        'hermes-setup',
        'config.set',
        { session_id: 'guide-runtime', key: 'reasoning', value: 'minimal' },
        undefined
      ]
    ])
  })

  it.each(['profile creation', 'profile swap'])('surfaces %s failure without a profile-less create', async stage => {
    const h = harness()
    const error = new Error(stage)

    if (stage === 'profile creation') {
      vi.mocked(h.options.requestGateway).mockRejectedValueOnce(error)
    } else {
      mocks.ensure.mockRejectedValueOnce(error)
    }

    await act(async () => expect(await h.result.current()).toBe(false))
    expect(h.options.createBackendSessionForSend).not.toHaveBeenCalled()
    expect(mocks.request).toHaveBeenCalledTimes(stage === 'profile creation' ? 0 : 1)
    expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', message: stage }))
  })

  it('adds no backend work or onboarding surface with the flag absent', async () => {
    vi.stubGlobal('hermesDesktop', {})
    const h = harness()
    await act(async () => expect(await h.result.current()).toBe(false))
    act(() => $setupHandoff.set({ ...task, phase: 'pending' }))
    expect(startChatOnboardingSolo).not.toHaveBeenCalled()
    expect(h.options.requestGateway).not.toHaveBeenCalled()
    expect(h.options.createBackendSessionForSend).not.toHaveBeenCalled()
    expect(mocks.request).not.toHaveBeenCalled()
    expect(mocks.ensure).not.toHaveBeenCalled()
  })

  it('surfaces rejection, saves facts before create, then retries the SAME build with its exact owner', async () => {
    const h = harness()
    act(() => $setupHandoff.set({ ...task, phase: 'pending' }))
    await waitFor(() => expect($setupHandoff.get()?.phase).toBe('error'))
    expect(hasCompletedSetupHandoff()).toBe(false)
    expect(mocks.watch).not.toHaveBeenCalled()
    expect(h.states.get('build-runtime')).toMatchObject({ busy: false, awaitingResponse: false, messages: [] })
    expect(h.options.createBackendSessionForSend).toHaveBeenCalledWith(task.brief, expect.any(Array))
    expect(mocks.request).toHaveBeenCalledWith(
      'source-a',
      'default',
      'profiles.remember_onboarding',
      expect.objectContaining({ answers: expect.objectContaining({ name: 'Ada' }) }),
      expect.anything()
    )
    expect(mocks.request.mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(h.options.createBackendSessionForSend).mock.invocationCallOrder[0]
    )
    expect(
      mocks.request.mock.calls.filter(([, profile, method]) => profile === 'hermes-setup' && method === 'prompt.submit')
    ).toHaveLength(0)

    mocks.request.mockImplementation(async (_connection, _profile, method) =>
      method === 'session.resume'
        ? { session_id: 'build-runtime-2', session_key: 'build-stored', running: false, messages: [] }
        : { status: 'streaming' }
    )
    mocks.connectionId = 'source-b'
    act(retrySetupHandoff)
    await waitFor(() => expect($setupHandoff.get()?.phase).toBe('done'))
    expect(h.options.createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(mocks.request.mock.calls.some(([connection]) => connection === 'source-b')).toBe(false)
    expect(hasCompletedSetupHandoff()).toBe(true)
    expect($onboardingGate.get().phase).toBe('done')
    expect(h.options.activeSessionIdRef.current).toBe('build-runtime-2')
    expect(mocks.tile).toHaveBeenCalledWith('build-stored', {
      runtimeId: 'build-runtime-2',
      ownerRoute: { connectionId: 'source-a', profile: 'default' }
    })
    expect(h.states.get('build-runtime-2')?.messages).toHaveLength(1)
    expect(mocks.request).toHaveBeenCalledWith(
      'source-a',
      'default',
      'prompt.submit',
      { session_id: 'build-runtime-2', text: task.brief },
      expect.anything()
    )
    expect(mocks.watch).toHaveBeenCalledWith('build-runtime-2', 'default')
  })

  it('reconciles a lost ACK after remount without creating or submitting another build', async () => {
    mocks.request.mockImplementation(async (_connection, profile, method) => {
      if (method === 'profiles.remember_onboarding') {
        return { saved: true, profile: 'default', target: 'user' }
      }

      if (method === 'prompt.submit' && profile === 'default') {
        throw new Error('request timed out')
      }

      return {}
    })
    const h = harness()
    act(() => $setupHandoff.set({ ...task, phase: 'pending' }))
    await waitFor(() => expect($setupHandoff.get()?.phase).toBe('error'))
    expect(hasCompletedSetupHandoff()).toBe(false)
    h.unmount()
    // Relaunch into the guide's replayed transcript: no beacon or transient
    // setup pointer survives. The durable receipt alone restores recovery.
    resetSetupHandoffForTests()
    $activeSessionId.set('guide-runtime')
    $selectedStoredSessionId.set('guide-stored')
    mocks.request.mockImplementation(async () => ({
      session_id: 'recovered',
      session_key: 'build-stored',
      running: true,
      messages: []
    }))
    const resumed = harness()
    await waitFor(() => expect($setupHandoff.get()?.phase).toBe('done'))
    expect(resumed.options.createBackendSessionForSend).not.toHaveBeenCalled()
    expect(
      mocks.request.mock.calls.filter(([, profile, method]) => profile === 'default' && method === 'prompt.submit')
    ).toHaveLength(1)
    expect(
      mocks.request.mock.calls.some(([, , method]) => method === 'session.close' || method === 'session.list')
    ).toBe(false)

    // The completed receipt also makes another relaunch inert: no extra
    // success whisper or status polling starts a new guide turn.
    resumed.unmount()
    resetSetupHandoffForTests()
    const calls = mocks.request.mock.calls.length
    harness()
    await waitFor(() => expect($setupHandoff.get()?.phase).toBe('done'))
    expect(mocks.request.mock.calls).toHaveLength(calls)
  })
})
