import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from './composer-queue'
import { $gatewayStatesByProfile } from './gateway'
import { $petUnread } from './pet'
import { $petActionCenter, clearPetActionCenterActionStatus } from './pet-action-center'
import {
  createPetActionCenterActions,
  type PetActionCenterActionDependencies,
  type PetActionCenterGateway
} from './pet-action-center-actions'
import {
  $petLiveSessions,
  completePetLiveSession,
  reconcilePetLiveSessionFocus,
  resetPetLiveSessions,
  syncPetLiveSessionState
} from './pet-live-session'
import { $activeGatewayProfile, $profiles } from './profile'
import { clearAllPrompts, setApprovalRequest } from './prompts'
import { $activeSessionId, $sessions } from './session'
import { profileSessionKey } from './session-identity'

interface Pending<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function pending<T>(): Pending<T> {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const snapshot = (
  profile: string,
  overrides: Partial<Parameters<typeof syncPetLiveSessionState>[0]> = {}
) => {
  const input = {
    profile,
    runtimeSessionId: 'runtime',
    storedSessionId: 'stored',
    busy: true,
    needsInput: false,
    awaitingResponse: true,
    turnStartedAt: 10,
    ...overrides
  }

  syncPetLiveSessionState(
    input,
    input.busy ? null : { profile, runtimeSessionId: input.runtimeSessionId }
  )
}

function liveItem(profile = 'default') {
  return $petActionCenter.get().items.find(item => item.kind === 'live-turn' && item.profile === profile)!
}

function gateway(
  request: (method: string, params?: Record<string, unknown>) => Promise<unknown>
): PetActionCenterGateway {
  return {
    connectionState: 'open',
    request<T>(method: string, params?: Record<string, unknown>): Promise<T> {
      return request(method, params) as Promise<T>
    }
  }
}

describe('pet live-turn main-renderer actions', () => {
  const gateways = new Map<string, PetActionCenterGateway>()
  const ensureProfile = vi.fn().mockResolvedValue(undefined)
  const resumeSession = vi.fn().mockResolvedValue(true)
  let dependencies: PetActionCenterActionDependencies

  beforeEach(() => {
    resetPetLiveSessions()
    clearAllPrompts()
    clearClarifyRequest()
    clearPetActionCenterActionStatus()
    $gatewayStatesByProfile.set({ default: 'open', work: 'open' })
    $queuedPromptsBySession.set({})
    $profiles.set([])
    $activeGatewayProfile.set('default')
    $activeSessionId.set('runtime')
    $sessions.set([
      { id: 'stored', profile: 'default', title: 'Default' },
      { id: 'stored', profile: 'work', title: 'Work' }
    ] as never)
    $petUnread.set(false)
    gateways.clear()
    ensureProfile.mockClear()
    resumeSession.mockClear()
    dependencies = {
      ensureProfile,
      enqueuePrompt: enqueueQueuedPrompt,
      gatewayForProfile: profile => gateways.get(profile) ?? null,
      resumeSession
    }
  })

  afterEach(() => {
    resetPetLiveSessions()
    clearAllPrompts()
    clearClarifyRequest()
    clearPetActionCenterActionStatus()
    $gatewayStatesByProfile.set({})
    $queuedPromptsBySession.set({})
    $sessions.set([])
    $activeGatewayProfile.set('default')
    $activeSessionId.set(null)
    $petUnread.set(false)
  })

  it('routes send through the exact profile/runtime and fails closed when disconnected', async () => {
    const defaultRequest = vi.fn().mockResolvedValue({})
    const workRequest = vi.fn().mockResolvedValue({})
    gateways.set('default', gateway(defaultRequest))
    gateways.set('work', gateway(workRequest))
    snapshot('default', { busy: false, awaitingResponse: false, turnStartedAt: null })
    snapshot('work', { busy: false, awaitingResponse: false, turnStartedAt: null })
    $activeGatewayProfile.set('work')
    const item = liveItem('work')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: item.id,
      text: '  hello work  '
    })

    expect(defaultRequest).not.toHaveBeenCalled()
    expect(workRequest).toHaveBeenCalledWith('prompt.submit', { session_id: 'runtime', text: 'hello work' })

    const disconnected = { connectionState: 'closed', request: vi.fn() } satisfies PetActionCenterGateway
    gateways.set('default', disconnected)
    $activeGatewayProfile.set('default')
    snapshot('default', { busy: false, awaitingResponse: false, turnStartedAt: null })

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: liveItem('default').id,
      text: 'hello'
    })

    expect(disconnected.request).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual(
      expect.objectContaining({ status: 'error', errorCode: 'disconnected' })
    )
  })

  it('recovers idle send once on the same profile with a verified stored id and never switches foreground', async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit' && params?.session_id === 'runtime') {
        throw new Error('session not found')
      }

      if (method === 'session.resume') {
        return { session_id: 'fresh-runtime' }
      }

      return {}
    })

    gateways.set('work', gateway(request))
    $activeGatewayProfile.set('work')
    snapshot('work', { busy: false, awaitingResponse: false, turnStartedAt: null })
    const item = liveItem('work')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: item.id,
      text: 'continue'
    })

    expect(request.mock.calls).toEqual([
      ['prompt.submit', { session_id: 'runtime', text: 'continue' }],
      ['session.resume', { session_id: 'stored', source: 'desktop' }],
      ['prompt.submit', { session_id: 'fresh-runtime', text: 'continue' }]
    ])
    expect(ensureProfile).not.toHaveBeenCalled()
    expect(resumeSession).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'success', itemId: item.id })
    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        awaitingResponse: true,
        busy: true,
        profile: 'work',
        runtimeSessionId: 'fresh-runtime',
        storedSessionId: 'stored'
      })
    ])
    expect($petActionCenter.get().items.some(candidate => candidate.id === item.id)).toBe(false)
  })

  it('accepts an expected idle runtime rotation projected during session.resume before retrying once', async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit' && params?.session_id === 'runtime') {
        throw new Error('session not found')
      }

      if (method === 'session.resume') {
        syncPetLiveSessionState(
          {
            profile: 'work',
            runtimeSessionId: 'fresh-runtime',
            storedSessionId: 'stored',
            busy: false,
            needsInput: false,
            awaitingResponse: false,
            turnStartedAt: null
          },
          { profile: 'work', runtimeSessionId: 'fresh-runtime' }
        )
        reconcilePetLiveSessionFocus({ profile: 'work', runtimeSessionId: 'fresh-runtime' })

        return { session_id: 'fresh-runtime' }
      }

      return {}
    })

    gateways.set('work', gateway(request))
    $activeGatewayProfile.set('work')
    snapshot('work', { busy: false, awaitingResponse: false, turnStartedAt: null })
    const item = liveItem('work')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: item.id,
      text: 'continue'
    })

    expect(request.mock.calls).toEqual([
      ['prompt.submit', { session_id: 'runtime', text: 'continue' }],
      ['session.resume', { session_id: 'stored', source: 'desktop' }],
      ['prompt.submit', { session_id: 'fresh-runtime', text: 'continue' }]
    ])
    expect($activeSessionId.get()).toBe('runtime')
    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        busy: true,
        profile: 'work',
        runtimeSessionId: 'fresh-runtime',
        storedSessionId: 'stored'
      })
    ])
  })

  it('fails closed when session.resume projects the recovered runtime as already busy', async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit' && params?.session_id === 'runtime') {
        throw new Error('request timed out: prompt.submit')
      }

      if (method === 'session.resume') {
        syncPetLiveSessionState(
          {
            profile: 'work',
            runtimeSessionId: 'fresh-runtime',
            storedSessionId: 'stored',
            busy: true,
            needsInput: false,
            awaitingResponse: true,
            turnStartedAt: 50
          },
          { profile: 'work', runtimeSessionId: 'fresh-runtime' }
        )
        reconcilePetLiveSessionFocus({ profile: 'work', runtimeSessionId: 'fresh-runtime' })

        return { session_id: 'fresh-runtime' }
      }

      return {}
    })

    gateways.set('work', gateway(request))
    $activeGatewayProfile.set('work')
    snapshot('work', { busy: false, awaitingResponse: false, turnStartedAt: null })
    const item = liveItem('work')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: item.id,
      text: 'continue'
    })

    expect(request).toHaveBeenCalledTimes(2)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId: item.id, errorCode: 'stale-runtime' })
    expect($activeSessionId.get()).toBe('runtime')
  })

  it('fails stale-runtime when resume recovery returns no fresh id', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        throw new Error('request timed out: prompt.submit')
      }

      return {}
    })

    gateways.set('work', gateway(request))
    $activeGatewayProfile.set('work')
    snapshot('work', { busy: false, awaitingResponse: false, turnStartedAt: null })
    const item = liveItem('work')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-submit',
      itemId: item.id,
      text: 'continue'
    })

    expect(request).toHaveBeenCalledTimes(2)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId: item.id, errorCode: 'stale-runtime' })
  })

  it('re-reads capability before side effects and serializes duplicate live actions', async () => {
    const rpc = pending<{}>()
    const request = vi.fn(() => rpc.promise)
    gateways.set('default', gateway(request))
    snapshot('default', { busy: false, awaitingResponse: false, turnStartedAt: null })
    const item = liveItem()
    const actions = createPetActionCenterActions(dependencies)

    const first = actions.handle({ type: 'action-center-submit', itemId: item.id, text: 'one' })
    const duplicate = actions.handle({ type: 'action-center-submit', itemId: item.id, text: 'one' })

    expect(request).toHaveBeenCalledTimes(1)
    rpc.resolve({})
    await Promise.all([first, duplicate])

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        awaitingResponse: true,
        busy: true,
        runtimeSessionId: 'runtime',
        storedSessionId: 'stored'
      })
    ])
    expect(liveItem().allowedActions).not.toContain('send')

    await actions.handle({ type: 'action-center-submit', itemId: item.id, text: 'two' })
    expect(request).toHaveBeenCalledTimes(1)
    expect($petActionCenter.get().action).toEqual(
      expect.objectContaining({ status: 'error', errorCode: 'capability-denied' })
    )
  })

  it('distinguishes accepted steer from rejected steer and never autoqueues or resumes', async () => {
    const accepted = vi.fn().mockResolvedValue({ status: 'queued' })
    gateways.set('default', gateway(accepted))
    snapshot('default')
    let item = liveItem()

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-steer',
      itemId: item.id,
      text: '  nudge  '
    })

    expect(accepted).toHaveBeenCalledWith('session.steer', { session_id: 'runtime', text: 'nudge' })
    expect($petActionCenter.get().action).toEqual({ status: 'steered', itemId: item.id })

    clearPetActionCenterActionStatus()
    const rejected = vi.fn().mockResolvedValue({ status: 'rejected' })
    gateways.set('default', gateway(rejected))
    item = liveItem()
    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-steer',
      itemId: item.id,
      text: 'later'
    })

    expect($petActionCenter.get().action).toEqual({ status: 'steer-rejected', itemId: item.id })
    expect(getQueuedPrompts(profileSessionKey('default', 'stored'))).toEqual([])
    expect(rejected).toHaveBeenCalledTimes(1)
    expect(ensureProfile).not.toHaveBeenCalled()
    expect(resumeSession).not.toHaveBeenCalled()
  })

  it('queues trimmed text locally at the canonical stored key with profile isolation', async () => {
    $gatewayStatesByProfile.set({ default: 'closed', work: 'connecting' })
    snapshot('default')
    snapshot('work')
    const gatewayForProfile = vi.fn(() => null)
    const actions = createPetActionCenterActions({ ...dependencies, gatewayForProfile })

    await actions.handle({ type: 'action-center-queue', itemId: liveItem('work').id, text: '  later  ' })

    expect(getQueuedPrompts(profileSessionKey('work', 'stored'))).toEqual([
      expect.objectContaining({ attachments: [], text: 'later' })
    ])
    expect(getQueuedPrompts(profileSessionKey('default', 'stored'))).toEqual([])
    expect(gatewayForProfile).not.toHaveBeenCalled()

    clearPetActionCenterActionStatus()
    await actions.handle({ type: 'action-center-queue', itemId: liveItem('default').id, text: 'offline' })
    expect(getQueuedPrompts(profileSessionKey('default', 'stored'))).toEqual([
      expect.objectContaining({ attachments: [], text: 'offline' })
    ])
    expect(gatewayForProfile).not.toHaveBeenCalled()

    clearPetActionCenterActionStatus()
    await actions.handle({ type: 'action-center-queue', itemId: liveItem('default').id, text: '   ' })
    expect($petActionCenter.get().action).toEqual(
      expect.objectContaining({ status: 'error', errorCode: 'invalid-text' })
    )

    clearPetActionCenterActionStatus()
    snapshot('default', { needsInput: true })
    await actions.handle({ type: 'action-center-queue', itemId: liveItem('default').id, text: 'not while waiting' })
    expect(getQueuedPrompts(profileSessionKey('default', 'stored'))).toHaveLength(1)
    expect($petActionCenter.get().action).toEqual(
      expect.objectContaining({ status: 'error', errorCode: 'capability-denied' })
    )
    expect(gatewayForProfile).not.toHaveBeenCalled()
  })

  it('stops only the exact runtime, never resumes stale runtime, and clears exact prompts only on success', async () => {
    const defaultRequest = vi.fn().mockResolvedValue({ interrupted: true })
    const workRequest = vi.fn()
    gateways.set('default', gateway(defaultRequest))
    gateways.set('work', gateway(workRequest))
    snapshot('default')
    snapshot('work')
    setApprovalRequest({ command: 'default', description: 'Default', profile: 'default', sessionId: 'other-runtime' })
    setClarifyRequest({
      choices: null,
      profile: 'work',
      question: 'Work?',
      requestId: 'work-request',
      sessionId: 'other-runtime'
    })
    const item = liveItem('default')

    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-stop', itemId: item.id })

    expect(defaultRequest).toHaveBeenCalledWith('session.interrupt', { session_id: 'runtime' })
    expect(workRequest).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'stopped', itemId: item.id })
    expect($petActionCenter.get().items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'approval', profile: 'default', sessionId: 'other-runtime' }),
        expect.objectContaining({ kind: 'clarify', profile: 'work', sessionId: 'other-runtime' })
      ])
    )

    clearPetActionCenterActionStatus()
    defaultRequest.mockRejectedValueOnce(new Error('session not found'))
    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-stop', itemId: item.id })
    expect(defaultRequest).toHaveBeenCalledTimes(2)
    expect(ensureProfile).not.toHaveBeenCalled()
    expect(resumeSession).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId: item.id, errorCode: 'stale-runtime' })
  })

  it('clears prompts for the stopped profile/runtime only after interrupt succeeds', async () => {
    const rpc = pending<{}>()
    const request = vi.fn(() => rpc.promise)
    gateways.set('default', gateway(request))
    snapshot('default')
    const item = liveItem('default')

    const action = createPetActionCenterActions(dependencies).handle({
      type: 'action-center-stop',
      itemId: item.id
    })

    setApprovalRequest({ command: 'exact', description: 'Exact', profile: 'default', sessionId: 'runtime' })
    setClarifyRequest({
      choices: null,
      profile: 'default',
      question: 'Other?',
      requestId: 'other',
      sessionId: 'other-runtime'
    })

    expect($petActionCenter.get().items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'approval', sessionId: 'runtime' }),
        expect.objectContaining({ kind: 'clarify', sessionId: 'other-runtime' })
      ])
    )

    rpc.resolve({})
    await action

    expect(
      $petActionCenter.get().items.some(candidate => candidate.kind === 'approval' && candidate.sessionId === 'runtime')
    ).toBe(false)
    expect($petActionCenter.get().items).toEqual(
      expect.arrayContaining([expect.objectContaining({ kind: 'clarify', sessionId: 'other-runtime' })])
    )
  })

  it('acknowledges only the exact outcome without RPC or clearing approvals and clarifies', async () => {
    snapshot('default')
    snapshot('work')
    completePetLiveSession('default', 'runtime', 'done')
    completePetLiveSession('work', 'runtime', 'failed')
    setApprovalRequest({ command: 'approve', description: 'Approve', profile: 'default', sessionId: 'prompt-runtime' })
    setClarifyRequest({
      choices: null,
      profile: 'work',
      question: 'Clarify?',
      requestId: 'clarify',
      sessionId: 'prompt-runtime'
    })
    $petUnread.set(true)
    const item = liveItem('default')

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-acknowledge',
      itemId: item.id
    })

    expect($petLiveSessions.get()).toEqual([expect.objectContaining({ profile: 'work', outcome: 'failed' })])
    expect($petActionCenter.get().items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'approval', profile: 'default' }),
        expect.objectContaining({ kind: 'clarify', profile: 'work' })
      ])
    )
    expect($petUnread.get()).toBe(true)
    expect(gateways.size).toBe(0)
    expect($petActionCenter.get().action).toEqual({ status: 'acknowledged', itemId: item.id })
  })
})
