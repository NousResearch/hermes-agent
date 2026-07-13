import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import { $petActionCenter, clearPetActionCenterActionStatus } from './pet-action-center'
import {
  createPetActionCenterActions,
  type PetActionCenterActionDependencies,
  type PetActionCenterGateway
} from './pet-action-center-actions'
import { $profiles } from './profile'
import { $approvalRequests, clearAllPrompts, setApprovalRequest } from './prompts'
import { $sessions } from './session'

interface Pending<T> {
  promise: Promise<T>
  reject: (error: unknown) => void
  resolve: (value: T) => void
}

function pending<T>(): Pending<T> {
  let reject!: (error: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve
    reject = onReject
  })

  return { promise, reject, resolve }
}

function openGateway(request = vi.fn().mockResolvedValue({ resolved: true })): PetActionCenterGateway {
  return { connectionState: 'open', request }
}

describe('pet action center main-renderer actions', () => {
  const gateways = new Map<string, PetActionCenterGateway>()
  const ensureProfile = vi.fn().mockResolvedValue(undefined)
  const resumeSession = vi.fn().mockResolvedValue(true)
  let dependencies: PetActionCenterActionDependencies

  beforeEach(() => {
    clearAllPrompts()
    clearClarifyRequest()
    clearPetActionCenterActionStatus()
    $profiles.set([])
    $sessions.set([])
    gateways.clear()
    ensureProfile.mockClear()
    resumeSession.mockClear()
    dependencies = {
      ensureProfile,
      gatewayForProfile: profile => gateways.get(profile) ?? null,
      resumeSession
    }
  })

  afterEach(() => {
    clearAllPrompts()
    clearClarifyRequest()
    clearPetActionCenterActionStatus()
    $sessions.set([])
  })

  it.each([
    ['approve-once', 'once'],
    ['approve-session', 'session'],
    ['approve-always', 'always'],
    ['deny', 'deny']
  ] as const)('maps %s through the item profile gateway to backend choice %s', async (choice, backendChoice) => {
    const defaultRequest = vi.fn().mockResolvedValue({ resolved: true })
    const workRequest = vi.fn().mockResolvedValue({ resolved: true })
    gateways.set('default', openGateway(defaultRequest))
    gateways.set('work', openGateway(workRequest))
    setApprovalRequest({
      command: 'npm test',
      description: 'Run tests',
      profile: 'work',
      sessionId: 'shared-runtime'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-approval', itemId, choice })

    expect(defaultRequest).not.toHaveBeenCalled()
    expect(workRequest).toHaveBeenCalledWith('approval.respond', {
      choice: backendChoice,
      session_id: 'shared-runtime'
    })
    expect($petActionCenter.get().items).toHaveLength(0)
    expect($petActionCenter.get().action).toEqual({ status: 'success', itemId })
  })

  it('includes a denial reason only for deny', async () => {
    const request = vi.fn().mockResolvedValue({ resolved: true })
    gateways.set('work', openGateway(request))
    setApprovalRequest({ command: 'rm tmp', description: 'Remove temp', profile: 'work', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'deny',
      reason: 'Not safe in this workspace'
    })

    expect(request).toHaveBeenCalledWith('approval.respond', {
      choice: 'deny',
      reason: 'Not safe in this workspace',
      session_id: 'runtime'
    })
  })

  it('does not cross-route equal runtime ids between profiles', async () => {
    const defaultRequest = vi.fn().mockResolvedValue({ resolved: true })
    const workRequest = vi.fn().mockResolvedValue({ resolved: true })
    gateways.set('default', openGateway(defaultRequest))
    gateways.set('work', openGateway(workRequest))
    setApprovalRequest({ command: 'default', description: 'Default', profile: 'default', sessionId: 'shared' })
    setApprovalRequest({ command: 'work', description: 'Work', profile: 'work', sessionId: 'shared' })
    const defaultItemId = $petActionCenter.get().items.find(item => item.profile === 'default')!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId: defaultItemId,
      choice: 'approve-once'
    })

    expect(defaultRequest).toHaveBeenCalledWith('approval.respond', { choice: 'once', session_id: 'shared' })
    expect(workRequest).not.toHaveBeenCalled()
    expect($petActionCenter.get().items).toEqual([expect.objectContaining({ profile: 'work', sessionId: 'shared' })])
  })

  it('rejects a choice that the current item does not allow', async () => {
    const request = vi.fn()
    gateways.set('default', openGateway(request))
    setApprovalRequest({
      allowPermanent: false,
      choices: ['once', 'deny'],
      command: 'npm test',
      description: 'Run tests',
      profile: 'default',
      sessionId: 'runtime'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'approve-always'
    })

    expect(request).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'capability-denied' })
  })

  it('uses the original clarify request id from the item identity', async () => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })
    gateways.set('work', openGateway(request))
    setClarifyRequest({
      choices: ['Blue', 'Green'],
      profile: 'work',
      question: 'Which color?',
      requestId: 'backend-request-id',
      sessionId: 'runtime'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-clarify',
      itemId,
      answer: 'Blue'
    })

    expect(request).toHaveBeenCalledWith('clarify.respond', {
      answer: 'Blue',
      request_id: 'backend-request-id'
    })
    expect($petActionCenter.get().items).toHaveLength(0)
  })

  it('sends an empty answer for clarify-skip', async () => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })
    gateways.set('default', openGateway(request))
    setClarifyRequest({
      choices: null,
      profile: 'default',
      question: 'Continue?',
      requestId: 'clarify-skip',
      sessionId: 'runtime'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-clarify',
      itemId,
      answer: ''
    })

    expect(request).toHaveBeenCalledWith('clarify.respond', { answer: '', request_id: 'clarify-skip' })
    expect($petActionCenter.get().action).toEqual({ status: 'success', itemId })
  })

  it('leaves a replacement parked when it arrives during an approval RPC', async () => {
    const rpc = pending<{ resolved: boolean }>()
    const request = vi.fn(() => rpc.promise)
    gateways.set('default', openGateway(request))
    setApprovalRequest({ command: 'first', description: 'First', profile: 'default', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    const action = createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'approve-once'
    })

    expect($petActionCenter.get().action).toEqual({ status: 'submitting', itemId })

    setApprovalRequest({ command: 'replacement', description: 'Replacement', profile: 'default', sessionId: 'runtime' })
    const replacementIdentity = Object.values($approvalRequests.get())[0]!.requestIdentity
    rpc.resolve({ resolved: true })
    await action

    expect(Object.values($approvalRequests.get())[0]?.requestIdentity).toBe(replacementIdentity)
    expect($petActionCenter.get().items[0]).toEqual(expect.objectContaining({ summary: 'Replacement' }))
  })

  it('serializes action dispatch so duplicate controls cannot resolve one request twice', async () => {
    const rpc = pending<{ resolved: boolean }>()
    const request = vi.fn(() => rpc.promise)
    gateways.set('default', openGateway(request))
    setApprovalRequest({ command: 'once', description: 'Once', profile: 'default', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id
    const actions = createPetActionCenterActions(dependencies)

    const first = actions.handle({ type: 'action-center-approval', itemId, choice: 'approve-once' })
    const duplicate = actions.handle({ type: 'action-center-approval', itemId, choice: 'approve-once' })

    expect(request).toHaveBeenCalledTimes(1)
    rpc.resolve({ resolved: true })
    await Promise.all([first, duplicate])
  })

  it('treats resolved zero as stale and removes only the still-identical old item', async () => {
    const request = vi.fn().mockResolvedValue({ resolved: 0 })
    gateways.set('default', openGateway(request))
    setApprovalRequest({ command: 'old', description: 'Old', profile: 'default', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'approve-once'
    })

    expect($petActionCenter.get().items).toHaveLength(0)
    expect($petActionCenter.get().action).toEqual({ status: 'stale', itemId })
  })

  it('treats the gateway clarify-expired contract as stale instead of a retryable RPC failure', async () => {
    const request = vi.fn().mockRejectedValue(new Error('no pending answer request'))
    gateways.set('default', openGateway(request))
    setClarifyRequest({ choices: null, profile: 'default', question: 'Continue?', requestId: 'expired', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-clarify',
      itemId,
      answer: 'Yes'
    })

    expect($petActionCenter.get().items).toHaveLength(0)
    expect($petActionCenter.get().action).toEqual({ status: 'stale', itemId })
  })

  it.each([{}, { ok: false }])('parks approval on an invalid non-stale RPC result %#', async result => {
    const request = vi.fn().mockResolvedValue(result)

    gateways.set('default', openGateway(request))
    setApprovalRequest({ command: 'npm test', description: 'Run tests', profile: 'default', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'approve-once'
    })

    expect($petActionCenter.get().items).toHaveLength(1)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'rpc-failed' })
  })

  it.each([{}, { ok: false }, { resolved: false }])('parks clarify on an invalid non-stale RPC result %#', async result => {
    const request = vi.fn().mockResolvedValue(result)

    gateways.set('default', openGateway(request))
    setClarifyRequest({ choices: null, profile: 'default', question: 'Continue?', requestId: 'req', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-clarify',
      itemId,
      answer: 'Yes'
    })

    expect($petActionCenter.get().items).toHaveLength(1)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'rpc-failed' })
  })

  it('leaves the item parked and exposes an error code when its profile is disconnected', async () => {
    setApprovalRequest({ command: 'npm test', description: 'Run tests', profile: 'work', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-approval',
      itemId,
      choice: 'approve-once'
    })

    expect($petActionCenter.get().items).toHaveLength(1)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'disconnected' })
  })

  it('leaves the item parked and reports RPC failures without prose', async () => {
    const request = vi.fn().mockRejectedValue(new Error('transport exploded'))
    gateways.set('default', openGateway(request))
    setClarifyRequest({ choices: null, profile: 'default', question: 'Continue?', requestId: 'req', sessionId: 'runtime' })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({
      type: 'action-center-clarify',
      itemId,
      answer: 'Yes'
    })

    expect($petActionCenter.get().items).toHaveLength(1)
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'rpc-failed' })
    expect(JSON.stringify($petActionCenter.get().action)).not.toContain('transport exploded')
  })

  it('opens only a currently verified exact stored session through the existing profile and resume route', async () => {
    $sessions.set([
      { id: 'stored-work', profile: 'default', title: 'Default' },
      { id: 'stored-work', profile: 'work', title: 'Work' }
    ] as never)
    setClarifyRequest({
      choices: null,
      profile: 'work',
      question: 'Continue?',
      requestId: 'req',
      sessionId: 'runtime',
      storedSessionId: 'stored-work'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-open-session', itemId })

    expect(ensureProfile).toHaveBeenCalledWith('work')
    expect(resumeSession).toHaveBeenCalledWith('work', 'stored-work')
    expect($petActionCenter.get().action).toEqual({ status: 'success', itemId })
  })

  it('reports open-failed when the exact resume adapter cannot confirm success', async () => {
    resumeSession.mockResolvedValueOnce(false)
    $sessions.set([{ id: 'stored-work', profile: 'work', title: 'Work' }] as never)
    setClarifyRequest({
      choices: null,
      profile: 'work',
      question: 'Continue?',
      requestId: 'req',
      sessionId: 'runtime',
      storedSessionId: 'stored-work'
    })
    const itemId = $petActionCenter.get().items[0]!.id

    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-open-session', itemId })

    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'open-failed' })
  })

  it('does not resume when the represented item vanishes during the awaited profile switch', async () => {
    const switching = pending<void>()

    ensureProfile.mockImplementationOnce(() => switching.promise)
    $sessions.set([{ id: 'stored-work', profile: 'work', title: 'Work' }] as never)
    setClarifyRequest({
      choices: null,
      profile: 'work',
      question: 'Continue?',
      requestId: 'req',
      sessionId: 'runtime',
      storedSessionId: 'stored-work'
    })
    const itemId = $petActionCenter.get().items[0]!.id
    const action = createPetActionCenterActions(dependencies).handle({ type: 'action-center-open-session', itemId })

    clearClarifyRequest({ profile: 'work', requestId: 'req', sessionId: 'runtime' })
    switching.resolve()
    await action

    expect(resumeSession).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'session-unverified' })
  })

  it('refuses an open action after the stored session loses verification', async () => {
    $sessions.set([{ id: 'stored-work', profile: 'work', title: 'Work' }] as never)
    setApprovalRequest({
      command: 'npm test',
      description: 'Run tests',
      profile: 'work',
      sessionId: 'runtime',
      storedSessionId: 'stored-work'
    })
    const itemId = $petActionCenter.get().items[0]!.id
    $sessions.set([])

    await createPetActionCenterActions(dependencies).handle({ type: 'action-center-open-session', itemId })

    expect(ensureProfile).not.toHaveBeenCalled()
    expect(resumeSession).not.toHaveBeenCalled()
    expect($petActionCenter.get().action).toEqual({ status: 'error', itemId, errorCode: 'session-unverified' })
  })
})
