import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import { $activeGatewayProfile } from './profile'
import {
  $activeSessionAwaitingInput,
  $approvalRequest,
  $approvalRequests,
  $secretRequest,
  $sudoRequest,
  $sudoRequests,
  clearAllPrompts,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from './prompts'
import { $activeSessionId } from './session'

// Prompts are parked per profile/session identity; these tests use default
// unless they explicitly exercise cross-profile behavior.
beforeEach(() => {
  $activeGatewayProfile.set('default')
  $activeSessionId.set('s1')
})

afterEach(() => {
  clearAllPrompts()
  clearClarifyRequest()
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
})

describe('profile-safe prompt identity', () => {
  it('stamps a stable local request identity and receive time at ingest', () => {
    const before = Date.now()

    setApprovalRequest({ command: 'test', description: 'metadata', profile: 'default', sessionId: 'runtime' })

    const stored = Object.values($approvalRequests.get())[0]

    expect(stored?.receivedAt).toBeGreaterThanOrEqual(before)
    expect(stored?.requestIdentity).toEqual(expect.any(String))
    expect(stored?.requestIdentity).not.toBe('')
  })

  it('keeps the same runtime session id independent across profiles', () => {
    setApprovalRequest({ command: 'default command', description: 'default', profile: 'default', sessionId: 'shared' })
    setApprovalRequest({ command: 'work command', description: 'work', profile: ' work ', sessionId: 'shared' })

    expect(Object.values($approvalRequests.get()).map(request => request.command)).toEqual([
      'default command',
      'work command'
    ])
  })

  it('derives the active request from active profile and active runtime session', () => {
    setApprovalRequest({ command: 'default command', description: 'default', profile: 'default', sessionId: 'shared' })
    setApprovalRequest({ command: 'work command', description: 'work', profile: 'work', sessionId: 'shared' })
    $activeSessionId.set('shared')

    $activeGatewayProfile.set('default')
    expect($approvalRequest.get()?.command).toBe('default command')

    $activeGatewayProfile.set('work')
    expect($approvalRequest.get()?.command).toBe('work command')
  })

  it('clears only the exact profile and session identity', () => {
    setApprovalRequest({ command: 'default command', description: 'default', profile: 'default', sessionId: 'shared' })
    setApprovalRequest({ command: 'work command', description: 'work', profile: 'work', sessionId: 'shared' })

    clearApprovalRequest({ profile: 'default', sessionId: 'shared' })

    expect(Object.values($approvalRequests.get())).toEqual([
      expect.objectContaining({ command: 'work command', profile: 'work', sessionId: 'shared' })
    ])
  })

  it('does not let a stale request id clear a replacement in the same identity', () => {
    setSudoRequest({ profile: 'work', requestId: 'replacement', sessionId: 'shared' })

    clearSudoRequest({ profile: 'work', requestId: 'stale', sessionId: 'shared' })

    expect(Object.values($sudoRequests.get())).toEqual([
      expect.objectContaining({ profile: 'work', requestId: 'replacement', sessionId: 'shared' })
    ])
  })
})

describe('approval prompt store', () => {
  it('holds the active session-keyed approval request', () => {
    setApprovalRequest({
      command: 'rm -rf /tmp/x',
      description: 'recursive delete',
      profile: 'default',
      sessionId: 's1'
    })

    expect($approvalRequest.get()).toEqual(
      expect.objectContaining({
        command: 'rm -rf /tmp/x',
        description: 'recursive delete',
        profile: 'default',
        sessionId: 's1'
      })
    )
  })

  it('parks a background session prompt out of the active view', () => {
    setApprovalRequest({ command: 'x', description: 'd', profile: 'default', sessionId: 's2' })

    // Not visible while s1 is focused …
    expect($approvalRequest.get()).toBeNull()

    // … but surfaces once the user switches to the session that raised it.
    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.sessionId).toBe('s2')
  })

  it('clears the active session prompt', () => {
    setApprovalRequest({ command: 'x', description: 'd', profile: 'default', sessionId: 's1' })
    clearApprovalRequest({ profile: 'default', sessionId: 's1' })

    expect($approvalRequest.get()).toBeNull()
  })

  it('carries allowPermanent so the bar can hide "Always allow"', () => {
    setApprovalRequest({
      allowPermanent: false,
      command: 'curl x | bash',
      description: 'content-security',
      profile: 'default',
      sessionId: 's1'
    })

    expect($approvalRequest.get()?.allowPermanent).toBe(false)
  })
})

describe('sudo prompt store', () => {
  it('clears only when the request id matches the in-flight prompt', () => {
    setSudoRequest({ profile: 'default', requestId: 'abc', sessionId: 's1' })

    // A stale clear for a different request must NOT drop the live prompt —
    // otherwise a late response to a prior sudo ask would dismiss the current
    // one and leave the agent blocked.
    clearSudoRequest({ profile: 'default', requestId: 'stale', sessionId: 's1' })
    expect($sudoRequest.get()).toEqual(
      expect.objectContaining({ profile: 'default', requestId: 'abc', sessionId: 's1' })
    )

    clearSudoRequest({ profile: 'default', requestId: 'abc', sessionId: 's1' })
    expect($sudoRequest.get()).toBeNull()
  })

  it('clears unconditionally when no request id is given', () => {
    setSudoRequest({ profile: 'default', requestId: 'abc', sessionId: 's1' })
    clearSudoRequest({ profile: 'default', sessionId: 's1' })

    expect($sudoRequest.get()).toBeNull()
  })
})

describe('secret prompt store', () => {
  it('carries env var and prompt, and clears on id match', () => {
    setSecretRequest({
      requestId: 'r1',
      envVar: 'OPENAI_API_KEY',
      profile: 'default',
      prompt: 'Paste your key',
      sessionId: 's1'
    })

    expect($secretRequest.get()).toEqual(
      expect.objectContaining({
        requestId: 'r1',
        envVar: 'OPENAI_API_KEY',
        profile: 'default',
        prompt: 'Paste your key',
        sessionId: 's1'
      })
    )

    clearSecretRequest({ profile: 'default', requestId: 'mismatch', sessionId: 's1' })
    expect($secretRequest.get()).not.toBeNull()

    clearSecretRequest({ profile: 'default', requestId: 'r1', sessionId: 's1' })
    expect($secretRequest.get()).toBeNull()
  })
})

describe('clearAllPrompts', () => {
  it('drops every kind for one session at once (turn end / interrupt)', () => {
    setApprovalRequest({ command: 'x', description: 'd', profile: 'default', sessionId: 's1' })
    setSudoRequest({ profile: 'default', requestId: 'abc', sessionId: 's1' })
    setSecretRequest({ requestId: 'r1', envVar: 'E', profile: 'default', prompt: 'p', sessionId: 's1' })

    clearAllPrompts({ profile: 'default', sessionId: 's1' })

    expect($approvalRequest.get()).toBeNull()
    expect($sudoRequest.get()).toBeNull()
    expect($secretRequest.get()).toBeNull()
  })

  it('leaves other sessions parked prompts intact', () => {
    setApprovalRequest({ command: 'x', description: 'd', profile: 'default', sessionId: 's1' })
    setApprovalRequest({ command: 'y', description: 'e', profile: 'default', sessionId: 's2' })

    clearAllPrompts({ profile: 'default', sessionId: 's1' })

    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.command).toBe('y')
  })
})

describe('$activeSessionAwaitingInput', () => {
  it('is true while any blocking prompt (clarify or approval/sudo/secret) is parked on the active session', () => {
    expect($activeSessionAwaitingInput.get()).toBe(false)

    setApprovalRequest({ command: 'x', description: 'd', profile: 'default', sessionId: 's1' })
    expect($activeSessionAwaitingInput.get()).toBe(true)

    clearApprovalRequest({ profile: 'default', sessionId: 's1' })
    expect($activeSessionAwaitingInput.get()).toBe(false)

    setClarifyRequest({ choices: null, profile: 'default', question: 'q', requestId: 'c1', sessionId: 's1' })
    expect($activeSessionAwaitingInput.get()).toBe(true)
  })

  it('ignores a prompt parked on a background session', () => {
    setSudoRequest({ profile: 'default', requestId: 'r', sessionId: 's2' })
    expect($activeSessionAwaitingInput.get()).toBe(false)

    $activeSessionId.set('s2')
    expect($activeSessionAwaitingInput.get()).toBe(true)
  })
})
