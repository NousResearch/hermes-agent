import { afterEach, describe, expect, it } from 'vitest'

import {
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  clearAllPrompts,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from './prompts'

afterEach(() => {
  clearAllPrompts()
})

describe('approval prompt store', () => {
  it('holds the most recent session-keyed approval request', () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'recursive delete', sessionId: 's1' })

    expect($approvalRequest.get()).toEqual({
      command: 'rm -rf /tmp/x',
      description: 'recursive delete',
      sessionId: 's1'
    })
  })

  it('clears unconditionally (approval is session-keyed, no request id)', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    clearApprovalRequest()

    expect($approvalRequest.get()).toBeNull()
  })
})

describe('sudo prompt store', () => {
  it('clears only when the request id matches the in-flight prompt', () => {
    setSudoRequest({ requestId: 'abc' })

    // A stale clear for a different request must NOT drop the live prompt —
    // otherwise a late response to a prior sudo ask would dismiss the current
    // one and leave the agent blocked.
    clearSudoRequest('stale')
    expect($sudoRequest.get()).toEqual({ requestId: 'abc' })

    clearSudoRequest('abc')
    expect($sudoRequest.get()).toBeNull()
  })

  it('clears unconditionally when no request id is given', () => {
    setSudoRequest({ requestId: 'abc' })
    clearSudoRequest()

    expect($sudoRequest.get()).toBeNull()
  })
})

describe('secret prompt store', () => {
  it('carries env var and prompt, and clears on id match', () => {
    setSecretRequest({ requestId: 'r1', envVar: 'OPENAI_API_KEY', prompt: 'Paste your key' })

    expect($secretRequest.get()).toEqual({
      requestId: 'r1',
      envVar: 'OPENAI_API_KEY',
      prompt: 'Paste your key'
    })

    clearSecretRequest('mismatch')
    expect($secretRequest.get()).not.toBeNull()

    clearSecretRequest('r1')
    expect($secretRequest.get()).toBeNull()
  })
})

describe('clearAllPrompts', () => {
  it('drops every in-flight prompt at once (turn end / interrupt)', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    setSudoRequest({ requestId: 'abc' })
    setSecretRequest({ requestId: 'r1', envVar: 'E', prompt: 'p' })

    clearAllPrompts()

    expect($approvalRequest.get()).toBeNull()
    expect($sudoRequest.get()).toBeNull()
    expect($secretRequest.get()).toBeNull()
  })
})
