import { beforeEach, describe, expect, it } from 'vitest'

import {
  $userInputRequests,
  clearUserInputRequest,
  replaceUserInputRequests,
  setUserInputRequest
} from './user-input'

const request = (requestId: string, sessionId = 'session-1') => ({
  context: 'Need a decision',
  expiresAt: 0,
  questions: [{
    allowFreeText: false,
    defaultValue: undefined,
    id: 'choice',
    options: ['a', 'b'],
    text: 'Pick one'
  }],
  requestId,
  sessionId,
  status: 'pending' as const,
  turnId: 'turn-1'
})

describe('native user-input store', () => {
  beforeEach(() => {
    $userInputRequests.set({})
  })

  it('deduplicates a request id within its session', () => {
    setUserInputRequest(request('request-1'))
    setUserInputRequest({ ...request('request-1'), context: 'Updated context' })

    expect($userInputRequests.get()['session-1']).toHaveLength(1)
    expect($userInputRequests.get()['session-1'][0].context).toBe('Updated context')
  })

  it('clears only the correlated request and preserves newer requests', () => {
    setUserInputRequest(request('request-1'))
    setUserInputRequest(request('request-2'))
    clearUserInputRequest('session-1', 'request-1')

    expect($userInputRequests.get()['session-1'].map(item => item.requestId)).toEqual(['request-2'])
    clearUserInputRequest('session-1', 'request-1')
    expect($userInputRequests.get()['session-1'].map(item => item.requestId)).toEqual(['request-2'])
  })

  it('replaces pending records on reconnect without cross-session bleed', () => {
    setUserInputRequest(request('old', 'session-1'))
    setUserInputRequest(request('other', 'session-2'))
    replaceUserInputRequests('session-1', [request('replayed', 'session-1')])

    expect($userInputRequests.get()['session-1'].map(item => item.requestId)).toEqual(['replayed'])
    expect($userInputRequests.get()['session-2'].map(item => item.requestId)).toEqual(['other'])
  })
})
