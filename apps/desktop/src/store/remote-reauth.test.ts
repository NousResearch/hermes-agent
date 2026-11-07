import { afterEach, describe, expect, it } from 'vitest'

import {
  $remoteReauthExhausted,
  clearRemoteReauthExhausted,
  isRemoteReauthExhaustedFor,
  latchRemoteReauthExhausted,
  remoteReauthOrigin
} from './remote-reauth'

afterEach(() => {
  $remoteReauthExhausted.set({})
})

describe('remote reauth exhaustion latch', () => {
  it('keys the latch by origin, not by path', () => {
    latchRemoteReauthExhausted('https://gateway.example.test/api/status')

    expect(isRemoteReauthExhaustedFor('https://gateway.example.test/sessions')).toBe(true)
    expect(Object.keys($remoteReauthExhausted.get())).toEqual(['https://gateway.example.test'])
  })

  it('is scoped: another gateway origin is not exhausted by this one', () => {
    latchRemoteReauthExhausted('https://gateway.example.test')
    latchRemoteReauthExhausted('https://other.example.test')

    clearRemoteReauthExhausted('https://gateway.example.test')

    expect(isRemoteReauthExhaustedFor('https://gateway.example.test')).toBe(false)
    expect(isRemoteReauthExhaustedFor('https://other.example.test')).toBe(true)
  })

  it('records when the episode started', () => {
    latchRemoteReauthExhausted('https://gateway.example.test', 1_700_000_000_000)

    expect($remoteReauthExhausted.get()['https://gateway.example.test']).toEqual({
      latchedAt: 1_700_000_000_000
    })
  })

  it('an unparsable URL still keys consistently instead of throwing', () => {
    expect(remoteReauthOrigin('not a url')).toBe('not a url')
    latchRemoteReauthExhausted('not a url')
    expect(isRemoteReauthExhaustedFor('not a url')).toBe(true)
  })
})
