import { describe, expect, it } from 'vitest'

import { isExpectedOauthNavigationAbort, isWakingAgentLoadFailure } from './oauth-navigation'

describe('isExpectedOauthNavigationAbort', () => {
  it('treats Electron ERR_ABORTED callback navigation as non-fatal', () => {
    const error = Object.assign(
      new Error("ERR_ABORTED (-3) loading 'https://agent.example.com/auth/callback?code=redacted'"),
      {
        code: -3
      }
    )

    expect(isExpectedOauthNavigationAbort(error)).toBe(true)
  })

  it('does not suppress genuine gateway load failures', () => {
    expect(isExpectedOauthNavigationAbort(new Error('ERR_NAME_NOT_RESOLVED (-105)'))).toBe(false)
    expect(isExpectedOauthNavigationAbort(new Error('HTTP 502 from gateway'))).toBe(false)
  })
})

describe('isWakingAgentLoadFailure', () => {
  it('retries what a resuming agent answers', () => {
    expect(isWakingAgentLoadFailure(new Error("ERR_CONNECTION_RESET (-101) loading 'https://a.example.com/'"))).toBe(
      true
    )
    expect(isWakingAgentLoadFailure(new Error('ERR_EMPTY_RESPONSE (-324)'))).toBe(true)
  })

  it('fails fast on errors a wake cannot fix', () => {
    expect(isWakingAgentLoadFailure(new Error('ERR_NAME_NOT_RESOLVED (-105)'))).toBe(false)
    expect(isWakingAgentLoadFailure(new Error('ERR_CERT_AUTHORITY_INVALID (-202)'))).toBe(false)
  })
})
