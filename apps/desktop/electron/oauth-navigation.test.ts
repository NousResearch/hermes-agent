import { describe, expect, it } from 'vitest'

import { isExpectedOauthNavigationAbort } from './oauth-navigation'

describe('isExpectedOauthNavigationAbort', () => {
  it('treats Electron ERR_ABORTED callback navigation as non-fatal', () => {
    const error = Object.assign(new Error("ERR_ABORTED (-3) loading 'https://agent.example.com/auth/callback?code=redacted'"), {
      code: -3
    })

    expect(isExpectedOauthNavigationAbort(error)).toBe(true)
  })

  it('does not suppress genuine gateway load failures', () => {
    expect(isExpectedOauthNavigationAbort(new Error('ERR_NAME_NOT_RESOLVED (-105)'))).toBe(false)
    expect(isExpectedOauthNavigationAbort(new Error('HTTP 502 from gateway'))).toBe(false)
  })
})
