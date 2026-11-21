import { describe, expect, it } from 'vitest'

import { signInFailedAt, signUpHref, visitorAccessFrom, withoutSignInOutcome } from './visitor-access'

const page = 'https://hermes.example/#/chat'

describe('visitor access', () => {
  it('is open on every page served today, where nothing names a sign up', () => {
    expect(visitorAccessFrom({ nativeApp: false, pageUrl: page, signUpUrl: undefined })).toEqual({ kind: 'open' })
  })

  it('requires sign up when the serving page names where it starts', () => {
    expect(visitorAccessFrom({ nativeApp: false, pageUrl: page, signUpUrl: 'https://portal.example/sign-up' })).toEqual(
      { kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' }
    )
    expect(visitorAccessFrom({ nativeApp: false, pageUrl: page, signUpUrl: '/start' })).toEqual({
      kind: 'sign_up_required',
      signUpUrl: 'https://hermes.example/start'
    })
  })

  it('never asks Hermes on your own computer to sign up, whatever the page says', () => {
    expect(visitorAccessFrom({ nativeApp: true, pageUrl: page, signUpUrl: 'https://portal.example/sign-up' })).toEqual({
      kind: 'open'
    })
  })

  it('ignores a sign-up address that is not a web page', () => {
    for (const signUpUrl of ['', '   ', 'javascript:alert(1)', 42, { url: 'x' }]) {
      expect(visitorAccessFrom({ nativeApp: false, pageUrl: page, signUpUrl })).toEqual({ kind: 'open' })
    }
  })
})

describe('the way to sign up and back', () => {
  it('comes back to the same place, without a previous attempt’s outcome', () => {
    const href = signUpHref('https://portal.example/sign-up?plan=plus', 'https://hermes.example/?hermes_sign_in=failed#/chat')
    const url = new URL(href)

    expect(url.origin + url.pathname).toBe('https://portal.example/sign-up')
    expect(url.searchParams.get('plan')).toBe('plus')
    expect(url.searchParams.get('return_to')).toBe('https://hermes.example/#/chat')
  })

  it('reads a failed sign in from the address it came back to, and can take it off', () => {
    expect(signInFailedAt('https://hermes.example/?hermes_sign_in=failed#/chat')).toBe(true)
    expect(signInFailedAt('https://hermes.example/#/chat')).toBe(false)
    expect(withoutSignInOutcome('https://hermes.example/?a=1&hermes_sign_in=failed#/chat')).toBe(
      'https://hermes.example/?a=1#/chat'
    )
  })
})
