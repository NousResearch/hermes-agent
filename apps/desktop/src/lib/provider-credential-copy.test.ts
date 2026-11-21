import { describe, expect, it } from 'vitest'

import { connectedCredentialLine, credentialRemoveIsCliManaged } from './provider-credential-copy'

const SIGN_IN = 'Sign in once in your terminal, then come back to chat'

describe('connected credential copy', () => {
  it('names an env-var API key instead of a pending CLI sign-in', () => {
    expect(
      connectedCredentialLine(
        { logged_in: true, source: 'env_var', source_label: 'ANTHROPIC_API_KEY (.env)' },
        SIGN_IN
      )
    ).toBe('ANTHROPIC_API_KEY (.env)')
    expect(credentialRemoveIsCliManaged({ source: 'env_var' })).toBe(false)
  })

  it('keeps the sign-in line for a CLI-owned login and for a logged-out row', () => {
    expect(connectedCredentialLine({ logged_in: true, source: 'hermes_pkce' }, SIGN_IN)).toBe(SIGN_IN)
    expect(connectedCredentialLine({ logged_in: false, source: 'env_var', source_label: 'ANTHROPIC_API_KEY' }, SIGN_IN)).toBe(
      SIGN_IN
    )
    expect(credentialRemoveIsCliManaged({ source: 'hermes_pkce' })).toBe(true)
    expect(credentialRemoveIsCliManaged(undefined)).toBe(true)
  })
})
