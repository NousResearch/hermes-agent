import { describe, expect, it } from 'vitest'

import { googleWizardStep } from './google-wizard-step'

describe('googleWizardStep', () => {
  it('resumes where a half-finished attempt stopped instead of starting over', () => {
    expect(googleWizardStep(null)).toBe('client')
    expect(googleWizardStep({ client_secret: false, token: false })).toBe('client')
    expect(googleWizardStep({ client_secret: true, token: false })).toBe('consent')
    expect(googleWizardStep({ client_secret: true, token: true })).toBe('verify')
    // A token without its client file cannot refresh: back to the first step.
    expect(googleWizardStep({ client_secret: false, token: true })).toBe('client')
  })
})
