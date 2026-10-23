import { describe, expect, it } from 'vitest'

import { shouldLogExpectedNotFoundApiFailure } from './api-request-failure'

describe('shouldLogExpectedNotFoundApiFailure', () => {
  it('keeps a not-found session delete attributable in the desktop log', () => {
    expect(shouldLogExpectedNotFoundApiFailure({ method: 'DELETE', path: '/api/sessions/missing-session' })).toBe(true)
  })

  it('keeps expected not-found read probes quiet', () => {
    expect(shouldLogExpectedNotFoundApiFailure({ method: 'GET', path: '/api/sessions/missing-session' })).toBe(false)
  })
})
