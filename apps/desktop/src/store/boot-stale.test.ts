import { describe, expect, it } from 'vitest'

import { shouldDismissStaleBootFailure } from './boot'

describe('stale backend start dialog', () => {
  it('clears a superseded or timeout dialog once the replacement gateway is ready', () => {
    const ready = { phase: 'backend.ready', running: true, error: null }

    expect(
      shouldDismissStaleBootFailure(
        'Hermes backend start was superseded by a newer connection attempt.',
        ready
      )
    ).toBe(true)
    expect(shouldDismissStaleBootFailure('Timed out connecting to Hermes backend', ready)).toBe(true)
    expect(shouldDismissStaleBootFailure('The disk is full, so Hermes could not start.', ready)).toBe(false)
    expect(
      shouldDismissStaleBootFailure('Timed out connecting to Hermes backend', {
        phase: 'backend.error',
        running: false,
        error: 'still failing'
      })
    ).toBe(false)
  })
})
