import { describe, expect, it } from 'vitest'

import { goneSessionVerdict } from './utils'

describe('goneSessionVerdict', () => {
  it('retries transient 404s while interrupted-turn reconciliation is active', () => {
    expect(
      goneSessionVerdict({
        createdThisRun: false,
        stillListed: false,
        switchInFlight: false,
        reconciliationInFlight: true,
      }),
    ).toBe('retry')
  })
})