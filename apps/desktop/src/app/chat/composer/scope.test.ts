import { describe, expect, it } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from './scope'

describe('MAIN_COMPOSER_SCOPE', () => {
  it('keeps the main composer fixed to the bottom dock', () => {
    expect(MAIN_COMPOSER_SCOPE.popoutAllowed).toBe(false)
  })
})
