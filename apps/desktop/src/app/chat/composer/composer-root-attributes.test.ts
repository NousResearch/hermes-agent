import { describe, expect, it } from 'vitest'

import { composerRootAttributes } from './composer-root-attributes'

describe('composer root layout attributes', () => {
  it.each([false, true])('exposes the expanded state only for a stacked composer (%s)', stacked => {
    expect(composerRootAttributes(stacked)).toEqual({
      'data-slot': 'composer-root',
      'data-composer-expanded': stacked ? '' : undefined
    })
  })
})
