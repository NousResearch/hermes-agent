import { describe, expect, it } from 'vitest'

import { ENUM_OPTIONS, FIELD_DESCRIPTIONS } from './constants'

describe('in-app update local changes setting', () => {
  it('offers and explains the fail-closed abort policy', () => {
    expect(ENUM_OPTIONS['updates.non_interactive_local_changes']).toEqual([
      'stash',
      'discard',
      'abort'
    ])
    expect(FIELD_DESCRIPTIONS['updates.nonInteractiveLocalChanges']).toContain('abort')
  })
})
