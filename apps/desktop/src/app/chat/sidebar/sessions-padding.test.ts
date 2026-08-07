import { describe, expect, it } from 'vitest'

import { resolveSessionContentClassName } from './sessions-section'

describe('resolveSessionContentClassName', () => {
  it('keeps right scrollbar clearance on non-virtualized wrappers', () => {
    expect(resolveSessionContentClassName('flex overflow-y-auto pr-2.5', false)).toContain('pr-2.5')
  })

  it('removes wrapper right clearance when the virtual scroller owns it', () => {
    const className = resolveSessionContentClassName('flex overflow-y-auto pr-2.5 pb-1.75', true)

    expect(className).not.toContain('pr-2.5')
    expect(className).toContain('overflow-y-visible')
  })
})
