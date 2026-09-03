import { describe, expect, it } from 'vitest'

import source from './index.tsx?raw'

describe('Team Hermes workspace frame', () => {
  it('boxes the pane tree with one responsive theme-native edge', () => {
    expect(source).toContain('data-hermes-workspace-frame')
    expect(source).toContain('overflow-hidden rounded-[0.875rem] border')
    expect(source).toContain('var(--ui-accent,#6e9fc5)_18%')
    expect(source).toContain('inset_0_1px_0')
    expect(source).toContain('max-[44rem]:rounded-[0.625rem]')
  })
})
