import { describe, expect, it } from 'vitest'

import { selectDecision } from './vault-guard'

describe('selectDecision', () => {
  it('never lets unsaved text be dropped by switching or closing without asking', () => {
    expect(selectDecision(true, 'a.md', 'b.md')).toBe('ask')
    expect(selectDecision(true, 'a.md', null)).toBe('ask')
  })

  it('switches freely when nothing is unsaved, and ignores re-selecting the open note', () => {
    expect(selectDecision(false, 'a.md', 'b.md')).toBe('switch')
    expect(selectDecision(true, 'a.md', 'a.md')).toBe('noop')
  })
})
