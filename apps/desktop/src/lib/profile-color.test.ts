import { describe, expect, it } from 'vitest'

import { assignProfileColors, profileColor, resolveProfileColor } from './profile-color'

describe('profile colors', () => {
  it('assigns distinct deterministic colors to names that collide in the legacy hash', () => {
    const colors = assignProfileColors(['worker-10', 'worker-104'], {})

    expect(colors['worker-10']).not.toBe(colors['worker-104'])
    expect(colors).toEqual(assignProfileColors(['worker-104', 'worker-10'], {}))
  })

  it('keeps a persisted override ahead of the allocated palette', () => {
    expect(assignProfileColors(['worker-10', 'worker-104'], { 'worker-104': '#e91e63' })['worker-104']).toBe('#e91e63')
  })

  it('keeps the canonical default neutral even when its display name changes', () => {
    expect(profileColor('default')).toBeNull()
    expect(resolveProfileColor('default', { default: '#e91e63' })).toBeNull()
  })
})
