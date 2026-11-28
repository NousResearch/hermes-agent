import { describe, expect, it } from 'vitest'

import { portalAgentSwatch } from './portal-profile-color'
import { PROFILE_SWATCHES, profileColor } from './profile-color'

const PRIMARY_HUES = ['#f20000', '#f2ca00', '#00f279', '#0000f2', '#a252ee']

describe('portalAgentSwatch', () => {
  it('draws the default agent and unreadable colours in the brand primary', () => {
    expect(portalAgentSwatch(null)).toEqual({ hue: '#0000f2', solid: '#0000f2' })
    expect(portalAgentSwatch('var(--accent)')).toEqual({ hue: '#0000f2', solid: '#0000f2' })
    expect(portalAgentSwatch('#808080').hue).toBe('#0000f2')
  })

  it('snaps every picker swatch and hashed profile hue to one of the five Figma primaries', () => {
    for (const color of [...PROFILE_SWATCHES, profileColor('ops')!, profileColor('research')!]) {
      expect(PRIMARY_HUES).toContain(portalAgentSwatch(color).hue)
    }
  })

  it('reads hex picks by hue', () => {
    expect(portalAgentSwatch('#22c55e').hue).toBe('#00f279')
    expect(portalAgentSwatch('#e11').hue).toBe('#f20000')
    expect(portalAgentSwatch('#7c3aed').hue).toBe('#a252ee')
  })
})
