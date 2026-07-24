import { describe, expect, it } from 'vitest'

import {
  composerDockCard,
  composerFill,
  composerInputSurface,
  composerPanelCard
} from '@/components/chat/composer-dock'

const hasBackdropFilter = (classes: string) =>
  classes.includes('backdrop-blur') || classes.includes('backdrop-saturate') || classes.includes('backdrop-filter')

describe('composer surface treatments', () => {
  it('keeps the frequently repainting input surface free of backdrop filters', () => {
    expect(composerInputSurface).toContain(composerFill)
    expect(hasBackdropFilter(composerInputSurface)).toBe(false)
  })

  it('retains the glass treatment on non-input composer chrome', () => {
    expect(hasBackdropFilter(composerDockCard())).toBe(true)
    expect(hasBackdropFilter(composerPanelCard)).toBe(true)
  })
})
