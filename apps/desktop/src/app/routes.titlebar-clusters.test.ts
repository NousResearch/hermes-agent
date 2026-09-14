import { describe, expect, it } from 'vitest'

import { hidesFixedTitlebarClusters, isOverlayView } from './routes'

describe('hidesFixedTitlebarClusters', () => {
  it('hides clusters on contributed full pages and overlays', () => {
    expect(hidesFixedTitlebarClusters('extension')).toBe(true)
    expect(hidesFixedTitlebarClusters('settings')).toBe(true)
  })

  it('keeps clusters on chat and first-party workspace pages', () => {
    expect(hidesFixedTitlebarClusters('chat')).toBe(false)
    expect(hidesFixedTitlebarClusters('skills')).toBe(false)
    expect(hidesFixedTitlebarClusters('messaging')).toBe(false)
    expect(hidesFixedTitlebarClusters('artifacts')).toBe(false)
  })

  it('does not treat extension as an overlay', () => {
    expect(isOverlayView('extension')).toBe(false)
  })
})
