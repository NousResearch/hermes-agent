import { describe, expect, it } from 'vitest'

import { hidesFixedTitlebarClusters } from './routes'

describe('hidesFixedTitlebarClusters', () => {
  it('hides clusters on contributed full pages and overlays', () => {
    expect(hidesFixedTitlebarClusters('extension')).toBe(true)
    expect(hidesFixedTitlebarClusters('settings')).toBe(true)
  })

  it('keeps clusters on chat and first-party workspace pages', () => {
    expect(hidesFixedTitlebarClusters('chat')).toBe(false)
  })
})
