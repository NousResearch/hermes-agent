import { describe, expect, it } from 'vitest'

import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from './escape-layers'

describe('escape-layers', () => {
  it('reports top when nothing is registered', () => {
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.narrowOverlay)).toBe(true)
  })

  it('a lower layer yields to a higher open one, and reclaims top when it closes', () => {
    const releaseOverlay = pushEscapeLayer(ESCAPE_PRIORITY.overlay)

    // Narrow overlay is open under a full-page overlay — it must not act.
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.narrowOverlay)).toBe(false)
    // The overlay itself is top.
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.overlay)).toBe(true)

    releaseOverlay()
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.narrowOverlay)).toBe(true)
  })

  it('equal-or-higher priority counts as top (ties act)', () => {
    const release = pushEscapeLayer(ESCAPE_PRIORITY.zoneEditor)
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.zoneEditor)).toBe(true)
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.layoutEdit)).toBe(false)
    release()
  })

  it('tracks the max across several open layers', () => {
    const releases = [
      pushEscapeLayer(ESCAPE_PRIORITY.narrowOverlay),
      pushEscapeLayer(ESCAPE_PRIORITY.layoutEdit),
      pushEscapeLayer(ESCAPE_PRIORITY.zoneEditor)
    ]

    expect(isTopEscapeLayer(ESCAPE_PRIORITY.zoneEditor)).toBe(true)
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.layoutEdit)).toBe(false)

    // Close the zone editor — layout edit becomes top.
    releases[2]()
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.layoutEdit)).toBe(true)

    releases.forEach(release => release())
    expect(isTopEscapeLayer(ESCAPE_PRIORITY.narrowOverlay)).toBe(true)
  })
})
