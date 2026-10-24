import { describe, expect, it } from 'vitest'

import { getPlayerService, isPlayerSource, PLAYER_SERVICES } from './player-services'

describe('player service catalog', () => {
  it('keeps the existing YouTube source identifiers and adds the requested experimental services', () => {
    expect(PLAYER_SERVICES.map(service => service.id)).toEqual([
      'youtube',
      'tv',
      'netflix',
      'prime-video',
      'disney-plus',
      'hulu',
      'hbo-max',
      'apple-tv',
      'peacock',
      'paramount-plus'
    ])
    expect(getPlayerService('youtube').homeUrl).toBe('https://www.youtube.com/')
    expect(getPlayerService('tv').homeUrl).toBe('https://tv.youtube.com/')
    expect(PLAYER_SERVICES.filter(service => service.validation === 'experimental')).toHaveLength(8)
  })

  it('returns only known sources and marks every new provider as Widevine-dependent but experimental', () => {
    expect(isPlayerSource('youtube')).toBe(true)
    expect(isPlayerSource('tv')).toBe(true)
    expect(isPlayerSource('not-a-service')).toBe(false)
    expect(isPlayerSource(null)).toBe(false)

    for (const service of PLAYER_SERVICES.filter(row => row.validation === 'experimental')) {
      expect(service.requiresWidevine).toBe(true)
      expect(new URL(service.homeUrl).protocol).toBe('https:')
    }
  })
})
