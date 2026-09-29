import { describe, expect, it } from 'vitest'

import { admitSound, SOUND_RATE_LIMIT, SOUND_RATE_WINDOW_MS } from './ui-sound'

describe('admitSound', () => {
  it('lets a burst through only up to the limit, then again once the window has passed', () => {
    const recent: number[] = []
    const burst = Array.from({ length: SOUND_RATE_LIMIT + 3 }, (_, i) => admitSound(recent, 100 + i))

    expect(burst.filter(Boolean)).toHaveLength(SOUND_RATE_LIMIT)
    expect(admitSound(recent, 100 + SOUND_RATE_WINDOW_MS + 20)).toBe(true)
  })
})
