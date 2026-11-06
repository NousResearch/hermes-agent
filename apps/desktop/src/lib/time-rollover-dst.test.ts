import { afterEach, expect, it, vi } from 'vitest'

import { nominalDayStart } from './time'

afterEach(() => vi.unstubAllEnvs())

it.each([
  [2, 8],
  [10, 1]
])('keeps the 4 AM boundary across DST on 2026/%i/%i (zero-based month)', (month, day) => {
  vi.stubEnv('TZ', 'America/New_York')

  expect(nominalDayStart(new Date(2026, month, day, 3, 30).getTime())).toBe(new Date(2026, month, day - 1).getTime())
  expect(nominalDayStart(new Date(2026, month, day, 4, 30).getTime())).toBe(new Date(2026, month, day).getTime())
})
