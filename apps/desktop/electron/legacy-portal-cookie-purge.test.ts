import { expect, test } from 'vitest'

import { purgeLegacyPortalCookiesOnce } from './legacy-portal-cookie-purge'

function harness(clear: () => Promise<boolean>, markerExists = false) {
  const calls: string[] = []

  return {
    calls,
    run: () =>
      purgeLegacyPortalCookiesOnce({
        markerExists: () => markerExists,
        warm: async () => void calls.push('warm'),
        clearCookies: async () => {
          calls.push('clear')

          return clear()
        },
        writeMarker: () => void calls.push('marker')
      })
  }
}

test('a complete purge warms the jar first, then records the marker', async () => {
  const h = harness(async () => true)

  await expect(h.run()).resolves.toBe(true)
  expect(h.calls).toEqual(['warm', 'clear', 'marker'])
})

test('a partial or failed purge writes no marker, so the next launch retries', async () => {
  for (const clear of [async () => false, async () => Promise.reject(new Error('jar unavailable'))]) {
    const h = harness(clear)

    await expect(h.run()).resolves.toBe(false)
    expect(h.calls).toEqual(['warm', 'clear'])
  }
})

test('an existing marker skips the purge entirely', async () => {
  const h = harness(async () => true, true)

  await expect(h.run()).resolves.toBe(true)
  expect(h.calls).toEqual([])
})
