import { expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import {
  forgetMediaImageDimensions,
  getMediaImageDimensions,
  mediaImageKey,
  rememberMediaImageDimensions,
  validImageDimensions
} from './media'

it('shares proven path aliases only within an owner and bounds regenerated metadata by LRU', () => {
  const a = { connectionId: 'a', profile: 'work', mode: 'remote' } as HermesConnection
  const b = { ...a, connectionId: 'b' }
  const key = mediaImageKey('/images/a b.png', a)
  const dimensions = { width: 900, height: 600 }

  rememberMediaImageDimensions(key, dimensions.width, dimensions.height)
  expect(getMediaImageDimensions(mediaImageKey('file:///images/a%20b.png', a))).toEqual(dimensions)
  expect(getMediaImageDimensions(mediaImageKey('/images/a b.png', b))).toBeUndefined()
  expect(getMediaImageDimensions(mediaImageKey('/images/a b.png', { ...a, profile: 'personal' }))).toBeUndefined()
  expect(getMediaImageDimensions(mediaImageKey('/images/a b.png?revision=2', a))).toBeUndefined()
  expect(getMediaImageDimensions(mediaImageKey('file:///images/a%20b.png?revision=2', a))).toBeUndefined()
  expect(getMediaImageDimensions(mediaImageKey('/images/a b.png', b, a))).toEqual(dimensions)

  // Discover the finite capacity behaviorally, not by pinning its current value.
  const cold = mediaImageKey('/images/old.png', a)
  rememberMediaImageDimensions(cold, 100, 100)

  for (let i = 0; i < 10000; i++) {
    rememberMediaImageDimensions(`bounded-${i}`, 100, 100)
    expect(getMediaImageDimensions(key)).toEqual(dimensions)
  }

  expect(getMediaImageDimensions(cold)).toBeUndefined()
  expect(validImageDimensions(0, 100)).toBeUndefined()
  expect(validImageDimensions(Infinity, 100)).toBeUndefined()
  rememberMediaImageDimensions(key, 0, 100)
  expect(getMediaImageDimensions(key)).toEqual(dimensions)
  rememberMediaImageDimensions('x'.repeat(100000), 100, 100)
  expect(getMediaImageDimensions('x'.repeat(100000))).toBeUndefined()
  forgetMediaImageDimensions(key)
  expect(getMediaImageDimensions(key)).toBeUndefined()
})

it('keeps multi-megabyte inline sources to a small, distinct key', () => {
  const a = { connectionId: 'a', profile: 'work', mode: 'local' } as HermesConnection
  const inline = (tail: string) => `data:image/png;base64,${'A'.repeat(4_000_000)}${tail}`

  expect(mediaImageKey(inline('x'), a).length).toBeLessThan(1024)
  expect(mediaImageKey(inline('x'), a)).toBe(mediaImageKey(inline('x'), a))
  expect(mediaImageKey(inline('x'), a)).not.toBe(mediaImageKey(inline('y'), a))
})
