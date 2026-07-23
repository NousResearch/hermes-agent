import { describe, expect, it } from 'vitest'

import type { HermesBackgroundImage } from '@/global'

import {
  backgroundPreferenceForProfile,
  carouselOrder,
  type DesktopBackgroundPreference,
  normalizeBackgroundPreference
} from './background'

const images = (count: number): HermesBackgroundImage[] =>
  Array.from({ length: count }, (_, index) => ({
    fingerprint: String(index),
    id: `image-${index}`,
    name: `${index}.png`,
    url: `hermes-background://image/${index}`
  }))

const preference = (overrides: Partial<DesktopBackgroundPreference> = {}): DesktopBackgroundPreference => ({
  intervalMinutes: 30,
  mode: 'folder',
  rotationAnchorMs: 1_000,
  shuffleSeed: 42,
  sourcePath: '/wallpapers',
  strength: 10,
  ...overrides
})

describe('background carousel', () => {
  it('shows every image once in a deterministic shuffled deck', () => {
    const source = images(5)

    const slots = Array.from(
      { length: source.length },
      (_, slot) => carouselOrder(source, preference(), 1_000 + slot * 30 * 60_000)[0].id
    )

    expect(new Set(slots)).toEqual(new Set(source.map(image => image.id)))
    expect(slots).toEqual(
      Array.from(
        { length: source.length },
        (_, slot) => carouselOrder(source, preference(), 1_000 + slot * 30 * 60_000)[0].id
      )
    )
  })

  it('avoids an immediate repeat when a new shuffled deck begins', () => {
    const source = images(4)
    const interval = 30 * 60_000
    const last = carouselOrder(source, preference(), 1_000 + 3 * interval)[0].id
    const next = carouselOrder(source, preference(), 1_000 + 4 * interval)[0].id

    expect(next).not.toBe(last)
  })

  it('never repeats across deck boundaries for small folders', () => {
    for (let count = 2; count <= 8; count += 1) {
      const source = images(count)
      const interval = 30 * 60_000

      const slots = Array.from(
        { length: count * 12 },
        (_, slot) => carouselOrder(source, preference(), 1_000 + slot * interval)[0].id
      )

      expect(slots.every((id, index) => index === 0 || id !== slots[index - 1])).toBe(true)
    }
  })

  it('catches up from wall time after multiple missed intervals', () => {
    const source = images(3)
    const interval = 30 * 60_000

    expect(carouselOrder(source, preference(), 1_000 + 7 * interval)[0].id).toBe(
      carouselOrder(source, preference(), 1_000 + 7 * interval)[0].id
    )
  })

  it('keeps a one-image folder stable', () => {
    const source = images(1)
    expect(carouselOrder(source, preference(), Number.MAX_SAFE_INTEGER)).toBe(source)
  })

  it('selects a far-future slot without recursive deck growth', () => {
    expect(() => carouselOrder(images(2), preference(), Date.now() + 100 * 365 * 24 * 60 * 60_000)).not.toThrow()
  })
})

describe('background preferences', () => {
  it('normalizes supported intervals and clamps strength', () => {
    expect(
      normalizeBackgroundPreference({
        intervalMinutes: 999,
        mode: 'folder',
        rotationAnchorMs: -1,
        shuffleSeed: 7,
        sourcePath: ' /wallpapers ',
        strength: 140
      })
    ).toEqual({
      intervalMinutes: 30,
      mode: 'folder',
      rotationAnchorMs: 0,
      shuffleSeed: 7,
      sourcePath: '/wallpapers',
      strength: 100
    })
  })

  it('rejects custom modes without a source path', () => {
    expect(normalizeBackgroundPreference({ mode: 'image' })).toBeNull()
  })

  it('inherits the default profile and migrates the legacy disabled state', () => {
    const configured = preference({ mode: 'image', sourcePath: '/wallpaper.png' })
    expect(backgroundPreferenceForProfile({ default: configured }, 'research')).toBe(configured)
    expect(backgroundPreferenceForProfile({}, 'default', false).mode).toBe('none')
  })
})
