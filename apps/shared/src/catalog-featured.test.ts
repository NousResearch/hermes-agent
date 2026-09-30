import { describe, expect, it } from 'vitest'

import { type FeaturedCandidate, pickFeatured } from './catalog-featured'

const WEEK = 7 * 24 * 60 * 60 * 1000
const NOW = Date.parse('2026-09-30T12:00:00Z')

type Row = FeaturedCandidate & { name: string }

const pick = (rows: Row[], now = NOW) => pickFeatured(rows, row => row, now)?.name

describe('pickFeatured', () => {
  it('rotates through curated ranks weekly, in rank order', () => {
    const rows: Row[] = [
      { name: 'third', featured: 3, official: true, pictured: false },
      { name: 'first', featured: 1, official: false, pictured: false },
      { name: 'uncurated', official: true, pictured: true, stars: 999 },
      { name: 'second', featured: 2, official: true, pictured: true }
    ]

    const week = Math.floor(NOW / WEEK)
    const order = ['first', 'second', 'third']

    expect(pick(rows)).toBe(order[week % 3])
    expect(pick(rows, NOW + WEEK)).toBe(order[(week + 1) % 3])
    expect(pick(rows, NOW + 2 * WEEK)).toBe(order[(week + 2) % 3])
  })

  it('is the same pick anywhere within one week', () => {
    const rows: Row[] = [1, 2, 3, 4].map(rank => ({ name: `r${rank}`, featured: rank, official: true, pictured: true }))
    const start = Math.floor(NOW / WEEK) * WEEK

    expect(pick(rows, start)).toBe(pick(rows, start + WEEK - 1))
  })

  it('falls back to the newest pictured official entry of the last 30 days', () => {
    const rows: Row[] = [
      { name: 'old', official: true, pictured: true, addedAt: '2026-06-01T00:00:00Z', stars: 900 },
      { name: 'new-plain', official: true, pictured: false, addedAt: '2026-09-29T00:00:00Z' },
      { name: 'new-pictured', official: true, pictured: true, addedAt: '2026-09-20T00:00:00Z' },
      { name: 'community', official: false, pictured: true, addedAt: '2026-09-29T00:00:00Z' }
    ]

    expect(pick(rows)).toBe('new-pictured')
  })

  it('then to an official entry with art, most stars first, never a community one', () => {
    const rows: Row[] = [
      { name: 'popular-community', official: false, pictured: true, stars: 5000 },
      { name: 'official-small', official: true, pictured: false, stars: 10 },
      { name: 'official-big', official: true, pictured: false, stars: 40 }
    ]

    expect(pick(rows)).toBe('official-big')
    expect(pick([...rows, { name: 'official-art', official: true, pictured: true, stars: 1 }])).toBe('official-art')
    expect(pick([{ name: 'community', official: false, pictured: true }])).toBeUndefined()
  })
})
