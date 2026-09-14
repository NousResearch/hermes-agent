import { describe, expect, it } from 'vitest'

import { hostLabel, parseListings, specLine } from './listing-embed'

// Shaped like what the agent actually emits after a portal sweep.
const REAL = JSON.stringify({
  listings: [
    {
      address: '51 Via Los Altos, Tiburon, CA 94920',
      baths: '3.5',
      beds: 3,
      catches: ['No pets', 'Fireplace decorative only'],
      facts: ['0.65 acre', 'Panoramic bay views', '12-mo lease'],
      images: ['https://images.homes.com/a.jpg', 'https://images.homes.com/b.jpg'],
      links: [
        { label: 'homes.com', url: 'https://www.homes.com/property/51-via-los-altos/' },
        { url: 'https://www.redfin.com/CA/Tiburon/51-Via-Los-Altos-94920/home/1244581' }
      ],
      note: 'Bay plus Belvedere Island plus Richmond Bridge.',
      price: '$12,500/mo',
      size: '4,290 sqft'
    }
  ]
})

describe('parseListings', () => {
  it('normalizes a real sweep payload', () => {
    const [listing] = parseListings(REAL)

    expect(listing.address).toBe('51 Via Los Altos, Tiburon, CA 94920')
    expect(listing.beds).toBe(3)
    // String decimals survive: half-baths are real.
    expect(listing.baths).toBe(3.5)
    expect(listing.images).toHaveLength(2)
    expect(listing.facts).toHaveLength(3)
    expect(listing.catches).toHaveLength(2)
    // A bare URL gets the portal host as its label.
    expect(listing.links[1].label).toBe('redfin.com')
  })

  it('accepts a lone object and a bare array', () => {
    expect(parseListings('{"address":"1 Main St"}')).toHaveLength(1)
    expect(parseListings('[{"address":"1 Main St"},{"address":"2 Main St"}]')).toHaveLength(2)
  })

  it('drops entries with no address — a card needs an identity', () => {
    expect(parseListings('[{"price":"$1"},{"address":"1 Main St"}]')).toHaveLength(1)
  })

  it('returns nothing for non-JSON so the plain code block wins', () => {
    expect(parseListings('not json at all')).toEqual([])
    expect(parseListings('{"address":"x"')).toEqual([])
  })

  it('refuses non-http URLs in images and links', () => {
    const [listing] = parseListings(
      JSON.stringify({
        address: 'x',
        images: ['javascript:alert(1)', 'https://ok.example/a.jpg'],
        links: ['data:text/html,hi', 'https://ok.example/listing']
      })
    )

    expect(listing.images).toEqual(['https://ok.example/a.jpg'])
    expect(listing.links.map(link => link.url)).toEqual(['https://ok.example/listing'])
  })

  it('dedupes repeated images and links', () => {
    const [listing] = parseListings(
      JSON.stringify({
        address: 'x',
        images: ['https://a.example/1.jpg', 'https://a.example/1.jpg'],
        links: ['https://a.example/l', 'https://a.example/l']
      })
    )

    expect(listing.images).toHaveLength(1)
    expect(listing.links).toHaveLength(1)
  })

  it('caps a pathological payload', () => {
    const images = Array.from({ length: 500 }, (_, i) => `https://a.example/${i}.jpg`)
    const [listing] = parseListings(JSON.stringify({ address: 'x', images }))

    expect(listing.images.length).toBeLessThanOrEqual(40)
  })

  it('rejects absurd bed counts but keeps the listing', () => {
    const [listing] = parseListings('{"address":"x","beds":9999,"baths":-2}')

    expect(listing.beds).toBeUndefined()
    expect(listing.baths).toBeUndefined()
    expect(listing.address).toBe('x')
  })

  it('takes a single string where a list was expected', () => {
    const [listing] = parseListings('{"address":"x","facts":"private dock"}')

    expect(listing.facts).toEqual(['private dock'])
  })
})

describe('specLine', () => {
  it('joins only the specs that exist', () => {
    const [full] = parseListings('{"address":"x","beds":3,"baths":2,"size":"2,136 sqft"}')
    const [sparse] = parseListings('{"address":"x","beds":3}')

    expect(specLine(full)).toBe('3 bd · 2 ba · 2,136 sqft')
    expect(specLine(sparse)).toBe('3 bd')
  })
})

describe('hostLabel', () => {
  it('strips www so the host reads as the portal name', () => {
    expect(hostLabel('https://www.homes.com/property/x')).toBe('homes.com')
    expect(hostLabel('https://www.har.com/homedetail/y')).toBe('har.com')
  })
})
