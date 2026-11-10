import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { CatalogEntry, CatalogKind } from './catalog-data'
import { CatalogDiscovery } from './catalog-discovery'

const entry = (name: string, category: string, source = 'community') =>
  ({ id: name, name, source, category, categoryLabel: category, tags: [], search: name, stars: 1, imageUrl: null }) as unknown as CatalogEntry

const sections = (container: HTMLElement) =>
  [...container.querySelectorAll('[data-catalog-section]')].map(node => node.getAttribute('data-catalog-section')).filter(id => id !== 'developer')

const renderDiscovery = (entries: CatalogEntry[], byCategory: boolean, kind: CatalogKind = 'plugins') =>
  render(
    <CatalogDiscovery
      byCategory={byCategory}
      card={item => <article key={item.id}>{item.name}</article>}
      entries={entries}
      kind={kind}
      narrowed={byCategory}
      onCategory={() => {}}
    />
  ).container

describe('multi-category discovery', () => {
  const picked = [entry('m1', 'memory'), entry('t1', 'tools'), entry('o1', 'tools', 'official'), entry('o2', 'memory', 'official')]

  it('gives every picked category its own shelf, small ones included, with no tier shelves', () => {
    expect(sections(renderDiscovery(picked, true)).sort()).toEqual(['memory', 'tools'])
  })

  it('keeps the default page folding small categories and splitting tiers', () => {
    // One official becomes the hero pick; the other keeps the Official shelf.
    expect(sections(renderDiscovery(picked, false))).toEqual(['featured', 'official', 'more'])
  })
})

describe('featured slider', () => {
  const curated = (name: string, featured: number) => ({ ...entry(name, 'tools', 'official'), featured, imageUrl: 'x.png' }) as CatalogEntry

  it('shows one hero bare, and several as a slider with a dot per pick', () => {
    const one = renderDiscovery([curated('a', 1), entry('b', 'tools')], false)
    expect(one.querySelector('[aria-roledescription=carousel]')).toBeNull()

    const many = renderDiscovery([curated('a', 1), curated('b', 2), curated('c', 3)], false)
    expect(many.querySelectorAll('[aria-roledescription=slide]')).toHaveLength(3)
    expect(many.querySelectorAll('.catalog-featured-dot')).toHaveLength(3)
    expect(many.querySelectorAll('[aria-roledescription=slide]:not([inert])')).toHaveLength(1)
  })
})

describe('skill discovery shelves', () => {
  const categoryReads = (categories: number) => {
    let reads = 0

    const entries = Array.from({ length: 2000 }, (_, index) => {
      const category = `c${index % categories}`

      return Object.defineProperty(entry(`s${index}`, category), 'category', { get: () => (reads++, category) })
    })

    renderDiscovery(entries, false, 'skills')

    return reads
  }

  it('reads the catalog about as much with many categories as with few', () => {
    // Same entry count with ten times the categories. A rescan per category would multiply the reads.
    expect(categoryReads(40)).toBeLessThan(categoryReads(4) * 2)
  })

  it('shows each skill on its own category shelf, most-starred shelf first', () => {
    const starred = (name: string, category: string, stars: number) => ({ ...entry(name, category), stars })

    // Scrambled on purpose, so the star ranking differs from feed order and from A to Z.
    const container = renderDiscovery(
      [
        starred('w1', 'writing', 5),
        starred('d1', 'data', 2),
        entry('m1', 'media'),
        entry('w2', 'writing'),
        starred('m2', 'media', 9),
        entry('d2', 'data')
      ],
      true,
      'skills'
    )

    const shelf = (category: string) =>
      [...container.querySelectorAll(`[data-catalog-section="${category}"] article`)]
        .map(node => node.textContent)
        .sort()

    expect(sections(container)).toEqual(['media', 'writing', 'data'])
    expect(shelf('media')).toEqual(['m1', 'm2'])
    expect(shelf('writing')).toEqual(['w1', 'w2'])
    expect(shelf('data')).toEqual(['d1', 'd2'])
  })
})
