import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import {
  FACET_ORDER,
  facetTag,
  HINT_ORDER,
  hintTag,
  hintTags,
  shortenUnknown,
  tagCopy,
  type VocabularyStrings
} from './hint-vocabulary'

const strings: VocabularyStrings = en.connectorsPage.vocabulary

describe('hint vocabulary', () => {
  // The lanes these labels ride in are fixed widths (70px for a facet, 132px for
  // the hints), so a label that grows past its budget silently truncates a row.
  it('ships a label and a sentence inside the lane budget for every facet', () => {
    for (const facet of FACET_ORDER) {
      const copy = tagCopy(facetTag(facet), strings)

      expect(copy.label.length).toBeGreaterThan(0)
      expect(copy.label.length).toBeLessThanOrEqual(14)
      expect(copy.long.length).toBeGreaterThan(0)
    }
  })

  it('ships a label and a sentence inside the lane budget for every hint', () => {
    for (const hint of HINT_ORDER) {
      const copy = tagCopy(hintTag(hint), strings)

      expect(copy.label.length).toBeGreaterThan(0)
      expect(copy.label.length).toBeLessThanOrEqual(11)
      expect(copy.long.length).toBeGreaterThan(0)
    }
  })

  it('covers all four facets and all seven hints', () => {
    expect(FACET_ORDER).toHaveLength(4)
    expect(HINT_ORDER).toHaveLength(7)
  })

  it('falls back to a readable tag for a value it has never seen', () => {
    const tag = hintTag('mutateEverythingHint')

    expect(tag.key).toBeNull()
    expect(tag.tone).toBe('unknown')
    expect(tagCopy(tag, strings).label).toBe('Mutate')
  })

  it('never renders a blank tag, whatever the wire sends', () => {
    for (const raw of ['', '   ', '___', 'Hint', '🙂']) {
      expect(tagCopy(facetTag(raw), strings).label.length).toBeGreaterThan(0)
      expect(tagCopy(hintTag(raw), strings).label.length).toBeGreaterThan(0)
    }
  })

  it('shortens an unknown value to at most eight characters', () => {
    expect(shortenUnknown('search_repositories_v2')).toBe('Search')
    expect(shortenUnknown('extraordinarilyLongHint')).toBe('Extraord')
    expect(shortenUnknown('readOnlyHint')).toBe('Read')
  })

  it('orders known hints first and keeps unknown ones at the end', () => {
    const tags = hintTags(['telepathyHint', 'deleteHint', 'readOnlyHint'])

    expect(tags.map(tag => tag.key)).toEqual(['hintReadOnly', 'hintDelete', null])
    expect(tags[2].shortLabel).toBe('Telepath')
  })

  it('keeps destructive and delete distinguishable', () => {
    const destructive = tagCopy(hintTag('destructiveHint'), strings).label
    const remove = tagCopy(hintTag('deleteHint'), strings).label

    expect(destructive).not.toBe(remove)
  })
})
