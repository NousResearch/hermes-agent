/**
 * Roster section collapse persistence.
 *
 * Folding TEAM or GROUP CHATS is a preference, not view state: it has to
 * survive a restart. These cover the storage contract (parse/serialize) and
 * the independence of the two sections, without needing a plugin context or a
 * rendered pane.
 */

import { describe, expect, it } from 'vitest'

import {
  parseCollapsedSections,
  serializeCollapsedSections,
  toggleCollapsedSection
} from './roster-section-collapse'

describe('parseCollapsedSections', () => {
  it('round-trips a stored array to the same section ids', () => {
    const parsed = parseCollapsedSections(['team', 'group-chats'])

    expect(parsed.size).toBe(2)
    expect(parsed.has('team')).toBe(true)
    expect(parsed.has('group-chats')).toBe(true)
  })

  it('reads absent, corrupt and wrong-shaped values as nothing collapsed', () => {
    // A pane that fails to render is a far worse outcome than a lost
    // preference, so every unusable shape degrades rather than throwing.
    for (const value of [undefined, null, '', 'team', 42, {}, { team: true }]) {
      expect(parseCollapsedSections(value).size, `expected empty for ${JSON.stringify(value)}`).toBe(0)
    }
  })

  it('drops non-string and blank entries and trims the survivors', () => {
    const parsed = parseCollapsedSections(['team', '', '   ', 7, null, ' group-chats '])

    expect([...parsed].sort()).toEqual(['group-chats', 'team'])
  })
})

describe('serializeCollapsedSections', () => {
  it('sorts, so an unchanged selection produces an unchanged value', () => {
    const a = serializeCollapsedSections(new Set(['team', 'group-chats', 'gateway:local']))
    const b = serializeCollapsedSections(new Set(['gateway:local', 'team', 'group-chats']))

    expect(a).toEqual(['gateway:local', 'group-chats', 'team'])
    expect(a).toEqual(b)
  })

  it('survives a serialize/parse round trip', () => {
    const original = new Set(['team', 'gateway:hermy-a600'])
    const restored = parseCollapsedSections(serializeCollapsedSections(original))

    expect([...restored].sort()).toEqual([...original].sort())
  })
})

describe('toggleCollapsedSection', () => {
  it('folds and unfolds one section without touching the others', () => {
    const start = new Set(['team'])

    const bothFolded = toggleCollapsedSection(start, 'group-chats')
    expect([...bothFolded].sort()).toEqual(['group-chats', 'team'])

    const teamOpen = toggleCollapsedSection(bothFolded, 'team')
    expect([...teamOpen]).toEqual(['group-chats'])
  })

  it('returns a new set and never mutates the input', () => {
    const start = new Set(['team'])
    const next = toggleCollapsedSection(start, 'group-chats')

    expect(next).not.toBe(start)
    expect([...start]).toEqual(['team'])
  })

  it('collapses TEAM and GROUP CHATS independently', () => {
    // The two sections the design brief requires to fold on their own.
    // Folding one must never imply the other.
    let sections = new Set<string>()

    sections = toggleCollapsedSection(sections, 'team')
    expect(sections.has('team')).toBe(true)
    expect(sections.has('group-chats'), 'folding TEAM must not fold GROUP CHATS').toBe(false)

    sections = toggleCollapsedSection(sections, 'group-chats')
    expect(sections.has('team')).toBe(true)
    expect(sections.has('group-chats')).toBe(true)

    sections = toggleCollapsedSection(sections, 'team')
    expect(sections.has('team')).toBe(false)
    expect(sections.has('group-chats'), 'unfolding TEAM must not unfold GROUP CHATS').toBe(true)
  })
})
