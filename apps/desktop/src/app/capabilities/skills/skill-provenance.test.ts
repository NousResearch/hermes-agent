import { describe, expect, it } from 'vitest'

import type { SkillInfo } from '@/types/hermes'

import { catalogSourceFor, isEditableProvenance, skillOrigin } from './skill-provenance'

// #108032: externally mounted skills get an 'external' provenance tier — origin
// labeling without losing in-place edit rights (commit 8c8fc6c1ec / PR #17512).
describe('skill provenance tiers', () => {
  describe('catalogSourceFor', () => {
    it('maps every provenance tier to a distinct catalog source', () => {
      expect(catalogSourceFor('bundled')).toBe('built-in')
      expect(catalogSourceFor('hub')).toBe('hub')
      expect(catalogSourceFor('external')).toBe('external')
      expect(catalogSourceFor('agent')).toBe('local')
    })

    it('treats an absent provenance as local (older backends)', () => {
      expect(catalogSourceFor(undefined)).toBe('local')
    })
  })

  describe('isEditableProvenance', () => {
    it('keeps learned/local skills editable', () => {
      expect(isEditableProvenance('agent')).toBe(true)
    })

    it('keeps external mounts editable in place', () => {
      expect(isEditableProvenance('external')).toBe(true)
    })

    it('keeps bundled and hub skills managed by their sources', () => {
      expect(isEditableProvenance('bundled')).toBe(false)
      expect(isEditableProvenance('hub')).toBe(false)
    })

    it('treats an absent provenance as not editable (older backends predate edit rights)', () => {
      const skill: SkillInfo = { category: 'general', description: 'd', enabled: true, name: 'legacy' }
      expect(isEditableProvenance(skill.provenance)).toBe(false)
    })
  })
})

// #70712: the explicit `origin` field drives human-facing labels. 'learned' is
// reserved for background_review (autonomously learned); a local skill — even
// one the legacy `agent` provenance fallback covers — is 'local', never 'learned'.
describe('skillOrigin', () => {
  const skill = (overrides: Partial<SkillInfo>): SkillInfo => ({
    category: 'general',
    description: 'd',
    enabled: true,
    name: 's',
    ...overrides
  })

  it('prefers the explicit origin field', () => {
    expect(skillOrigin(skill({ origin: 'background_review' }))).toBe('background_review')
    expect(skillOrigin(skill({ origin: 'local' }))).toBe('local')
  })

  it('falls back to legacy provenance tiers without inventing learned', () => {
    expect(skillOrigin(skill({ provenance: 'bundled' }))).toBe('bundled')
    expect(skillOrigin(skill({ provenance: 'hub' }))).toBe('hub')
    expect(skillOrigin(skill({ provenance: 'external' }))).toBe('external')
  })

  it('classifies the agent fallback as local, not learned (#70712)', () => {
    // The exact regression: provenance 'agent' previously meant a 'learned' badge.
    expect(skillOrigin(skill({ provenance: 'agent' }))).toBe('local')
    expect(skillOrigin(skill({}))).toBe('local')
  })

  it('keeps an explicit origin over a stale provenance', () => {
    expect(skillOrigin(skill({ provenance: 'agent', origin: 'background_review' }))).toBe('background_review')
  })
})
