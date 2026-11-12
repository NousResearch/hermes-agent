import { describe, expect, it } from 'vitest'

import { agentCapabilityCopy } from '../ui/agent-capability-copy'

import { agentCatalogSections, agentCatalogTags } from './agent-catalog-sections'
import type { CatalogEntry } from './catalog-data'

const copy = agentCapabilityCopy('en')

const entry = (id: string, patch: Partial<CatalogEntry> = {}) =>
  ({ id, name: id, source: 'community', category: 'general', categoryLabel: 'General', ...patch }) as CatalogEntry

describe('agentCatalogSections', () => {
  it('splits an agent’s skills into built in, added from the library and not added, dropping empty ones', () => {
    const installed = new Set(['notion', 'honcho'])

    const sections = agentCatalogSections({
      kind: 'skills',
      copy,
      entries: [entry('notion', { source: 'built-in' }), entry('honcho'), entry('dynamic-workflow')],
      isInstalled: item => installed.has(item.id)
    })

    expect(sections.map(section => [section.title, section.entries.map(item => item.id)])).toEqual([
      ['Built in', ['notion']],
      ['Added from the library', ['honcho']],
      ['Not added to this agent', ['dynamic-workflow']]
    ])
    expect(sections[0]!.describe(7, 58)).toBe('Ship with every agent. Showing 7 of 58.')
  })

  it('keeps app-wide desktop plugins apart from the agent’s own copies', () => {
    const sections = agentCatalogSections({
      kind: 'plugins',
      copy,
      entries: [entry('langfuse'), entry('accent-picker', { category: 'desktop' })],
      isInstalled: () => true
    })

    expect(sections.map(section => [section.title, section.entries.map(item => item.id)])).toEqual([
      ['On this agent', ['langfuse']],
      ['Runs in the app', ['accent-picker']]
    ])
  })

  it('tags skills with their category and plugins with their provenance', () => {
    expect(agentCatalogTags('skills', entry('pdf', { categoryLabel: 'Productivity' }))).toEqual(['Productivity'])
    expect(agentCatalogTags('plugins', entry('kanban', { source: 'official' }))).toHaveLength(1)
  })
})
