import { describe, expect, it } from 'vitest'

import { PROJECT_IDEA_TEMPLATES, randomIdeaTemplates } from './project-idea-templates'

describe('randomIdeaTemplates', () => {
  it('returns Turkish labels and ideas for the Turkish locale', () => {
    const templates = randomIdeaTemplates('tr', PROJECT_IDEA_TEMPLATES.length)

    expect(templates).toHaveLength(PROJECT_IDEA_TEMPLATES.length)
    expect(templates).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: 'Oyun geliştirme maratonu' }),
        expect.objectContaining({ idea: expect.stringContaining('Tek bir temel mekanik') }),
        expect.objectContaining({ label: 'Bilgi kartları' })
      ])
    )
  })

  it('translates every template, not just the ones spot-checked above', () => {
    const englishLabels = new Set(PROJECT_IDEA_TEMPLATES.map(template => template.label))
    const englishIdeas = new Set(PROJECT_IDEA_TEMPLATES.map(template => template.idea))

    for (const template of randomIdeaTemplates('tr', PROJECT_IDEA_TEMPLATES.length)) {
      expect(englishLabels.has(template.label)).toBe(false)
      expect(englishIdeas.has(template.idea)).toBe(false)
    }
  })

  it('keeps English template copy for the default locale', () => {
    expect(randomIdeaTemplates('en', PROJECT_IDEA_TEMPLATES.length)).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: 'Game jam' })])
    )
  })
})
