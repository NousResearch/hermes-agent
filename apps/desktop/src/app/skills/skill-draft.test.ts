import { describe, expect, it } from 'vitest'

import { buildSkillDraft } from './skill-draft'

const draft = {
  name: 'raport',
  category: 'biuro',
  description: 'Gdy proszę o raport.',
  instructions: 'Zbierz wyniki. Podaj źródła.'
}

describe('pasted skill draft', () => {
  it('wraps instructions but preserves full documents, including their metadata', () => {
    const wrapped = buildSkillDraft(draft)
    expect(wrapped.content).toContain(draft.instructions)
    expect(wrapped.content).toContain(JSON.stringify(draft.description))
    const document = '---\nname: raport\ndescription: Raport.\nlicense: MIT\n---\n\nInstrukcja.'
    expect(buildSkillDraft({ ...draft, instructions: document }).content).toBe(document)
  })
  it('rejects path traversal, empty instructions and incomplete documents before saving', () => {
    expect(() => buildSkillDraft({ ...draft, name: '../escape' })).toThrow()
    expect(() => buildSkillDraft({ ...draft, instructions: '' })).toThrow()
    expect(() => buildSkillDraft({ ...draft, instructions: '---\nname: raport' })).toThrow()
    expect(() => buildSkillDraft({ ...draft, description: 'x'.repeat(61) })).toThrow()
  })
})
