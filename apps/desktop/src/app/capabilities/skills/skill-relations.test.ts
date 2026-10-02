import { expect, it } from 'vitest'
import { skillRelations } from './skill-relations'

it.each([
  ['related_skills: [one, two, one]', ['one', 'two']],
  ['related_skills:\n  - one\n  - two', ['one', 'two']],
  ['related_skills: one, two', ['one', 'two']],
  ['related_skills: [fallback]\nmetadata:\n  hermes:\n    related_skills: [chosen]', ['chosen']],
  ['metadata:\n  hermes:\n    related_skills:\n      - one', ['one']],
  ['related_skills: [one, 7, null, {bad: value}]', ['one']],
  ['related_skills: [', []],
  ['related_skills: []', []],
  ['metadata: null', []]
])('reads soft links without changing the content: %s', (frontmatter, expected) => {
  expect(skillRelations(`---\n${frontmatter}\n---\nBody`)).toEqual(expected)
})
it('does not interpret body text as frontmatter', () => {
  expect(skillRelations('related_skills: [one]')).toEqual([])
  expect(skillRelations('\uFEFF---\r\nrelated_skills: [one]\r\n---')).toEqual(['one'])
})
