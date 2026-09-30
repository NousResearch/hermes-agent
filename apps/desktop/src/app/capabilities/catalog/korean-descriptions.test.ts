import { expect, it } from 'vitest'

import { ko } from '@/i18n/ko'

import { parseCatalog } from './catalog-data'
import { localizeSkillEntry } from './korean-descriptions'

it('restores known descriptions and category labels in display/search without changing install identity', () => {
  const entry = parseCatalog('skills', [
    { name: 'codebase-inspection', source: 'built-in', category: 'github', description: 'Inspect code' }
  ])[0]

  const result = localizeSkillEntry(entry, ko.skills)
  expect(result.description).toBe(ko.skills.skillDescriptions?.['codebase-inspection'])
  expect(result.categoryLabel).toBe(ko.skills.skillCategoryNames?.github)
  expect(result.search).toContain(result.description.toLowerCase())
  expect(result.id).toBe(entry.id)
  expect(result.installIdentifier).toBe(entry.installIdentifier)
  expect(result.category).toBe('github')
  expect(entry.description).toBe('Inspect code')
})

it('preserves same-named community/local descriptions and unknown first-party copy', () => {
  for (const source of ['hub', 'local']) {
    const entry = parseCatalog('skills', [
      { name: 'codebase-inspection', source, description: 'User-specific instructions' }
    ])[0]

    expect(localizeSkillEntry(entry, ko.skills).description).toBe(entry.description)
  }

  const unknown = parseCatalog('skills', [
    { name: 'future-skill', source: 'built-in', description: 'Future description' }
  ])[0]

  expect(localizeSkillEntry(unknown, ko.skills).description).toBe('Future description')
})
