import { expect, it } from 'vitest'

import { ko } from '@/i18n/ko'
import type { OfficialSkillInfo, SkillInfo } from '@/types/hermes'

import { skillDescription } from './korean-descriptions'
import { filteredOfficial, filteredSkills } from './skills-data'

it('searches translated first-party descriptions and categories without changing backend identity or text', () => {
  const skill: SkillInfo = {
    name: 'codebase-inspection',
    category: 'github',
    description: 'Inspect code',
    enabled: true,
    provenance: 'bundled'
  }

  const text = ko.skills.skillDescriptions!['codebase-inspection']
  expect(skillDescription(skill, ko.skills)).toBe(text)
  expect(filteredSkills([skill], text.normalize('NFD'), true, ko.skills)).toEqual([skill])
  expect(filteredSkills([skill], ko.skills.skillCategoryNames!.github, true, ko.skills)[0]).toBe(skill)
  expect(filteredSkills([skill], 'Inspect code', true, ko.skills)[0]).toBe(skill)

  const official: OfficialSkillInfo = {
    ...skill,
    identifier: 'official/codebase-inspection',
    installed: false,
    tags: []
  }

  expect(filteredOfficial([official], text, ko.skills)[0]).toBe(official)
  expect(skill.description).toBe('Inspect code')
})

it('preserves same-named user/hub/external descriptions and unknown bundled text', () => {
  for (const provenance of ['agent', 'hub', 'external', undefined] as const) {
    const skill: SkillInfo = {
      name: 'codebase-inspection',
      category: 'custom',
      description: 'User instructions',
      enabled: true,
      provenance
    }

    expect(skillDescription(skill, ko.skills)).toBe('User instructions')
    expect(filteredSkills([skill], ko.skills.skillDescriptions!['codebase-inspection'], true, ko.skills)).toEqual([])
  }

  expect(
    skillDescription(
      { name: 'future', category: '', description: 'Future text', enabled: true, provenance: 'bundled' },
      ko.skills
    )
  ).toBe('Future text')
})
