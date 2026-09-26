import { expect, it } from 'vitest'

import { normalizePersonality, withPersonality } from './personality'

it('replaces only the personality block and preserves other instructions', () => {
  const first = normalizePersonality({ name: 'Ada', purpose: 'Planowanie', tone: 'direct' })
  const next = normalizePersonality({ name: 'Jan', purpose: 'Nauka', detail: 'thorough' })
  const result = withPersonality(withPersonality('Keep my existing rules.', first), next)
  expect(result).toContain('Keep my existing rules.')
  expect(result).toContain('Jan')
  expect(result).not.toContain('Ada')
  expect(withPersonality(result, next)).toBe(result)
})
