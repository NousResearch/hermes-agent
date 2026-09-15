// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'

import { COMPACT_SCOPE_CONTENT_CLASS, COMPACT_SCOPE_TRIGGER_CLASS } from './scope-label'

const skillsIndexSource = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'index.tsx'), 'utf8')

function compactTriggerClassFromSource() {
  const match = skillsIndexSource.match(/compactSelector \? '([^']+)' : 'h-7 w-56'/)
  if (match) {
    return match[1]
  }

  return skillsIndexSource.includes('COMPACT_SCOPE_TRIGGER_CLASS') ? COMPACT_SCOPE_TRIGGER_CLASS : ''
}

describe('plugins pageBlurb', () => {
  it('drops the English one-row changelog residue and still explains the switches', () => {
    expect(en.skills.plugins.pageBlurb).not.toMatch(/One row per plugin/i)
    expect(en.skills.plugins.pageBlurb).toMatch(/switch/i)
    expect(en.skills.plugins.pageBlurb.trim().length).toBeGreaterThan(0)
  })

  it('drops the Chinese one-row changelog residue and stays user-facing', () => {
    expect(zh.skills.plugins.pageBlurb).not.toMatch(/每个插件一行/)
    expect(zh.skills.plugins.pageBlurb).toMatch(/开关|agent/)
    expect(zh.skills.plugins.pageBlurb.trim().length).toBeGreaterThan(0)
  })
})

describe('plugins compact Agent-column scope selector', () => {
  it('uses a min-w-0 truncating trigger without max-w-64', () => {
    const triggerClass = compactTriggerClassFromSource()
    expect(triggerClass).toMatch(/min-w-0/)
    expect(triggerClass).toMatch(/truncate|overflow-hidden/)
    expect(triggerClass).not.toMatch(/max-w-64/)
    expect(skillsIndexSource).not.toMatch(/max-w-64/)
    expect(skillsIndexSource).toMatch(/SelectValue className=\{compactSelector \? 'min-w-0 truncate'/)
  })

  it('caps compact SelectContent to the available viewport width', () => {
    expect(skillsIndexSource).toMatch(/COMPACT_SCOPE_CONTENT_CLASS/)
    expect(COMPACT_SCOPE_CONTENT_CLASS).toMatch(
      /max-w-\(--radix-select-available-width\)|max-w-\[min\(var\(--radix-select-available-width\)/
    )
  })

  it('wires compact roster labels through rosterScopeLabel', () => {
    expect(skillsIndexSource).toMatch(/rosterScopeLabel\(agent, activeId, compactSelector\)/)
  })
})
