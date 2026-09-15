import { describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { en } from './en'
import { ko } from './ko'

describe('Korean translation contract', () => {
  it('matches English value kinds throughout the catalog and keeps callable labels usable', () => {
    const compare = (base: unknown, translated: unknown, path: string) => {
      expect(typeof translated, path).toBe(typeof base)
      expect(Array.isArray(translated), path).toBe(Array.isArray(base))

      if (base && typeof base === 'object' && !Array.isArray(base)) {
        for (const [key, value] of Object.entries(base)) {
          compare(value, (translated as Record<string, unknown>)[key], `${path}.${key}`)
        }
      }
    }

    compare(en, ko, 'ko')
    expect(ko.boot.failure.remoteSignInHint('SIGN_IN')).toContain('SIGN_IN')
    expect(ko.desktop.branchTitle(17)).toContain('17')
    expect(ko.ui.sidebar.toggle(true)).not.toBe(ko.ui.sidebar.toggle(false))
    expect(ko.skills.toggleToolset('TOOLS', true)).not.toBe(ko.skills.toggleToolset('TOOLS', false))
  })

  it('keeps original description extensions and schema field copy available', () => {
    expect(ko.settings.envDescriptions?.GITHUB_TOKEN).toBeTruthy()
    expect(ko.skills.skillDescriptions?.['codebase-inspection']).toBeTruthy()
    expect(ko.skills.toolsetDescriptions?.web).toBeTruthy()
    expect(ko.skills.skillCategoryNames?.github).toBeTruthy()
    expect(fieldCopyForSchemaKey(ko.settings.fieldLabels, 'display.show_reasoning')).toBeTruthy()
    expect(fieldCopyForSchemaKey(ko.settings.fieldDescriptions, 'delegation.max_iterations')).toBeTruthy()
    expect(ko.keybinds.title).toBeTruthy()
  })
})
