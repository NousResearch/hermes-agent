import { describe, expect, it } from 'vitest'

import type { SkillInfo } from '@/types/hermes'

import { type CatalogEntry, parseCatalog } from './catalog-data'
import { isSkillEntryInstalled, type SkillCatalogInstallIndex } from './skill-catalog'

const HUB_IDENTIFIER = 'clawhub/hoyaryyj/kepano-defuddle'

function setupIndex() {
  const hubSkill: SkillInfo = {
    category: 'dev',
    description: 'defuddle workflow',
    enabled: true,
    name: 'defuddle',
    provenance: 'hub'
  }

  // Installed row, shaped exactly like the memo's localEntries mapping.
  const installed = {
    ...parseCatalog('skills', [
      { name: 'defuddle', description: 'defuddle workflow', category: 'dev', source: 'hub' }
    ])[0],
    id: 'installed:defuddle',
    installIdentifier: null
  } satisfies CatalogEntry

  const skillsById = new Map([[installed.id, hubSkill]])
  const installedByIdentifier = new Map([[HUB_IDENTIFIER, installed]])

  const index: SkillCatalogInstallIndex = {
    skillsById,
    // Mirrors the memo's matchInstalled: exact identifier hits only.
    matchInstalled: entry => installedByIdentifier.get(entry.installIdentifier ?? entry.identifier),
    officialFor: () => undefined,
    installedIdentifiers: new Set([HUB_IDENTIFIER])
  }

  // Same-name feed rows from another source: the #126991 ghost rows.
  const lookalikes = parseCatalog('skills', [
    {
      name: 'defuddle',
      description: 'lookalike one',
      category: 'dev',
      source: 'skills.sh',
      identifier: 'panniantong/defuddle'
    },
    {
      name: 'defuddle',
      description: 'lookalike two',
      category: 'dev',
      source: 'skills.sh',
      identifier: 'someone-else/defuddle'
    }
  ])

  const unrelated = parseCatalog('skills', [
    {
      name: 'other-skill',
      description: 'unrelated',
      category: 'dev',
      source: 'skills.sh',
      identifier: 'someone-else/other-skill'
    }
  ])[0]

  return { installed, lookalikes, unrelated, index }
}

describe('isSkillEntryInstalled', () => {
  it('keeps the true installed row installed', () => {
    const { installed, index } = setupIndex()

    expect(isSkillEntryInstalled(installed, index)).toBe(true)
  })

  it('leaves same-name feed rows from other sources installable (no ghosts)', () => {
    const { lookalikes, index } = setupIndex()

    expect(lookalikes).toHaveLength(2)

    for (const entry of lookalikes) {
      expect(entry.installIdentifier).not.toBe(HUB_IDENTIFIER)
      expect(isSkillEntryInstalled(entry, index)).toBe(false)
    }
  })

  it('leaves unrelated feed rows uninstalled', () => {
    const { unrelated, index } = setupIndex()

    expect(isSkillEntryInstalled(unrelated, index)).toBe(false)
  })
})
