import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import { backupPaletteEntries, backupSettingsHrefForLegacy } from './backup-palette'

const host = vi.hoisted(() => ({ cloud: false }))

vi.mock('@/lib/hermes-cloud-host', () => ({ servedByHermesCloud: () => host.cloud }))

afterEach(() => {
  host.cloud = false
})

describe('Backup & Transfer entry points', () => {
  it('lists exactly one ⌘K row per task', () => {
    expect(backupPaletteEntries(en).map(entry => [entry.id, entry.label])).toEqual([
      ['backup-agent', 'Back up this agent'],
      ['transfer-to-cloud', 'Transfer to Hermes Cloud']
    ])
  })

  it('offers only the backup row on a Webapp Hermes Cloud serves', () => {
    host.cloud = true

    expect(backupPaletteEntries(en).map(entry => entry.id)).toEqual(['backup-agent'])
  })

  it('forwards the portal link to Settings, keeping the agent it names', () => {
    expect(backupSettingsHrefForLegacy("?agent=Sam's%20agent")).toBe('/settings?tab=backup&cloudAgent=Sam%27s+agent')
    expect(backupSettingsHrefForLegacy('')).toBe('/settings?tab=backup')
  })
})
