import type * as Shared from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import { backupPaletteEntries, backupSettingsHrefForLegacy } from './backup-palette'

const host = vi.hoisted(() => ({ cloud: false }))

vi.mock('@hermes/shared', async importOriginal => ({
  ...(await importOriginal<typeof Shared>()),
  isHermesCloudUrl: () => host.cloud
}))

afterEach(() => {
  host.cloud = false
  window.document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete (window as Window & { __HERMES_HOST_AGENT__?: boolean }).__HERMES_HOST_AGENT__
})

describe('Backup & Transfer entry points', () => {
  it('lists exactly one ⌘K row per task', () => {
    expect(backupPaletteEntries(en).map(entry => [entry.id, entry.label])).toEqual([
      ['backup-agent', 'Back up this agent'],
      ['transfer-to-cloud', 'Transfer to Hermes Cloud']
    ])
  })

  it('offers only the backup row on a Webapp Hermes Cloud serves', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    host.cloud = true

    expect(backupPaletteEntries(en).map(entry => entry.id)).toEqual(['backup-agent'])
  })

  it('offers only the backup row on a host that runs no agent', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    ;(window as Window & { __HERMES_HOST_AGENT__?: boolean }).__HERMES_HOST_AGENT__ = false

    expect(backupPaletteEntries(en).map(entry => entry.id)).toEqual(['backup-agent'])
  })

  it('forwards the portal link to Settings, keeping the agent it names', () => {
    expect(backupSettingsHrefForLegacy("?agent=Sam's%20agent")).toBe('/settings?tab=backup&cloudAgent=Sam%27s+agent')
    expect(backupSettingsHrefForLegacy('')).toBe('/settings?tab=backup')
  })
})
