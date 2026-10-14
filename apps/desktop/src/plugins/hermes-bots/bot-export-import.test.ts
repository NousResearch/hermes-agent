/**
 * Bot Mode's door to profile export/import (#121619): a bot's whole identity
 * lives in its profile dir, so saving that profile as an archive is how a bot
 * survives an uninstall or moves to another machine.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileRoute, RosterRow } from './types'

const { exportProfile, importProfile, invalidateQueries } = vi.hoisted(() => ({
  exportProfile: vi.fn(),
  importProfile: vi.fn(),
  invalidateQueries: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, exportProfile, importProfile },
    queryClient: { invalidateQueries }
  }
})

const { canExportBot, exportBot, importBot } = await import('./profile-ops')
const { ROSTER_KEY } = await import('./data')

const localRoute: ProfileRoute = {
  connectionId: 'local',
  mode: 'local',
  profile: 'researcher',
  targetProfile: 'backend-researcher'
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('exporting a bot', () => {
  it('exports the backend profile a source-scoped row is served by', async () => {
    exportProfile.mockResolvedValue('/tmp/researcher.tar.gz')

    await exportBot({ name: 'researcher', route: localRoute, sourceScoped: true } as RosterRow)

    expect(exportProfile.mock.calls).toEqual([['backend-researcher']])
  })

  it('is not offered for a row the active backend does not serve', () => {
    expect(canExportBot({ name: 'researcher' } as RosterRow)).toBe(true)
    expect(canExportBot({ name: 'worker', remoteSource: true } as RosterRow)).toBe(false)
  })
})

describe('importing a bot', () => {
  it('repaints the roster so the restored bot appears', async () => {
    importProfile.mockResolvedValue('researcher')

    expect(await importBot()).toBe('researcher')
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ROSTER_KEY })
  })

  it('leaves the roster alone when the user cancels', async () => {
    importProfile.mockResolvedValue(null)

    expect(await importBot()).toBeNull()
    expect(invalidateQueries).not.toHaveBeenCalled()
  })
})
