// A profile rename keeps its sessions: every localStorage family keyed by the old profile name must move
// to the new one, or the restored tabs dial a backend that no longer exists and 404 forever (#111868).

// Loaded once at collection, outside every test's budget: vi.resetModules() keeps each module's
// transformed code, so the fresh copy a test imports only re-evaluates it. Without this the first
// test pays the cold transform of the whole graph and times out on a loaded runner.
import '@/lib/storage'
import '@/store/session'
import '@/store/session-states'
import '@/store/transcript-tail-cache'

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'

describe('migrateTilesForProfile', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('moves tabs, bot tiles, cached tails, remembered ids and owner hints to the new profile name', async () => {
    const storage = await import('@/lib/storage')
    storage.writeJson('hermes.desktop.sessionTiles.v2', {
      webdesign_bhp: [
        { storedSessionId: 's-1', dir: 'right', ownerRoute: { connectionId: 'local', profile: 'webdesign_bhp' } }
      ],
      __bots_workspace__: [
        {
          storedSessionId: 'bot-1',
          dir: 'right',
          workspaceMode: 'bots',
          ownerRoute: { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }
        },
        {
          storedSessionId: 'bot-2',
          dir: 'right',
          workspaceMode: 'bots',
          ownerRoute: { connectionId: 'local', profile: 'other' }
        }
      ]
    })
    storage.writeJson('hermes.desktop.sessionOwnerHints.v1', [
      ['s-1', { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }]
    ])

    const tails = await import('@/store/transcript-tail-cache')
    tails.saveTranscriptTail('s-1', [{ id: 'm1', parts: [{ text: 'hello', type: 'text' }], role: 'user' } as never], {
      connectionId: 'local',
      profile: 'webdesign_bhp'
    })
    // Same-named profile on a remote connection: not renamed, its tail must stay put.
    tails.saveTranscriptTail('s-9', [{ id: 'm9', parts: [{ text: 'remote', type: 'text' }], role: 'user' } as never], {
      connectionId: 'remote-1',
      profile: 'webdesign_bhp'
    })

    const sessionStore = await import('@/store/session')
    sessionStore.setRememberedSessionId('s-1', 'webdesign_bhp')

    const { migrateTilesForProfile } = await import('@/store/session-states')
    const released = Promise.withResolvers<void>()

    vi.stubGlobal('navigator', { locks: {
      request: (_key: string, run: () => unknown) => released.promise.then(run)
    } })

    try {
      const migrating = migrateTilesForProfile('webdesign_bhp', 'hutnik-projectmanager')

      expect(sessionStore.getRememberedSessionId('webdesign_bhp')).toBe('s-1')
      expect(sessionStore.getRememberedSessionId('hutnik-projectmanager')).toBeNull()
      expect(sessionStore.getSessionOwnerHints('s-1')).toEqual([
        { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }
      ])
      released.resolve()
      await migrating
    } finally {
      released.resolve()
      vi.unstubAllGlobals()
    }

    type Tile = { ownerRoute?: { profile: string; targetProfile?: string }; storedSessionId: string }
    const tiles = storage.readJson<Record<string, Tile[]>>('hermes.desktop.sessionTiles.v2')
    expect(tiles?.webdesign_bhp).toBeUndefined()
    expect(tiles?.['hutnik-projectmanager']?.map(t => [t.storedSessionId, t.ownerRoute?.profile])).toEqual([
      ['s-1', 'hutnik-projectmanager']
    ])
    expect(tiles?.__bots_workspace__?.map(t => t.ownerRoute)).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' },
      { connectionId: 'local', profile: 'other' }
    ])

    expect(tails.loadTranscriptTail('s-1', { connectionId: 'local', profile: 'hutnik-projectmanager' })).toHaveLength(1)
    expect(tails.loadTranscriptTail('s-1', { connectionId: 'local', profile: 'webdesign_bhp' })).toBeNull()
    expect(tails.loadTranscriptTail('s-9', { connectionId: 'remote-1', profile: 'webdesign_bhp' })).toHaveLength(1)
    expect(tails.loadTranscriptTail('s-9', { connectionId: 'remote-1', profile: 'hutnik-projectmanager' })).toBeNull()

    expect(sessionStore.getRememberedSessionId('hutnik-projectmanager')).toBe('s-1')
    expect(sessionStore.getRememberedSessionId('webdesign_bhp')).toBeNull()
    expect(sessionStore.getSessionOwnerHints('s-1')).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' }
    ])
  })

  it('leaves owner hints of a same-named profile on another connection alone', async () => {
    const storage = await import('@/lib/storage')
    const remote = { connectionId: 'remote-1', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }
    storage.writeJson('hermes.desktop.sessionOwnerHints.v1', [
      ['s-1', { connectionId: 'local', profile: 'webdesign_bhp', targetProfile: 'webdesign_bhp' }],
      ['s-9', remote]
    ])

    const sessionStore = await import('@/store/session')
    const { migrateTilesForProfile } = await import('@/store/session-states')
    await migrateTilesForProfile('webdesign_bhp', 'hutnik-projectmanager')

    expect(sessionStore.getSessionOwnerHints('s-1')).toEqual([
      { connectionId: 'local', profile: 'hutnik-projectmanager', targetProfile: 'hutnik-projectmanager' }
    ])
    // The rename ran on the local backend; the remote profile still has its old name.
    expect(sessionStore.getSessionOwnerHints('s-9')).toEqual([remote])
    expect(storage.readJson<unknown[]>('hermes.desktop.sessionOwnerHints.v1')).toContainEqual(['s-9', remote])
  })

  it.each(['rename', 'delete'] as const)('%s keeps frame journals scoped and revokes the old profile', async operation => {
    // Fresh module instances are intentional: these tests exercise persisted profile ownership.
    const metrics = await import('@/store/desktop-metrics')
    const profiles = await import('@/store/session-states')
    const minute = 60_000
    const noon = Date.UTC(2026, 8, 27, 12)

    vi.useFakeTimers({ toFake: ['Date'] })
    immediateMetricsLocks()
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    vi.setSystemTime(noon)

    try {
      metrics.bindDesktopMetrics(null, 'local|alpha')
      await setMetricsGate(metrics.captureDesktopMetricsGate, 'on')
      metrics.trackDesktopVisibleTime(11 * minute)
      metrics.recordLongFrames([120])
      vi.setSystemTime(noon + minute)
      metrics.tickDesktopMetrics()
      metrics.bindDesktopMetrics(null, 'remote-1|alpha')
      await setMetricsGate(metrics.captureDesktopMetricsGate, 'on')
      metrics.recordLongFrames([70, 70])
      vi.setSystemTime(noon + 2 * minute)
      metrics.tickDesktopMetrics()

      if (operation === 'rename') {
        await profiles.migrateTilesForProfile('alpha', 'renamed')
      } else {
        profiles.dropTilesForProfile('alpha', { connectionId: 'local', profile: 'alpha' })
      }

      const reports: [string, Record<string, unknown>][] = []

      const requester = (owner: string) => async (method: string, params: Record<string, unknown> = {}) => {
        if (method === 'shared_metrics.desktop_long_frames') {
          reports.push([owner, params])
        }

        return { recorded: true } as never
      }

      vi.setSystemTime(noon + 24 * 60 * minute)
      metrics.bindDesktopMetrics(requester('local'), `local|${operation === 'rename' ? 'renamed' : 'alpha'}`)
      await setMetricsGate(metrics.captureDesktopMetricsGate, 'on')

      if (operation === 'rename') {
        await vi.waitFor(() => expect(reports).toHaveLength(1))
        expect(reports[0]).toEqual(['local', {
          day: '2026-09-27', visible_ms: minute, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
        }])
        metrics.bindDesktopMetrics(requester('old-name'), 'local|alpha')
        await setMetricsGate(metrics.captureDesktopMetricsGate, 'on')
      }

      metrics.bindDesktopMetrics(requester('remote'), 'remote-1|alpha')
      await setMetricsGate(metrics.captureDesktopMetricsGate, 'on')
      await vi.waitFor(() => expect(reports.some(([owner]) => owner === 'remote')).toBe(true))
      expect(reports.filter(([owner]) => owner !== 'local')).toEqual([['remote', {
        day: '2026-09-27', visible_ms: minute, frames: [{ bucket: '50ms_to_100ms', count: 2 }]
      }]])
      expect(reports).toHaveLength(operation === 'rename' ? 2 : 1)
    } finally {
      metrics.resetDesktopMetricsForTests()
      vi.restoreAllMocks()
      vi.unstubAllGlobals()
      vi.useRealTimers()
    }
  })

  it('does not turn a successful profile rename into a failure when telemetry storage is unavailable', async () => {
    // Keep the actual rename seam, after module loading has initialized its stores.
    const profiles = await import('@/store/session-states')
    const descriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')!

    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() { throw new DOMException('storage disabled', 'SecurityError') }
    })

    try {
      await expect(profiles.migrateTilesForProfile('alpha', 'renamed')).resolves.toBeUndefined()
    } finally {
      Object.defineProperty(window, 'localStorage', descriptor)
    }
  })
})
