import { type PluginStorage, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $alertsMode, ALERTS_MODE_KEY, parseAlertsMode } from './alerts-mode'
import { $boardSlug, bindApi, boardKey } from './api'

afterEach(() => {
  $alertsMode.set('toast')
  $boardSlug.set('')
  queryClient.clear()
})

describe('parseAlertsMode', () => {
  it('keeps the known modes and maps anything else to toast', () => {
    expect(parseAlertsMode('toast')).toBe('toast')
    expect(parseAlertsMode('quiet')).toBe('quiet')
    expect(parseAlertsMode('badge')).toBe('badge')

    for (const garbage of [undefined, null, '', 'loud', 'QUIET', 3, {}, ['badge']]) {
      expect(parseAlertsMode(garbage)).toBe('toast')
    }
  })
})

describe('alerts mode persistence', () => {
  it('hydrates from plugin storage before the events socket dials, and writes changes back', () => {
    queryClient.setQueryData(boardKey('local', 'ops', false), {
      assignees: [],
      columns: [],
      latest_event_id: 7,
      now: 0,
      tenants: []
    })

    const stored: Record<string, unknown> = { [ALERTS_MODE_KEY]: 'badge', boardSlug: 'ops' }

    const storage: PluginStorage = {
      get: <T>(key: string, fallback: T) => (key in stored ? (stored[key] as T) : fallback),
      remove: vi.fn(),
      set: vi.fn((key: string, value: unknown) => {
        stored[key] = value
      })
    }

    const modeAtDial: string[] = []

    const socket = vi.fn(() => {
      modeAtDial.push($alertsMode.get())

      return vi.fn()
    })

    const dispose = bindApi(
      async () => {
        throw new Error('cached snapshot must not refetch')
      },
      storage,
      socket
    )

    expect(modeAtDial).toEqual(['badge'])

    $alertsMode.set('quiet')
    expect(stored[ALERTS_MODE_KEY]).toBe('quiet')

    dispose()
  })

  it('an unknown stored value hydrates as toast', () => {
    const storage: PluginStorage = {
      get: <T>(key: string, fallback: T) => (key === ALERTS_MODE_KEY ? ('future-mode' as T) : fallback),
      remove: vi.fn(),
      set: vi.fn()
    }

    $alertsMode.set('badge')

    const dispose = bindApi(
      async () => ({}) as never,
      storage,
      vi.fn(() => vi.fn())
    )

    expect($alertsMode.get()).toBe('toast')
    dispose()
  })
})
