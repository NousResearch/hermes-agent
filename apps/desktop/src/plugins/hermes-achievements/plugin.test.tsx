import type { PluginContext } from '@hermes/plugin-sdk'
import { describe, expect, it, vi } from 'vitest'

import plugin, { ACHIEVEMENTS_ROUTE } from './plugin'

interface Registration {
  area: string
  data?: Record<string, unknown>
  id: string
  render?: () => unknown
}

function recordingContext() {
  const registrations: Registration[] = []
  const disposers: Array<() => void> = []
  const rest = vi.fn()
  const registerLocales = vi.fn()

  const ctx = {
    i18n: {
      register: registerLocales,
      t: (key: string) => ({ nav: 'Achievements', open: 'Achievements: Open', title: 'Achievements' })[key] ?? key
    },
    onDispose: (dispose: () => void) => disposers.push(dispose),
    registerMany: (items: Registration[]) => {
      registrations.push(...items)

      return vi.fn()
    },
    rest
  } as unknown as PluginContext

  return { ctx, disposers, registerLocales, registrations, rest }
}

describe('achievements plugin registration', () => {
  it('is enabled by default and contributes a reachable page, nav row, and palette action', () => {
    const harness = recordingContext()

    plugin.register(harness.ctx)

    expect(plugin.defaultEnabled).toBe(true)
    expect(harness.registerLocales).toHaveBeenCalledTimes(1)
    expect(harness.registrations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ area: 'routes', data: { path: ACHIEVEMENTS_ROUTE }, id: 'page' }),
        expect.objectContaining({
          area: 'sidebar.nav',
          data: expect.objectContaining({ label: 'Achievements', path: ACHIEVEMENTS_ROUTE }),
          id: 'nav'
        }),
        expect.objectContaining({
          area: 'palette',
          data: expect.objectContaining({ id: 'hermes-achievements.open', label: 'Achievements: Open' }),
          id: 'open'
        })
      ])
    )
    expect(harness.registrations.find(item => item.id === 'page')?.render).toEqual(expect.any(Function))
    expect(harness.disposers).toHaveLength(1)
  })
})
