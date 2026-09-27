import { host } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import plugin from './plugin'
import { $boardRequest } from './ui'

vi.mock('@/hermes', () => ({ getGlobalModelOptions: vi.fn(), setApiRequestProfile: vi.fn() }))

type Ctx = Parameters<typeof plugin.register>[0]
type Registered = { area: string; data?: unknown; id: string }

/** Just enough host to register: the doors bindApi opens, and a recorder for
 *  the contributions. */
function fakeContext() {
  const contributions: Registered[] = []
  const disposers: Array<() => void> = []

  const ctx = {
    i18n: { onLocaleChange: vi.fn(() => vi.fn()), register: vi.fn(), t: (key: string) => key },
    onDispose: (dispose: () => void) => disposers.push(dispose),
    os: undefined,
    registerMany: (items: Registered[]) => contributions.push(...items),
    rest: vi.fn(),
    socket: () => vi.fn(),
    storage: { get: (_key: string, fallback: unknown) => fallback, remove: vi.fn(), set: vi.fn() }
  }

  return { contributions, ctx: ctx as unknown as Ctx, dispose: () => disposers.forEach(fn => fn()) }
}

afterEach(() => {
  vi.restoreAllMocks()
  $boardRequest.set(null)
})

describe('fleet-scoped entry command', () => {
  it('parks a fresh fleet-board request on every run and navigates to the board page', () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => undefined)
    const { contributions, ctx, dispose } = fakeContext()

    plugin.register(ctx)

    const row = contributions.find(c => c.id === 'open-fleet')?.data as { id: string; run: () => void } | undefined

    expect(row?.id).toBe('kanban.openFleet')

    row!.run()
    const first = $boardRequest.get()

    expect(first).toMatchObject({ slug: 'fleet' })
    expect(navigate).toHaveBeenCalledWith('/kanban')

    // Running it again is a NEW request — even with nothing else changed — so
    // the page re-enters after a manual switch instead of ignoring the command.
    row!.run()

    expect($boardRequest.get()).toMatchObject({ slug: 'fleet' })
    expect($boardRequest.get()).not.toBe(first)

    dispose()
  })
})
