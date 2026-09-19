import { host } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import plugin from './plugin'

vi.mock('@/hermes', () => ({ getGlobalModelOptions: vi.fn(), setApiRequestProfile: vi.fn() }))

type Ctx = Parameters<typeof plugin.register>[0]
type Registered = { area: string; data?: unknown; id: string }

/** Just enough host to register: the doors bindApi opens, and a recorder for
 *  the contributions. */
function fakeContext() {
  const contributions: Registered[] = []
  const disposers: Array<() => void> = []

  const ctx = {
    i18n: { register: vi.fn(), t: (key: string) => key },
    onDispose: (dispose: () => void) => disposers.push(dispose),
    os: undefined,
    registerMany: (items: Registered[]) => contributions.push(...items),
    rest: vi.fn(),
    socket: () => vi.fn(),
    storage: { get: (_key: string, fallback: unknown) => fallback, remove: vi.fn(), set: vi.fn() }
  }

  return { contributions, ctx: ctx as unknown as Ctx, dispose: () => disposers.forEach(fn => fn()) }
}

afterEach(() => vi.restoreAllMocks())

describe('fleet-scoped entry command', () => {
  it('offers a palette row that enters the board page scoped to the fleet board', () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => undefined)
    const { contributions, ctx, dispose } = fakeContext()

    plugin.register(ctx)

    const row = contributions.find(c => c.id === 'open-fleet')?.data as { id: string; run: () => void } | undefined

    expect(row?.id).toBe('kanban.openFleet')
    row!.run()
    expect(navigate).toHaveBeenCalledWith('/kanban?board=fleet')

    dispose()
  })
})
