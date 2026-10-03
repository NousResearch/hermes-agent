import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest } from '@/global'

import { deferred } from '../test/deferred'

import { saveProfileAppearance } from './profile-appearance'

// A profile name belongs to one gateway: a pick queued behind a slow write must
// still land on the gateway it was picked on after the window switches to
// another gateway that has the same profile name. `null` = the untagged
// primary, 'local' = an explicit "This device" tag.

interface Held {
  request: HermesApiRequest
  settle: ReturnType<typeof deferred<{ ok: boolean }>>
}

const route = ({ request }: Held) => [
  (request.body as { config: { desktop: { theme: string } } }).config.desktop.theme,
  request.connectionId ?? null,
  request.profile
]

describe('profile appearance write owner', () => {
  let held: Held[] = []

  const api = vi.fn((request: HermesApiRequest) => {
    const settle = deferred<{ ok: boolean }>()
    held.push({ request, settle })

    return settle.promise
  })

  const flush = () => new Promise(resolve => setTimeout(resolve, 0))

  function on(connectionId: null | string) {
    setApiRequestLocalMode(connectionId === null)
    setApiRequestConnection(connectionId)
  }

  beforeEach(() => {
    held = []
    api.mockClear()
    setApiRequestProfile('alpha')
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
    on(null)
    setApiRequestLocalMode(false)
    setApiRequestProfile(null)
  })

  it.each([
    ['A', 'B', 'ok'],
    ['B', 'A', 'ok'],
    ['local', 'B', 'ok'],
    ['A', 'local', 'ok'],
    [null, 'B', 'ok'],
    ['A', 'B', 'failed'],
    ['local', 'B', 'failed']
  ] as const)('a pick queued on %s keeps that owner after a switch to %s (first write %s)', async (from, to, first) => {
    on(from)
    const firstPick = saveProfileAppearance('alpha', { theme: 'mono' }).catch(() => 'rejected')
    const secondPick = saveProfileAppearance('alpha', { theme: 'everforest' })
    await flush()
    expect(held.map(route)).toEqual([['mono', from, 'alpha']])

    on(to)
    held.shift()!.settle.resolve({ ok: first === 'ok' })
    await flush()

    // FIFO within the original owner, still addressed to it after the switch,
    // whether the first write landed or failed.
    expect(held.map(route)).toEqual([['everforest', from, 'alpha']])

    // A pick on the new owner does not queue behind the old owner's write.
    const otherOwnerPick = saveProfileAppearance('alpha', { theme: 'ember' })
    await flush()
    expect(held.map(route)).toEqual([
      ['everforest', from, 'alpha'],
      ['ember', to, 'alpha']
    ])

    for (const { settle } of held.splice(0)) {
      settle.resolve({ ok: true })
    }

    await expect(firstPick).resolves.toBe(first === 'ok' ? undefined : 'rejected')
    await expect(secondPick).resolves.toBeUndefined()
    await expect(otherOwnerPick).resolves.toBeUndefined()
  })
})
