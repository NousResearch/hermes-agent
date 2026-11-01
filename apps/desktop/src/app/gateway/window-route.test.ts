import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'

import { followWindowRoute, sendWindowRoute } from './window-route'

const conn = (connectionId: string, profile: string, registryScoped = true) =>
  ({ baseUrl: `https://${connectionId}`, connectionId, profile, registryScoped }) as unknown as HermesConnection

const tower = conn('tower', 'default')
const laptop = conn('laptop', 'default')

let setActiveConnectionRoute: ReturnType<typeof vi.fn>
let off: () => void

beforeEach(() => {
  $connection.set(null)
  setActiveConnectionRoute = vi.fn()
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { setActiveConnectionRoute }
  off = followWindowRoute()
})

afterEach(() => {
  off()
  $connection.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('followWindowRoute', () => {
  it('sends nothing at subscribe time, so the route main seeded survives boot', () => {
    off()
    $connection.set(tower)
    setActiveConnectionRoute.mockClear()
    off = followWindowRoute()

    expect(setActiveConnectionRoute).not.toHaveBeenCalled()
  })

  it('mirrors every foreground change, including a return to an earlier route', () => {
    $connection.set(tower)
    $connection.set(conn('laptop', 'work'))
    $connection.set(tower)

    expect(setActiveConnectionRoute.mock.calls.map(([route]) => route)).toEqual([
      { connectionId: 'tower', profile: 'default', registryScoped: true },
      { connectionId: 'laptop', profile: 'work', registryScoped: true },
      { connectionId: 'tower', profile: 'default', registryScoped: true }
    ])
  })

  it('does not re-send an identical route carried by a new descriptor object', () => {
    $connection.set(tower)
    $connection.set({ ...tower, isFullScreen: true } as HermesConnection)

    expect(setActiveConnectionRoute).toHaveBeenCalledTimes(1)
  })

  it('keeps the route through a reconnect blip instead of clearing it', () => {
    $connection.set(laptop)
    $connection.set(null)
    $connection.set(laptop)

    expect(setActiveConnectionRoute.mock.calls).toEqual([
      [{ connectionId: 'laptop', profile: 'default', registryScoped: true }]
    ])
  })
})

describe('sendWindowRoute', () => {
  it('force re-asserts an identical route', () => {
    sendWindowRoute(tower)
    sendWindowRoute(tower)
    sendWindowRoute(tower, { force: true })

    expect(setActiveConnectionRoute).toHaveBeenCalledTimes(2)
  })

  it('an explicit null clears the route, and the next route is sent again', () => {
    $connection.set(tower)
    sendWindowRoute(null, { force: true })
    $connection.set({ ...tower })

    expect(setActiveConnectionRoute.mock.calls.map(([route]) => route)).toEqual([
      { connectionId: 'tower', profile: 'default', registryScoped: true },
      null,
      { connectionId: 'tower', profile: 'default', registryScoped: true }
    ])
  })
})
