import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const $gateway = atom<unknown>(null)
const request = vi.fn(async (_profile: string, _method: string, _params?: Record<string, unknown>) => undefined)

// The viewed profile is not the launch profile: an unscoped write would land in
// the wrong config.yaml (#125969 class), so every push must be routed by profile.
vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGatewayProfileKey: () => 'work',
  requestGatewayForProfile: request
}))

const { mirrorDisplayToggle } = await import('./display-toggles')

const STORAGE_KEY = 'hermes.desktop.test-toggle.v1'

const $enabled = atom(true)

mirrorDisplayToggle('display.test_toggle', STORAGE_KEY, $enabled)

const sets = () =>
  request.mock.calls.filter(([, method]) => method === 'config.set').map(([profile, , params]) => ({ ...params, profile }))

beforeEach(() => {
  localStorage.clear()
  request.mockClear()
  $enabled.set(true)
  request.mockClear()
})

describe('display toggle mirror', () => {
  it('sends the user answer to the gateway when it changes', () => {
    $enabled.set(false)

    expect(sets()).toEqual([{ key: 'display.test_toggle', profile: 'work', value: 'false' }])
  })

  it('re-sends a touched setting to a gateway that has never seen it', () => {
    localStorage.setItem(STORAGE_KEY, 'false')
    $enabled.set(false)
    request.mockClear()

    $gateway.set({})

    expect(sets()).toEqual([{ key: 'display.test_toggle', profile: 'work', value: 'false' }])
  })

  it('leaves an untouched setting alone, so a hand-edited config.yaml wins', () => {
    $gateway.set({})

    expect(sets()).toEqual([])
  })
})
