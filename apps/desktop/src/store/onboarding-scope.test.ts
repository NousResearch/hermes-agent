import { afterEach, expect, it, vi } from 'vitest'

import { isTimeoutError, RECONNECT_ATTEMPT_TIMEOUT_MS } from '@/lib/with-timeout'

import { requestOnboardingGateway } from './onboarding-scope'

const mocks = vi.hoisted(() => ({ request: vi.fn(async () => ({ ok: true })) }))

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: mocks.request }))

const owner = { connectionId: 'athena', profile: 'leverage-ai' }

function installDesktop(bridge: Record<string, unknown>) {
  const getConnection = vi.fn(async () => ({ sharedPrimary: true }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { getConnection, ...bridge } })

  return getConnection
}

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

// Each failure must surface without sending the RPC anywhere, and without
// falling back to the profile-only resolver that would pick another device.
it('rejects a registry owner on a bridge that cannot dial registry connections', async () => {
  const getConnection = installDesktop({})

  await expect(requestOnboardingGateway(owner, 'setup.runtime_check')).rejects.toThrow(/Update Hermes Desktop/)
  expect(getConnection).not.toHaveBeenCalled()
  expect(mocks.request).not.toHaveBeenCalled()
})

it('surfaces a removed registry connection instead of retargeting setup', async () => {
  const getConnectionFor = vi.fn(async () => {
    throw new Error('No connection with id "athena"')
  })

  const getConnection = installDesktop({ getConnectionFor })

  await expect(requestOnboardingGateway(owner, 'setup.runtime_check')).rejects.toThrow('No connection with id "athena"')
  expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'athena', profile: 'leverage-ai' })
  expect(getConnection).not.toHaveBeenCalled()
  expect(mocks.request).not.toHaveBeenCalled()
})

it('times out a wedged connection lookup', async () => {
  vi.useFakeTimers()
  const getConnectionFor = vi.fn(() => new Promise(() => undefined))
  const getConnection = installDesktop({ getConnectionFor })
  const pending = requestOnboardingGateway(owner, 'setup.runtime_check').catch((error: unknown) => error)

  await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)
  const error = await pending

  expect(isTimeoutError(error)).toBe(true)
  expect((error as Error).message).toBe('Timed out resolving provider setup for "leverage-ai"')
  expect(getConnection).not.toHaveBeenCalled()
  expect(mocks.request).not.toHaveBeenCalled()
})

it('surfaces a failed legacy lookup for an untagged owner', async () => {
  const getConnection = vi.fn(async () => {
    throw new Error('Profile "legacy" is unavailable')
  })

  installDesktop({ getConnection, getConnectionFor: vi.fn() })

  await expect(requestOnboardingGateway({ connectionId: null, profile: 'legacy' }, 'setup.status')).rejects.toThrow(
    'Profile "legacy" is unavailable'
  )
  expect(getConnection).toHaveBeenCalledWith('legacy')
  expect(mocks.request).not.toHaveBeenCalled()
})
