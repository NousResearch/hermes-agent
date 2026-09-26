import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ registry: vi.fn(), apply: vi.fn(), overlay: vi.fn(), error: vi.fn() }))
vi.mock('@/store/connections', () => ({ refreshConnectionsRegistry: mocks.registry }))
vi.mock('@/store/updates', () => ({ applyEverythingUpdate: mocks.apply, setUpdateOverlayOpen: mocks.overlay }))
vi.mock('@/store/notifications', () => ({ notifyError: mocks.error }))
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))

import { updateAllFromLauncher } from './update-all-launch'

beforeEach(() => vi.resetAllMocks())
afterEach(() => vi.useRealTimers())

it('waits for the local registry then calls the existing all-target flow, not a local-only branch', async () => {
  let ready: (value: unknown) => void = () => undefined
  mocks.registry.mockImplementation(
    () =>
      new Promise(resolve => {
        ready = resolve
      })
  )
  const request = updateAllFromLauncher()
  expect(mocks.apply).not.toHaveBeenCalled()
  ready({ connections: [{ id: 'local' }, { id: 'offline-vps' }] })
  await request
  expect(mocks.overlay).toHaveBeenCalledExactlyOnceWith(true)
  expect(mocks.apply).toHaveBeenCalledTimes(1)
  expect(mocks.error).not.toHaveBeenCalled()
})

it('fails closed on missing/failed registry reads, including a late result after timeout', async () => {
  vi.useFakeTimers()
  mocks.registry.mockResolvedValueOnce(null).mockRejectedValueOnce(new Error('read failed'))
  await updateAllFromLauncher()
  await updateAllFromLauncher()
  let ready: (value: unknown) => void = () => undefined
  mocks.registry.mockImplementation(
    () =>
      new Promise(resolve => {
        ready = resolve
      })
  )
  const request = updateAllFromLauncher()
  await vi.advanceTimersByTimeAsync(10_001)
  await request
  ready({ connections: [{ id: 'local' }] })
  await Promise.resolve()
  expect(mocks.apply).not.toHaveBeenCalled()
  expect(mocks.overlay).not.toHaveBeenCalled()
  expect(mocks.error).toHaveBeenCalledTimes(3)
})
