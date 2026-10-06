import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { MemorySetupConfirmation, runMemoryProviderAction } from './system'

beforeEach(() => {
  vi.useFakeTimers()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
  setApiRequestConnection('local-test')
  setApiRequestProfile('profile-a')
})
afterEach(() => vi.useRealTimers())

it('keeps all progress polls on the owner that started setup after a connection switch', async () => {
  const api = vi.mocked(window.hermesDesktop.api)
  api
    .mockResolvedValueOnce({ id: 'one', status: 'running', progress: { message: 'Installing' } })
    .mockResolvedValueOnce({ id: 'one', status: 'completed', result: { ok: true } })
  const progress = vi.fn()
  const result = runMemoryProviderAction('example', 'save', { values: {} }, undefined, { onProgress: progress })
  setApiRequestConnection('remote-test')
  setApiRequestProfile('profile-b')
  await vi.runAllTimersAsync()
  expect(await result).toEqual({ ok: true })
  expect(progress).toHaveBeenCalledWith(expect.objectContaining({ progress: { message: 'Installing' } }))
  expect(api).toHaveBeenCalledTimes(2)

  for (const [request] of api.mock.calls) {
    expect(request).toMatchObject({ connectionId: 'local-test', profile: 'profile-a' })
  }
})

it('stops polling when its form unmounts without cancelling the server operation', async () => {
  const api = vi.mocked(window.hermesDesktop.api)
  api.mockResolvedValue({ id: 'one', status: 'running' })
  const controller = new AbortController()
  const result = runMemoryProviderAction('example', 'save', {}, undefined, { signal: controller.signal })
  const rejected = expect(result).rejects.toThrow()
  await vi.advanceTimersByTimeAsync(0)
  controller.abort()
  await vi.runAllTimersAsync()
  await rejected
  expect(api).toHaveBeenCalledTimes(1)
})

it('returns a typed provider confirmation without claiming success', async () => {
  vi.mocked(window.hermesDesktop.api).mockResolvedValue({
    status: 'confirmation_required',
    confirmation: 'source_build',
    message: 'Build?'
  })
  await expect(runMemoryProviderAction('example', 'save', {})).rejects.toEqual(
    new MemorySetupConfirmation('Build?', 'source_build')
  )
})
