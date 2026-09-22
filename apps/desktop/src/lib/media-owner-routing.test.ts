import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { $connection } from '@/store/session'

import { resolveMediaDisplaySrc } from './media'

const localData = 'data:image/png;base64,bG9jYWw='
const remoteData = 'data:image/png;base64,cmVtb3Rl'
const read = vi.fn(async () => localData)
const api = vi.fn(async () => ({ dataUrl: remoteData }))

beforeEach(() => {
  read.mockClear()
  api.mockReset()
  api.mockResolvedValue({ dataUrl: remoteData })
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: read, api })
  $connection.set({ connectionId: 'foreground-remote', mode: 'remote', profile: 'active' } as never)
  setApiRequestConnection('foreground-remote')
})

afterEach(() => {
  $connection.set(null)
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
})

it('uses the native bridge for an explicit local owner, even with a remote foreground', async () => {
  await expect(resolveMediaDisplaySrc('/local/image.png', { connectionId: 'local', profile: 'work' })).resolves.toBe(
    localData
  )
  expect(read).toHaveBeenCalledTimes(1)
  expect(read).toHaveBeenCalledWith('/local/image.png')
  expect(api).not.toHaveBeenCalled()
})

it('keeps an explicit remote owner pinned to the API', async () => {
  await expect(
    resolveMediaDisplaySrc('/remote/image.png', { connectionId: 'owner-remote', profile: 'work' })
  ).resolves.toBe(remoteData)
  expect(api).toHaveBeenCalledTimes(1)
  expect(api).toHaveBeenCalledWith({
    connectionId: 'owner-remote',
    profile: 'work',
    path: '/api/fs/read-data-url?path=%2Fremote%2Fimage.png'
  })
  expect(read).not.toHaveBeenCalled()
})

it('never falls back to this device after a remote owner read fails', async () => {
  api.mockRejectedValueOnce(new Error('remote read failed'))
  await expect(
    resolveMediaDisplaySrc('/same/path.png', { connectionId: 'owner-remote', profile: 'work' })
  ).rejects.toThrow('remote read failed')
  expect(read).not.toHaveBeenCalled()
})
