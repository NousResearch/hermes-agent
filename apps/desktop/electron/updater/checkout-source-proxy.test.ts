import type * as ChildProcess from 'node:child_process'

import { beforeEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ execute: vi.fn(), proxy: vi.fn() }))

vi.mock('node:child_process', async (importOriginal) => {
  const original = await importOriginal<typeof ChildProcess>()
  const { promisify } = await import('node:util')
  const execFile = Object.assign(vi.fn(), { [promisify.custom]: mocks.execute })

  return { ...original, execFile }
})
vi.mock('../backend-env', () => ({ buildDesktopBackendEnv: () => ({ PATH: '/test-bin' }) }))
vi.mock('../updater-process', () => ({ resolveInstallationLauncher: () => '/test-launcher' }))
vi.mock('./system-proxy-env', () => ({ updateProxyEnv: mocks.proxy }))

import { readSourceUpdate } from './checkout-source'

const probe = { python: 'python', git: 'git', updateRoot: '/nonexistent-proxy-test', hermesHome: '/test-home' }
const success = { stdout: JSON.stringify({ supported: true, updateAvailable: true }), stderr: '' }

beforeEach(() => {
  vi.unstubAllEnvs()
  vi.resetAllMocks()
  mocks.proxy.mockResolvedValue({ HTTPS_PROXY: 'http://system-proxy', NO_PROXY: 'localhost' })
})

it('awaits upstream proxy resolution and retries a reported transport failure once with force', async () => {
  mocks.execute.mockResolvedValueOnce({ stdout: JSON.stringify({ supported: true, error: 'network', message: 'Failed to connect' }) })
    .mockResolvedValueOnce(success)

  expect(await readSourceUpdate(probe)).toMatchObject({ updateAvailable: true })
  expect(mocks.proxy).toHaveBeenCalledOnce()
  expect(mocks.execute).toHaveBeenCalledTimes(2)
  const first = mocks.execute.mock.calls[0]
  const retry = mocks.execute.mock.calls[1]

  expect(first[2].env).toMatchObject({ HTTPS_PROXY: 'http://system-proxy', HERMES_HOME: '/test-home', PATH: '/test-bin' })
  expect(first[1]).not.toContain('--force')
  expect(retry[1]).toContain('--force')
  expect(retry[2].env).toMatchObject({ HTTPS_PROXY: '', https_proxy: '', NO_PROXY: 'localhost', HERMES_HOME: '/test-home' })
})

it('retries thrown transport errors but propagates a second failure', async () => {
  mocks.execute.mockRejectedValue(new Error('Failed to connect'))

  await expect(readSourceUpdate(probe)).rejects.toThrow('Failed to connect')
  expect(mocks.execute).toHaveBeenCalledTimes(2)
})

it('does not retry non-network errors', async () => {
  mocks.execute.mockRejectedValue(new Error('missing launcher'))

  await expect(readSourceUpdate(probe)).rejects.toThrow('missing launcher')
  expect(mocks.execute).toHaveBeenCalledOnce()
})

it('does not retry successful checks', async () => {
  mocks.execute.mockResolvedValue(success)

  expect(await readSourceUpdate(probe)).toMatchObject({ updateAvailable: true })
  expect(mocks.execute).toHaveBeenCalledOnce()
})

it('does not retry transport failures when no proxy was configured', async () => {
  for (const key of Object.keys(process.env)) {
    if (/^(http|https|all)_proxy$/i.test(key)) {
      vi.stubEnv(key, undefined)
    }
  }

  mocks.proxy.mockResolvedValue({})
  mocks.execute.mockRejectedValue(new Error('Failed to connect'))

  await expect(readSourceUpdate(probe)).rejects.toThrow('Failed to connect')
  expect(mocks.execute).toHaveBeenCalledOnce()
  vi.unstubAllEnvs()
})
