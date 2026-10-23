import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { settings } from './settings'
import { setActiveProfile } from './transport'

beforeEach(() => {
  localStorage.clear()
  window.__HERMES_BASE_PATH__ = ''
  setActiveProfile('coder')
})
afterEach(() => vi.unstubAllGlobals())

it('validates backend directories and isolates defaults by profile and mount', async () => {
  vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(Response.json({ cwd: '/backend', entries: [] }))))
  await settings!.setDefaultProjectDir('/projects/coder')
  expect(await settings!.getDefaultProjectDir()).toMatchObject({ dir: '/projects/coder', resolvedCwd: '/projects/coder' })
  setActiveProfile('reviewer')
  expect(await settings!.getDefaultProjectDir()).toMatchObject({ dir: null, resolvedCwd: '/backend' })
  setActiveProfile('coder')
  window.__HERMES_BASE_PATH__ = '/other'
  expect(await settings!.getDefaultProjectDir()).toMatchObject({ dir: null })
  window.__HERMES_BASE_PATH__ = ''
  await settings!.setDefaultProjectDir(null)
  expect(await settings!.getDefaultProjectDir()).toMatchObject({ dir: null })
})

it('does not persist directories rejected by the backend', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ error: 'not a directory' })))
  await expect(settings!.setDefaultProjectDir('/missing')).rejects.toThrow('not a directory')
  expect(localStorage.length).toBe(0)
})

it('keeps an in-flight settings read on its original profile', async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ cwd: '/backend', entries: [] }))
  vi.stubGlobal('fetch', fetch)
  await settings!.setDefaultProjectDir('/projects/coder')
  let finish!: (value: Response) => void
  fetch.mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
  const pending = settings!.getDefaultProjectDir()
  setActiveProfile('reviewer')
  finish(Response.json({ cwd: '/coder-backend' }))
  expect(await pending).toMatchObject({ dir: '/projects/coder', defaultLabel: '/coder-backend' })
  expect(fetch.mock.calls.at(-1)?.[0]).toContain('profile=coder')
})
