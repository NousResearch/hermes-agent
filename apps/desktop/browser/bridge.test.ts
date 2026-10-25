import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createBrowserHost, registerBrowserFile } from './bridge'
import { setActiveProfile } from './transport'

beforeEach(() => { window.__HERMES_BASE_PATH__ = '/hermes'; setActiveProfile(undefined) })
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

describe('browser host contract', () => {
  it('preserves the server prefix, profile, and spectator route in popout tabs', async () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    await createBrowserHost().openSessionWindow('a/b', { profile: 'coder', watch: true, connectionId: 'browser' })
    const url = new URL(String(open.mock.calls[0][0]))
    expect(url.pathname).toBe('/hermes/')
    expect(url.searchParams.get('profile')).toBe('coder')
    expect(url.searchParams.get('watch')).toBe('1')
    expect(url.searchParams.get('win')).toBe('secondary')
    expect(url.hash).toBe('#/a%2Fb')
  })

  it('inherits the active profile when opening a new tab', async () => {
    setActiveProfile('coder')
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    await createBrowserHost().openWindow()
    const url = new URL(String(open.mock.calls[0][0]))
    expect(url.pathname).toBe('/hermes/')
    expect(url.searchParams.get('profile')).toBe('coder')
  })

  it('uses opaque upload handles for dragged files and expires retained bytes', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now())
    const host = createBrowserHost()
    const path = host.getPathForFile(new File(['text'], 'notes.txt'))
    expect(path).toMatch(/^web-upload:\/\//)
    expect(await host.readFileDataUrlForAttach!(path)).toContain('data:')
    clock.mockReturnValue(Date.now() + 31 * 60_000)
    await expect(host.readFileDataUrlForAttach!(path)).rejects.toThrow('expired')
  })

  it('rejects oversized attachments before reading them into memory', () => {
    const file = new File([''], 'too-large.bin')
    Object.defineProperty(file, 'size', { value: 17 * 1024 * 1024 })
    expect(() => registerBrowserFile(file)).toThrow('16 MB')
  })
})
