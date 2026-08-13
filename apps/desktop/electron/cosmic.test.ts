import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { execFile } from 'node:child_process'

import { isCosmic, readCosmicWindows } from './cosmic'
import type { EnumeratedWindow } from './window-below'

const win = (pid: number, app = `app-${pid}`): EnumeratedWindow => ({
  app,
  bounds: { x: 0, y: 0, width: 800, height: 600 },
  id: pid * 10,
  pid,
  title: `${app} window`
})

vi.mock('node:child_process', () => ({
  execFile: vi.fn()
}))
vi.mock('node:util', () => ({
  promisify: (fn: unknown) => fn
}))

const mockedExecFile = vi.mocked(execFile)

describe('isCosmic', () => {
  it('detects COSMIC from XDG_CURRENT_DESKTOP', () => {
    expect(isCosmic({ XDG_CURRENT_DESKTOP: 'COSMIC' })).toBe(true)
  })

  it('detects COSMIC from XDG_SESSION_DESKTOP', () => {
    expect(isCosmic({ XDG_SESSION_DESKTOP: 'cosmic' })).toBe(true)
  })

  it('is case-insensitive', () => {
    expect(isCosmic({ XDG_CURRENT_DESKTOP: 'COSMIC' })).toBe(true)
    expect(isCosmic({ XDG_CURRENT_DESKTOP: 'pop-cosmic' })).toBe(true)
  })

  it('returns false on non-COSMIC sessions', () => {
    expect(isCosmic({ XDG_CURRENT_DESKTOP: 'GNOME' })).toBe(false)
    expect(isCosmic({ XDG_CURRENT_DESKTOP: 'Hyprland' })).toBe(false)
    expect(isCosmic({})).toBe(false)
  })
})

describe('readCosmicWindows', () => {
  beforeEach(() => {
    mockedExecFile.mockReset()
  })

  afterEach(() => {
    mockedExecFile.mockReset()
  })

  it('returns null off COSMIC so the established path is untouched', async () => {
    const enumerate = () => Promise.resolve([win(1)])

    expect(await readCosmicWindows(42, true, { XDG_CURRENT_DESKTOP: 'GNOME' }, enumerate)).toBeNull()
    expect(await readCosmicWindows(42, true, {}, enumerate)).toBeNull()
  })

  it('prefers the native COSMIC helper when available', async () => {
    const enumerate = vi.fn(() => Promise.resolve([win(1)]))
    mockedExecFile.mockResolvedValue({
      stdout: JSON.stringify([
        { title: 'Cosmic Term', app_id: 'com.system76.CosmicTerm', identifier: 'abc', geometry: null },
        { title: 'Sheets', app_id: 'brave-browser', identifier: 'def', geometry: null }
      ])
    } as never)

    const result = await readCosmicWindows(42, true, { XDG_CURRENT_DESKTOP: 'COSMIC' }, enumerate)

    expect(result).not.toBeNull()
    expect(result).toHaveLength(2)
    expect(result?.[0]).toMatchObject({
      app: 'com.system76.CosmicTerm',
      title: 'Cosmic Term'
    })
    // geometry/pid are placeholders on native Wayland (COSMIC 1.0 limitation)
    expect(result?.[0].bounds).toEqual({ x: 0, y: 0, width: 0, height: 0 })
    expect(result?.[0].pid).toBe(0)
    // X11 enumerator must NOT have been called when the helper answered
    expect(enumerate).not.toHaveBeenCalled()
  })

  it('falls back to the X11 enumerator when the helper fails', async () => {
    const enumerated: EnumeratedWindow[] = [win(1), win(2)]
    const enumerate = vi.fn(() => Promise.resolve(enumerated))
    mockedExecFile.mockRejectedValue(new Error('cosmic-toplevel-list: not found'))

    const result = await readCosmicWindows(42, true, { XDG_CURRENT_DESKTOP: 'COSMIC' }, enumerate)

    expect(result).toBe(enumerated)
    expect(enumerate).toHaveBeenCalledOnce()
  })

  it('falls back to the X11 enumerator when the helper returns no windows', async () => {
    const enumerated: EnumeratedWindow[] = [win(1)]
    const enumerate = vi.fn(() => Promise.resolve(enumerated))
    mockedExecFile.mockResolvedValue({ stdout: '[]' } as never)

    const result = await readCosmicWindows(42, true, { XDG_CURRENT_DESKTOP: 'COSMIC' }, enumerate)

    expect(result).toBe(enumerated)
    expect(enumerate).toHaveBeenCalledOnce()
  })

  it('surfaces the enumerator returning null on native-Wayland COSMIC', async () => {
    const enumerate = () => Promise.resolve(null)
    mockedExecFile.mockRejectedValue(new Error('no helper'))

    expect(await readCosmicWindows(42, true, { XDG_CURRENT_DESKTOP: 'COSMIC' }, enumerate)).toBeNull()
  })
})
