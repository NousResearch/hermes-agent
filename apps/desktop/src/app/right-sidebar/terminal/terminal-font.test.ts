import { describe, expect, it, vi } from 'vitest'

import {
  applyTerminalFontFamily,
  DEFAULT_TERMINAL_FONT_FAMILY,
  prepareTerminalFontFamily,
  resolveTerminalFontFamily,
  warmTerminalFontFamily
} from './terminal-font'

describe('terminal font resolution', () => {
  it('keeps the bundled stack when no preference is configured', () => {
    expect(resolveTerminalFontFamily('')).toBe(DEFAULT_TERMINAL_FONT_FAMILY)
    expect(resolveTerminalFontFamily(undefined)).toBe(DEFAULT_TERMINAL_FONT_FAMILY)
  })

  it('quotes a friendly family name and appends the bundled fallback stack', () => {
    expect(resolveTerminalFontFamily('  MesloLGS NF  ')).toBe(`'MesloLGS NF', ${DEFAULT_TERMINAL_FONT_FAMILY}`)
  })

  it('preserves an authored CSS stack before the bundled fallbacks', () => {
    expect(resolveTerminalFontFamily("'Hack Nerd Font', monospace")).toBe(
      `'Hack Nerd Font', monospace, ${DEFAULT_TERMINAL_FONT_FAMILY}`
    )
  })
})

describe('terminal font lifecycle', () => {
  it('warms regular, bold, and italic faces using the effective stack', async () => {
    const load = vi.fn().mockResolvedValue([])

    await warmTerminalFontFamily("'MesloLGS NF', monospace", { load } as Pick<FontFaceSet, 'load'>)

    expect(load.mock.calls.map(([descriptor]) => descriptor)).toEqual([
      "400 11px 'MesloLGS NF', monospace",
      "700 11px 'MesloLGS NF', monospace",
      "italic 400 11px 'MesloLGS NF', monospace"
    ])
  })

  it('restarts initial warming when config arrives late', async () => {
    let latest = 'fallback'

    const warm = vi.fn(async (fontFamily: string) => {
      if (fontFamily === 'fallback') {
        latest = 'MesloLGS NF'
      }
    })

    await expect(
      prepareTerminalFontFamily(
        () => latest,
        () => true,
        warm
      )
    ).resolves.toBe('MesloLGS NF')
    expect(warm.mock.calls.map(([font]) => font)).toEqual(['fallback', 'MesloLGS NF'])
  })

  it('cancels a stale initial font request before xterm mounts', async () => {
    let current = true

    const warm = vi.fn(async () => {
      current = false
    })

    await expect(
      prepareTerminalFontFamily(
        () => 'MesloLGS NF',
        () => current,
        warm
      )
    ).resolves.toBeNull()
  })

  it('updates a mounted terminal without replacing it', async () => {
    const term = {
      options: { fontFamily: 'fallback' },
      rows: 24,
      refresh: vi.fn()
    }

    const fit = vi.fn()
    const clearTextureAtlas = vi.fn()

    await expect(
      applyTerminalFontFamily({
        clearTextureAtlas,
        fit,
        fontFamily: 'MesloLGS NF',
        isCurrent: () => true,
        term,
        warm: vi.fn().mockResolvedValue(undefined)
      })
    ).resolves.toBe(true)

    expect(term.options.fontFamily).toBe('MesloLGS NF')
    expect(fit).toHaveBeenCalledOnce()
    expect(clearTextureAtlas).toHaveBeenCalledOnce()
    expect(term.refresh).toHaveBeenCalledWith(0, 23)
  })

  it('does not paint a stale live font request', async () => {
    const term = {
      options: { fontFamily: 'newer' },
      rows: 24,
      refresh: vi.fn()
    }

    await expect(
      applyTerminalFontFamily({
        clearTextureAtlas: vi.fn(),
        fit: vi.fn(),
        fontFamily: 'stale',
        isCurrent: () => false,
        term,
        warm: vi.fn().mockResolvedValue(undefined)
      })
    ).resolves.toBe(false)

    expect(term.options.fontFamily).toBe('newer')
    expect(term.refresh).not.toHaveBeenCalled()
  })
})
