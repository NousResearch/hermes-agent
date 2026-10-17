// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $terminalFontSize, setTerminalFontSize } from '@/store/terminal-font-size'

import { $terminalFontFamily, setTerminalFontFamilyFromConfig } from './terminal-font'
import { useTerminalFontController } from './use-terminal-font'

const terminals = vi.hoisted(() => ({ redrawAllTerminals: vi.fn() }))

vi.mock('./terminals', () => ({ redrawAllTerminals: terminals.redrawAllTerminals }))

describe('useTerminalFontController', () => {
  afterEach(() => {
    setTerminalFontFamilyFromConfig('')
    setTerminalFontSize(11)
    terminals.redrawAllTerminals.mockClear()
  })

  it('repaints an already-mounted xterm when the profile font changes', async () => {
    const term = {
      options: { fontFamily: 'fallback' },
      refresh: vi.fn(),
      rows: 18
    }

    const fit = vi.fn()

    const clearTextureAtlas = vi.fn()

    const refs = {
      fitRef: { current: fit },
      termRef: { current: term },
      webglRef: { current: { clearTextureAtlas } }
    }

    const { result } = renderHook(() =>
      useTerminalFontController(refs as unknown as Parameters<typeof useTerminalFontController>[0])
    )

    result.current.mountedRef.current = true
    act(() => $terminalFontFamily.set('MesloLGS NF'))

    await waitFor(() => expect(term.options.fontFamily).toContain('MesloLGS NF'))
    expect(fit).toHaveBeenCalledOnce()
    expect(clearTextureAtlas).toHaveBeenCalledOnce()
    expect(term.refresh).toHaveBeenCalledWith(0, 17)
  })

  it('applies a live font size change to a mounted xterm and refits', () => {
    const term = { options: { fontFamily: 'x', fontSize: 11 }, refresh: vi.fn(), rows: 10 }
    const fit = vi.fn()
    const clearTextureAtlas = vi.fn()

    const refs = {
      fitRef: { current: fit },
      termRef: { current: term },
      webglRef: { current: { clearTextureAtlas } }
    }

    const { result } = renderHook(() =>
      useTerminalFontController(refs as unknown as Parameters<typeof useTerminalFontController>[0])
    )

    expect(result.current.latestFontSizeRef.current).toBe(11)

    result.current.mountedRef.current = true
    terminals.redrawAllTerminals.mockClear()
    act(() => setTerminalFontSize(14))

    expect(term.options.fontSize).toBe(14)
    expect(result.current.latestFontSizeRef.current).toBe(14)
    expect(fit).toHaveBeenCalledWith(true)
    expect(clearTextureAtlas).toHaveBeenCalledOnce()
    expect(terminals.redrawAllTerminals.mock.calls).toEqual([[]])

    act(() => setTerminalFontSize(11))
    expect($terminalFontSize.get()).toBe(11)
  })
})
