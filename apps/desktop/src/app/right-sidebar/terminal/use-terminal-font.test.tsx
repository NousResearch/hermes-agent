// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $terminalFontFamily, setTerminalFontFamilyFromConfig } from './terminal-font'
import { useTerminalFontController } from './use-terminal-font'

describe('useTerminalFontController', () => {
  afterEach(() => setTerminalFontFamilyFromConfig(''))

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
})
