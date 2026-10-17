import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

const toggleTerminalPane = vi.hoisted(() => vi.fn())
const togglePanesFlipped = vi.hoisted(() => vi.fn())
const toggleCommandCenter = vi.hoisted(() => vi.fn())

vi.mock('@/app/right-sidebar/terminal/reveal-focus', async importOriginal => ({
  ...(await importOriginal<object>()),
  toggleTerminalPane
}))

vi.mock('@/store/layout', async importOriginal => ({
  ...(await importOriginal<object>()),
  togglePanesFlipped
}))

import { useKeybinds } from '@/app/hooks/use-keybinds'
import { isMacPlatform } from '@/lib/platform'

// #137849: the Ctrl+` terminal toggle must dispatch on layouts where the
// physical Backquote key types a different character. The dispatcher resolves
// the character combo first and falls back to the physical-code combo only when
// the character one is unbound: Czech/Slovak (';') and pt-BR ("'") keydowns —
// unbound character combos — reach the toggle, while Italian ('\') keeps firing
// flip-panes because its character combo IS bound.

const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

let unmountKeybinds: (() => void) | undefined

function press(key: string, code: string, altKey = false) {
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      altKey,
      bubbles: true,
      cancelable: true,
      code,
      ctrlKey: !isMacPlatform(),
      key,
      metaKey: isMacPlatform()
    })
  )
}

beforeEach(() => {
  toggleTerminalPane.mockClear()
  togglePanesFlipped.mockClear()
  toggleCommandCenter.mockClear()
  unmountKeybinds = renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: () => undefined,
        openNewSessionTab: () => undefined,
        requestGateway: <T>() => Promise.resolve(undefined as T),
        startFreshSession: () => undefined,
        toggleCommandCenter,
        toggleSelectedPin: () => undefined
      }),
    { wrapper }
  ).unmount
})

afterEach(() => {
  unmountKeybinds?.()
})

describe('Ctrl+` on layouts where Backquote types another character (#137849)', () => {
  it('toggles the terminal for Czech/Slovak Backquote (types ";")', () => {
    press(';', 'Backquote')

    expect(toggleTerminalPane).toHaveBeenCalledTimes(1)
  })

  it('toggles the terminal for Brazilian Backquote (types "\'")', () => {
    press("'", 'Backquote')

    expect(toggleTerminalPane).toHaveBeenCalledTimes(1)
  })

  it('still toggles the terminal for German Backquote (dead key)', () => {
    press('Dead', 'Backquote')

    expect(toggleTerminalPane).toHaveBeenCalledTimes(1)
  })

  it("keeps flip-panes for Italian Backquote (types '\\') — its character combo is bound", () => {
    press('\\', 'Backquote')

    expect(togglePanesFlipped).toHaveBeenCalledTimes(1)
    expect(toggleTerminalPane).not.toHaveBeenCalled()
  })

  it('does not steal Dvorak Cmd+"." (command center) for the physical V key', () => {
    press('.', 'KeyV')

    expect(toggleCommandCenter).toHaveBeenCalledTimes(1)
  })

  it('fires nothing when neither the character nor the code combo is bound (AltGr-only layouts)', () => {
    press('`', 'Quote', true)

    expect(toggleTerminalPane).not.toHaveBeenCalled()
    expect(togglePanesFlipped).not.toHaveBeenCalled()
  })
})
