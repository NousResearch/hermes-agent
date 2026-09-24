import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { useKeybinds } from '@/app/hooks/use-keybinds'
import { setTerminalTakeover } from '@/app/right-sidebar/store'
import { useStatusbarItems } from '@/app/shell/hooks/use-statusbar-items'
import { collapseTreePane, isPaneVisible, revealTreePane, setPaneCollapsed } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

// Ctrl+`, the ⌘K row, and the statusbar pill reveal the terminal. That reveal
// must make the on-screen xterm the keyboard target, and the claim has to
// survive the composer taking focus back on the next frame. Hiding must not.

const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

let unmountKeybinds: (() => void) | undefined

beforeAll(async () => {
  await import('@/app/contrib/controller')
})

beforeEach(() => {
  // Testing Library unmounts renderHook trees after each test, so the keybind
  // listener has to be installed again or later presses never dispatch.
  unmountKeybinds = renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: () => undefined,
        openNewSessionTab: () => undefined,
        startFreshSession: () => undefined,
        toggleCommandCenter: () => undefined,
        toggleSelectedPin: () => undefined
      }),
    { wrapper }
  ).unmount
})

afterAll(() => {
  unmountKeybinds?.()
})

afterEach(() => {
  vi.unstubAllGlobals()
  document.body.replaceChildren()
  setTerminalTakeover(false)
  setPaneCollapsed('terminal', true)
})

function installRaf() {
  const frames: FrameRequestCallback[] = []
  let nextId = 1

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.push(callback)

    return nextId++
  })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)

  return () => frames.splice(0).forEach(callback => callback(0))
}

function mountTerminal(visible: boolean): HTMLTextAreaElement {
  const host = document.createElement('div')

  host.dataset.terminal = ''

  if (!visible) {
    host.className = 'invisible pointer-events-none'
  }

  const textarea = document.createElement('textarea')

  textarea.className = 'xterm-helper-textarea'
  host.append(textarea)
  document.body.append(host)

  return textarea
}

function mountComposer(): HTMLTextAreaElement {
  const composer = document.createElement('textarea')

  composer.dataset.composer = ''
  document.body.append(composer)

  return composer
}

function pressShowTerminal() {
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      code: 'Backquote',
      ctrlKey: true,
      key: '`'
    })
  )
}

function paletteRun() {
  const row = registry.getArea(PALETTE_AREA).find(item => item.id === 'view.showTerminal')
  const run = (row?.data as PaletteContribution | undefined)?.run

  if (!run) {
    throw new Error('palette toggle view.showTerminal is not registered')
  }

  run()
}

function statusbarSelect() {
  const { result } = renderHook(
    () =>
      useStatusbarItems({
        agentsOpen: false,
        chatOpen: true,
        commandCenterOpen: false,
        extraLeftItems: [],
        extraRightItems: [],
        freshDraftReady: false,
        gatewayState: 'ready',
        inferenceStatus: null,
        openAgents: () => undefined,
        openCommandCenterSection: () => undefined,
        requestGateway: async () => undefined as never,
        statusSnapshot: null,
        toggleCommandCenter: () => undefined
      }),
    { wrapper }
  )

  const terminal = result.current.statusbarItems.find(item => item.id === 'terminal')

  if (!terminal?.onSelect) {
    throw new Error('statusbar terminal toggle is not registered')
  }

  terminal.onSelect({ shiftKey: false })
}

function hideTerminal() {
  setTerminalTakeover(false)
  setPaneCollapsed('terminal', true)

  if (isPaneVisible('terminal')) {
    collapseTreePane('terminal')
  }
}

function showTerminalPane() {
  setTerminalTakeover(true)

  if (!isPaneVisible('terminal')) {
    revealTreePane('terminal')
  }
}

describe('terminal reveal keyboard focus', () => {
  it('takes focus from the keybind on reveal and keeps it when the composer steals it', () => {
    const frame = installRaf()

    hideTerminal()
    expect(isPaneVisible('terminal')).toBe(false)

    const hidden = mountTerminal(false)
    const visible = mountTerminal(true)
    const composer = mountComposer()

    composer.focus()
    pressShowTerminal()
    frame()

    expect(document.activeElement).toBe(visible)
    expect(document.activeElement).not.toBe(hidden)
    expect(document.activeElement).not.toBe(composer)

    composer.focus()
    expect(document.activeElement).toBe(composer)

    frame()
    expect(document.activeElement).toBe(visible)
  })

  it('returns focus to the terminal on every later reveal from the keybind', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const composer = mountComposer()

    hideTerminal()
    expect(isPaneVisible('terminal')).toBe(false)
    composer.focus()
    pressShowTerminal()
    frame()
    expect(document.activeElement).toBe(visible)

    showTerminalPane()
    expect(isPaneVisible('terminal')).toBe(true)
    composer.focus()
    pressShowTerminal()
    frame()
    frame()
    frame()
    expect(document.activeElement).toBe(composer)
    expect(isPaneVisible('terminal')).toBe(false)

    pressShowTerminal()
    frame()
    expect(document.activeElement).toBe(visible)
  })

  it('does not keep refocusing once the terminal already holds the keyboard', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const focusSpy = vi.spyOn(visible, 'focus')

    hideTerminal()
    expect(isPaneVisible('terminal')).toBe(false)
    pressShowTerminal()
    frame()
    frame()
    frame()

    expect(focusSpy).toHaveBeenCalledTimes(1)
  })

  it('takes focus from the palette toggle on reveal and re-asserts if the composer steals it', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const composer = mountComposer()

    hideTerminal()
    expect(isPaneVisible('terminal')).toBe(false)
    composer.focus()
    paletteRun()
    frame()

    expect(document.activeElement).toBe(visible)

    composer.focus()
    frame()
    expect(document.activeElement).toBe(visible)
  })

  it('does not move focus onto the terminal when the palette toggle hides it', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const composer = mountComposer()

    showTerminalPane()
    expect(isPaneVisible('terminal')).toBe(true)
    composer.focus()
    paletteRun()
    frame()
    frame()
    frame()

    expect(document.activeElement).toBe(composer)
    expect(document.activeElement).not.toBe(visible)
    expect(isPaneVisible('terminal')).toBe(false)
  })

  it('takes focus from the statusbar toggle on reveal', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const composer = mountComposer()

    hideTerminal()
    composer.focus()
    statusbarSelect()
    frame()

    expect(document.activeElement).toBe(visible)

    composer.focus()
    frame()
    expect(document.activeElement).toBe(visible)
  })
})
