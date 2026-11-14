import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { useKeybinds } from '@/app/hooks/use-keybinds'
import { $terminalCollapsedRail, $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { terminalPaletteToggle } from '@/app/right-sidebar/terminal/reveal-focus'
import { useStatusbarItems } from '@/app/shell/hooks/use-statusbar-items'
import { group, split } from '@/components/pane-shell/tree/model'
import {
  $hiddenTreePanes,
  $layoutTree,
  bindToolPaneCollapse,
  collapseTreePane,
  isPaneVisible,
  revealTreePane,
  setPaneCollapsed
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $interfaceMode } from '@/store/interface-mode'
import { $shellChoice } from '@/store/shell-variant'

import { $activeTerminalId, $terminals } from './terminals'

// Ctrl+`, the ⌘K row, and the statusbar pill reveal the terminal. That reveal
// must make the on-screen xterm the keyboard target, and the claim has to
// survive the composer taking focus back on the next frame. Hiding must not.

const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

let unmountKeybinds: (() => void) | undefined

// The contrib controller owns this wiring in the app, but importing it pulls
// the whole app graph (every pane, bundled plugin, HUD) into a hook: minutes of
// cold transform under the parallel ui run, which blew the 30s hook timeout in
// CI. Wire the terminal slice here through the same production functions and
// the same palette row the controller registers.
// The arrangement is scoped per shell, so a test that flips the shell seeds the scope it lands in.
function seedLayout() {
  $layoutTree.set(
    split('column', [
      group(['workspace'], { active: 'workspace', id: 'grp-main' }),
      group(['terminal'], { active: 'terminal', id: 'grp-terminal' })
    ])
  )
}

beforeAll(() => {
  registry.registerMany([
    {
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => null,
      title: 'workspace'
    },
    {
      area: 'panes',
      data: { placement: 'bottom', lifecycleKeepAlive: true },
      id: 'terminal',
      render: () => null,
      title: 'terminal'
    },
    terminalPaletteToggle
  ])
  seedLayout()
  bindToolPaneCollapse(
    'terminal',
    $terminalTakeover,
    () => setTerminalTakeover(false),
    () => setTerminalTakeover(true),
    $terminalCollapsedRail
  )
})

beforeEach(() => {
  // Testing Library unmounts renderHook trees after each test, so the keybind
  // listener has to be installed again or later presses never dispatch.
  unmountKeybinds = renderHook(
    () =>
      useKeybinds({
        requestGateway: vi.fn(),
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
        changeSessionCwd: async () => {},
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

it.each([false, true])('hides the closed terminal rail only in the redesign (%s), preserving shells on reopen', redesigned => {
  installRaf()
  $interfaceMode.set('advanced')
  $shellChoice.set(redesigned)
  seedLayout()
  const terminal = { id: 'retained-shell', kind: 'user' as const, title: 'Shell', cwd: '/tmp', auto: true }
  $terminals.set([terminal])
  $activeTerminalId.set(terminal.id)

  try {
    showTerminalPane()
    paletteRun()
    expect(isPaneVisible('terminal')).toBe(false)
    expect($hiddenTreePanes.get().has('terminal')).toBe(redesigned)
    paletteRun()
    expect(isPaneVisible('terminal')).toBe(true)
    expect($hiddenTreePanes.get().has('terminal')).toBe(false)
    expect($terminals.get()).toEqual([terminal])
    expect($activeTerminalId.get()).toBe(terminal.id)
  } finally {
    $terminals.set([])
    $activeTerminalId.set(null)
    $shellChoice.set(null)
  }
})
