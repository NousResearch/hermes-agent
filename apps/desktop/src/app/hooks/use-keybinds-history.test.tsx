import { act, renderHook } from '@testing-library/react'
import type { Terminal } from '@xterm/xterm'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { keybindAction } from '@/lib/keybinds/actions'
import { IS_MAC } from '@/lib/keybinds/combo'
import { $commandPaletteOpen, $commandPalettePage, closeCommandPalette } from '@/store/command-palette'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { closeSwitcher, switcherActive } from '@/store/session-switcher'
import type { SessionInfo } from '@/types/hermes'

import { bindTerminalClipboard } from '../right-sidebar/terminal/terminal-input-bindings'

import { useKeybinds } from './use-keybinds'

const mocks = vi.hoisted(() => ({ openSession: vi.fn() }))
vi.mock('../open-session', () => ({ openSession: mocks.openSession }))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'light', setMode: vi.fn() }) }))

const deps = {
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

function press(key: string, code: string, shiftKey = false) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key, code, ctrlKey: true, shiftKey, bubbles: true }))
  })
}

describe('tab cycling never opens saved sessions', () => {
  let unmount: () => void
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.openSession.mockClear()
    closeSwitcher()
    $sessions.set([{ id: 'current' }, { id: 'saved-only' }] as SessionInfo[])
    $selectedStoredSessionId.set('current')
    $layoutTree.set(group(['workspace']))
    unmount = renderHook(() => useKeybinds(deps), {
      wrapper: ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
    }).unmount
  })
  afterEach(() => {
    unmount()
    closeCommandPalette()
    closeSwitcher()
    $layoutTree.set(null)
    $sessions.set([])
    $selectedStoredSessionId.set(null)
    vi.useRealTimers()
  })

  it('cycles forward and backward through open tabs, wrapping without saved-session navigation', () => {
    for (const id of ['workspace', 'open-a', 'open-b']) {
      registry.register({ area: 'panes', data: { placement: 'main' }, id, render: () => null, title: id })
    }

    $layoutTree.set(group(['workspace', 'open-a', 'open-b'], { active: 'workspace' }))

    const active = () => {
      const tree = $layoutTree.get()

      return tree?.type === 'group' ? tree.active : null
    }

    press('Tab', 'Tab')
    expect(active()).toBe('open-a')
    press('PageDown', 'PageDown')
    expect(active()).toBe('open-b')
    press('Tab', 'Tab')
    expect(active()).toBe('workspace')
    press('Tab', 'Tab', true)
    expect(active()).toBe('open-b')
    press('PageUp', 'PageUp')
    expect(active()).toBe('open-a')
    expect(mocks.openSession).not.toHaveBeenCalled()
    expect(switcherActive()).toBe(false)
  })

  it('uses literal Control+H and preserves palette and HUD shortcuts', () => {
    expect(keybindAction('session.history')?.defaults).toEqual(['ctrl+h'])
    expect(keybindAction('nav.commandPalette')?.defaults).toEqual(['mod+k', 'mod+p'])
    expect(keybindAction('view.toggleHud')?.defaults).toEqual(['mod+shift+h'])

    if (IS_MAC) {
      act(() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'h', code: 'KeyH', metaKey: true })))
      expect($commandPaletteOpen.get()).toBe(false)
    }
  })

  it('Ctrl+H opens the session history page without navigating', () => {
    press('h', 'KeyH')
    expect($commandPaletteOpen.get()).toBe(true)
    expect($commandPalettePage.get()).toBe('history')
    expect(mocks.openSession).not.toHaveBeenCalled()
  })

  it('leaves Control+H exclusively to the interactive shell through the real xterm callback', () => {
    const host = window.document.createElement('div')
    host.dataset.interactiveTerminal = ''
    const input = window.document.createElement('textarea')
    host.append(input)
    window.document.body.append(host)
    input.focus()
    const attachCustomKeyEventHandler = vi.fn()

    const dispose = bindTerminalClipboard(
      { attachCustomKeyEventHandler, getSelection: () => '' } as unknown as Terminal,
      host,
      vi.fn(),
      vi.fn(),
      () => true
    )

    const callback = attachCustomKeyEventHandler.mock.calls[0]![0] as (event: KeyboardEvent) => boolean
    const pty = vi.fn()
    input.addEventListener('keydown', event => {
      if (callback(event)) {
        pty('\x08')
      }
    })
    act(() =>
      input.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'h', code: 'KeyH', ctrlKey: true, bubbles: true, cancelable: true })
      )
    )
    expect(pty).toHaveBeenCalledExactlyOnceWith('\x08')
    expect($commandPaletteOpen.get()).toBe(false)
    dispose()
    host.remove()
  })

  it('keeps Control+H history available on a noninteractive agent terminal', () => {
    const host = window.document.createElement('div')
    host.dataset.terminal = ''
    host.tabIndex = 0
    window.document.body.append(host)
    host.focus()
    act(() =>
      host.dispatchEvent(new KeyboardEvent('keydown', { key: 'h', code: 'KeyH', ctrlKey: true, bubbles: true }))
    )
    expect($commandPalettePage.get()).toBe('history')
    host.remove()
  })

  it.each([
    ['Tab', 'Tab', false],
    ['Tab', 'Tab', true],
    ['PageDown', 'PageDown', false],
    ['PageUp', 'PageUp', false]
  ])('a single open tab stays put for Ctrl+%s (shift=%s)', (key, code, shift) => {
    press(key, code, shift)
    act(() => vi.advanceTimersByTime(500))
    expect(mocks.openSession).not.toHaveBeenCalled()
    expect(switcherActive()).toBe(false)
  })
})
