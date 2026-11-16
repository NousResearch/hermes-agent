import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Chords Chrome, Edge and Firefox act on before the page's keydown handler can
// cancel them (new window/tab/private window, close tab/window, reopen tab,
// tab cycling, quit), in the canonical form `comboFromEvent` emits per OS.
// Alt+Tab is the OS window switcher off macOS.
const BROWSER_RESERVED = {
  mac: [
    'mod+n',
    'mod+t',
    'mod+w',
    'mod+q',
    'mod+shift+n',
    'mod+shift+t',
    'mod+shift+w',
    'mod+shift+p',
    'ctrl+tab',
    'ctrl+shift+tab'
  ],
  other: [
    'mod+n',
    'mod+t',
    'mod+w',
    'mod+shift+n',
    'mod+shift+t',
    'mod+shift+w',
    'mod+shift+p',
    'mod+shift+q',
    'mod+tab',
    'mod+shift+tab',
    'mod+pagedown',
    'mod+pageup',
    'alt+tab',
    'alt+shift+tab'
  ]
}

const shortcuts = [
  { id: 'session.new', native: ['mod+n'], mac: 'mod+ctrl+n', other: 'alt+n' },
  { id: 'session.newTab', native: ['mod+t'], mac: 'mod+ctrl+t', other: 'alt+t' },
  { id: 'session.newWindow', native: ['mod+shift+n'], mac: 'mod+ctrl+shift+n', other: 'alt+shift+n' },
  { id: 'session.next', native: ['ctrl+tab', 'ctrl+pagedown'], mac: 'mod+ctrl+]', other: 'alt+]' },
  { id: 'session.prev', native: ['ctrl+shift+tab', 'ctrl+pageup'], mac: 'mod+ctrl+[', other: 'alt+[' },
  { id: 'view.closeTab', native: ['mod+w'], mac: 'mod+ctrl+w', other: 'alt+w' },
  { id: 'view.reopenTab', native: ['mod+shift+t'], mac: 'mod+ctrl+shift+t', other: 'alt+shift+t' },
  // ⌃⇧W is literal Control on macOS, which the browser leaves to the page.
  { id: 'view.closeTerminal', native: ['ctrl+shift+w'], mac: 'ctrl+shift+w', other: 'alt+shift+w' }
]

// The keydown a real keyboard sends for a combo: macOS reports Cmd (`mod`)
// and Control separately; Windows and Linux report the key with Alt held.
function keydownFor(combo: string): KeyboardEvent {
  const parts = combo.split('+')
  const key = parts.pop()!
  const code = /^[a-z]$/.test(key) ? `Key${key.toUpperCase()}` : key === ']' ? 'BracketRight' : 'BracketLeft'

  return new KeyboardEvent('keydown', {
    altKey: parts.includes('alt'),
    code,
    ctrlKey: parts.includes('ctrl'),
    key,
    metaKey: parts.includes('mod'),
    shiftKey: parts.includes('shift')
  })
}

const setPlatform = (value: string) => Object.defineProperty(navigator, 'platform', { configurable: true, value })

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

afterEach(() => {
  delete window.document.documentElement.dataset.hermesDesktopHost
  setPlatform('')
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('browser-hosted shortcut defaults', () => {
  describe.each([
    ['macOS', 'MacIntel'],
    ['Windows and Linux', 'Win32']
  ])('on %s', (_, platform) => {
    const mac = platform === 'MacIntel'

    it('ships no default the browser takes first, and leaves native defaults alone', async () => {
      setPlatform(platform)
      window.document.documentElement.dataset.hermesDesktopHost = 'browser'
      const browser = await import('./actions')
      const { canonicalizeCombo } = await import('./combo')
      const reserved = new Set(BROWSER_RESERVED[mac ? 'mac' : 'other'])

      const offenders = browser
        .allKeybindActions()
        .flatMap(action => action.defaults.map(combo => `${action.id}=${canonicalizeCombo(combo)}`))
        .filter(entry => reserved.has(entry.split('=')[1]))

      expect(offenders).toEqual([])

      delete window.document.documentElement.dataset.hermesDesktopHost
      vi.resetModules()
      const native = await import('./actions')

      for (const shortcut of shortcuts) {
        expect(native.keybindAction(shortcut.id)?.defaults).toEqual(shortcut.native)
      }

      const changed = native.KEYBIND_ACTIONS.filter(
        action => JSON.stringify(action.defaults) !== JSON.stringify(browser.keybindAction(action.id)?.defaults)
      ).map(action => action.id)

      expect(changed.sort()).toEqual(
        shortcuts
          .filter(shortcut => !(mac && shortcut.mac === shortcut.native[0]))
          .map(shortcut => shortcut.id)
          .sort()
      )
    })

    it.each(['webapp', 'authenticated', 'bridge'])(
      'resolves browser defaults before the store starts in the %s host, and a real keydown dispatches them',
      async host => {
        setPlatform(platform)

        if (host === 'bridge') {
          window.document.documentElement.dataset.hermesDesktopHost = 'browser'
        } else {
          vi.stubGlobal('hermesDesktop', undefined)
          vi.stubGlobal(
            host === 'webapp' ? '__HERMES_UI_SURFACE__' : '__HERMES_AUTH_REQUIRED__',
            host === 'webapp' ? 'webapp' : true
          )
        }

        const { actionAllowedInInput, comboFromEvent, formatCombo } = await import('./combo')
        const { $comboIndex, bindingsFor, conflictsFor, resetBinding, setBinding } = await import('@/store/keybinds')

        for (const shortcut of shortcuts) {
          const expected = mac ? shortcut.mac : shortcut.other
          expect(bindingsFor(shortcut.id)).toEqual([expected])
          expect(actionAllowedInInput(shortcut.id, expected)).toBe(true)
          expect(conflictsFor(shortcut.id, expected)).toEqual([])

          for (const native of shortcut.native.filter(combo => combo !== expected)) {
            expect($comboIndex.get().has(native)).toBe(false)
          }

          if (mac && expected.startsWith('mod+ctrl+')) {
            expect(formatCombo(expected).startsWith('⌃⌘')).toBe(true)
          }

          if (!mac) {
            expect(formatCombo(expected).startsWith('Alt+')).toBe(true)
          }

          expect($comboIndex.get().get(comboFromEvent(keydownFor(expected))!)).toEqual([shortcut.id])

          setBinding(shortcut.id, ['mod+shift+f6'])
          expect(bindingsFor(shortcut.id)).toEqual(['mod+shift+f6'])
          resetBinding(shortcut.id)
          expect(bindingsFor(shortcut.id)).toEqual([expected])
        }
      }
    )
  })

  it('preserves saved remaps and explicit unbinding across a browser reload', async () => {
    setPlatform('Win32')
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    localStorage.setItem(
      'hermes.desktop.keybinds',
      JSON.stringify({
        'session.new': ['mod+shift+f6'],
        'session.newTab': [],
        'view.closeTab': ['mod+w']
      })
    )
    const { bindingsFor, resetAllBindings } = await import('@/store/keybinds')

    expect(bindingsFor('session.new')).toEqual(['mod+shift+f6'])
    expect(bindingsFor('session.newTab')).toEqual([])
    expect(bindingsFor('view.closeTab')).toEqual(['mod+w'])
    resetAllBindings()
    expect(bindingsFor('session.new')).toEqual(['alt+n'])
    expect(JSON.parse(localStorage.getItem('hermes.desktop.keybinds')!)).toEqual({})
  })
})
