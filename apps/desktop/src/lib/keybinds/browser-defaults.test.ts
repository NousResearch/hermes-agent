import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const shortcuts = [
  { id: 'session.new', native: 'mod+n', mac: 'mod+ctrl+n', other: 'mod+shift+f7' },
  { id: 'session.newTab', native: 'mod+t', mac: 'mod+ctrl+t', other: 'mod+shift+f8' },
  { id: 'view.closeTab', native: 'mod+w', mac: 'mod+ctrl+w', other: 'mod+shift+f9' }
]

// The keydown each platform's browser default arrives as: macOS reports Cmd and
// Control separately; Windows and Linux report Control (or Meta) with Shift.
const keydownFor = (combo: string, mac: boolean) => {
  const key = combo.split('+').at(-1)!
  const code = key.length === 1 ? `Key${key.toUpperCase()}` : key.toUpperCase()

  return mac
    ? [new KeyboardEvent('keydown', { key, code, metaKey: true, ctrlKey: true })]
    : [{ metaKey: true }, { ctrlKey: true }].map(m => new KeyboardEvent('keydown', { key: code, code, shiftKey: true, ...m }))
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

describe('browser session shortcuts', () => {
  describe.each([['macOS', 'MacIntel'], ['Windows and Linux', 'Win32']])('on %s', (_, platform) => {
    const mac = platform === 'MacIntel'

    it.each(['webapp', 'authenticated', 'bridge', 'native'])(
      'resolves defaults before the store starts in the %s host',
      async host => {
        setPlatform(platform)

        if (host === 'bridge') {
          window.document.documentElement.dataset.hermesDesktopHost = 'browser'
        } else if (host !== 'native') {
          vi.stubGlobal('hermesDesktop', undefined)
          vi.stubGlobal(
            host === 'webapp' ? '__HERMES_UI_SURFACE__' : '__HERMES_AUTH_REQUIRED__',
            host === 'webapp' ? 'webapp' : true
          )
        }

        const { actionAllowedInInput, comboFromEvent, formatCombo } = await import('./combo')
        const { $comboIndex, bindingsFor, conflictsFor, resetBinding, setBinding } = await import('@/store/keybinds')

        for (const shortcut of shortcuts) {
          const expected = host === 'native' ? shortcut.native : mac ? shortcut.mac : shortcut.other
          expect(bindingsFor(shortcut.id)).toEqual([expected])
          expect(actionAllowedInInput(shortcut.id, expected)).toBe(true)
          expect(conflictsFor(shortcut.id, expected)).toEqual([])

          if (host !== 'native') {
            expect($comboIndex.get().has(shortcut.native)).toBe(false)

            if (mac) {
              expect(formatCombo(expected)).toBe(`⌃⌘${shortcut.native.at(-1)!.toUpperCase()}`)
            }

            for (const event of keydownFor(expected, mac)) {
              expect($comboIndex.get().get(comboFromEvent(event)!)).toEqual([shortcut.id])
            }
          }

          setBinding(shortcut.id, ['mod+shift+f6'])
          expect(bindingsFor(shortcut.id)).toEqual(['mod+shift+f6'])
          resetBinding(shortcut.id)
          expect(bindingsFor(shortcut.id)).toEqual([expected])
        }
      }
    )
  })

  it('preserves saved remaps and explicit unbinding across a browser reload', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    localStorage.setItem('hermes.desktop.keybinds', JSON.stringify({
      'session.new': ['mod+shift+f6'],
      'session.newTab': [],
      'view.closeTab': ['mod+w']
    }))
    const { bindingsFor, resetAllBindings } = await import('@/store/keybinds')

    expect(bindingsFor('session.new')).toEqual(['mod+shift+f6'])
    expect(bindingsFor('session.newTab')).toEqual([])
    expect(bindingsFor('view.closeTab')).toEqual(['mod+w'])
    resetAllBindings()
    expect(bindingsFor('session.new')).toEqual(['mod+shift+f7'])
    expect(JSON.parse(localStorage.getItem('hermes.desktop.keybinds')!)).toEqual({})
  })
})
