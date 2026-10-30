import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// The Alt (and ⌃⌘) browser defaults exist because a browser TAB acts on these
// Ctrl chords before the page sees them. A Chrome app window reserves none of
// them, so off macOS it ships the same Ctrl chords native Desktop does.
const shortcuts = [
  { id: 'session.new', native: ['mod+n'], mac: 'mod+ctrl+n', tab: 'alt+n' },
  { id: 'session.newTab', native: ['mod+t'], mac: 'mod+ctrl+t', tab: 'alt+t' },
  { id: 'session.newWindow', native: ['mod+shift+n'], mac: 'mod+ctrl+shift+n', tab: 'alt+shift+n' },
  { id: 'session.next', native: ['ctrl+tab', 'ctrl+pagedown'], mac: 'mod+ctrl+]', tab: 'alt+]' },
  { id: 'session.prev', native: ['ctrl+shift+tab', 'ctrl+pageup'], mac: 'mod+ctrl+[', tab: 'alt+[' },
  { id: 'view.closeTab', native: ['mod+w'], mac: 'mod+ctrl+w', tab: 'alt+w' },
  { id: 'view.reopenTab', native: ['mod+shift+t'], mac: 'mod+ctrl+shift+t', tab: 'alt+shift+t' },
  { id: 'view.closeTerminal', native: ['ctrl+shift+w'], mac: 'ctrl+shift+w', tab: 'alt+shift+w' }
]

const setPlatform = (value: string) => Object.defineProperty(navigator, 'platform', { configurable: true, value })

function stubDisplayMode(mode: string) {
  vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === `(display-mode: ${mode})`, media: query }))
}

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

afterEach(() => {
  delete window.document.documentElement.dataset.hermesDesktopHost
  setPlatform('')
  vi.unstubAllGlobals()
  localStorage.clear()
})

it.each(['standalone', 'minimal-ui', 'window-controls-overlay'])(
  'a %s app window off macOS ships the Ctrl chords, and a real Ctrl keydown dispatches them',
  async mode => {
    setPlatform('Win32')
    stubDisplayMode(mode)
    const { comboFromEvent, formatCombo } = await import('./combo')
    const { $comboIndex, bindingsFor } = await import('@/store/keybinds')

    for (const shortcut of shortcuts) {
      expect(bindingsFor(shortcut.id)).toEqual(shortcut.native)
      expect(formatCombo(shortcut.native[0]).startsWith('Ctrl+')).toBe(true)
    }

    const ctrlN = new KeyboardEvent('keydown', { code: 'KeyN', ctrlKey: true, key: 'n' })
    expect($comboIndex.get().get(comboFromEvent(ctrlN)!)).toEqual(['session.new'])
    expect($comboIndex.get().has('alt+n')).toBe(false)
  }
)

it.each([
  ['a macOS app window', 'MacIntel', 'standalone'],
  ['a Windows tab', 'Win32', 'browser']
])('%s keeps the browser-tab defaults', async (_, platform, mode) => {
  setPlatform(platform)
  stubDisplayMode(mode)
  const { bindingsFor } = await import('@/store/keybinds')

  for (const shortcut of shortcuts) {
    expect(bindingsFor(shortcut.id)).toEqual([platform === 'MacIntel' ? shortcut.mac : shortcut.tab])
  }
})
