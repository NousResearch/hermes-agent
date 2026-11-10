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

const CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
const FIREFOX_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:144.0) Gecko/20100101 Firefox/144.0'

const setUserAgent = (value: string) => Object.defineProperty(navigator, 'userAgent', { configurable: true, value })

// A display mode the test can flip live, the way "Open in Chrome" moves an
// app window into a tab: every query list fires `change`.
function stubDisplayMode(initial: string) {
  let mode = initial
  const listeners = new Set<() => void>()

  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query === `(display-mode: ${mode})`
    },
    media: query,
    addEventListener: (_: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_: string, listener: () => void) => listeners.delete(listener)
  }))

  return (next: string) => {
    mode = next
    listeners.forEach(listener => listener())
  }
}

beforeEach(() => {
  vi.resetModules()
  setUserAgent(CHROME_UA)
  localStorage.clear()
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

afterEach(() => {
  delete window.document.documentElement.dataset.hermesDesktopHost
  setPlatform('')
  Reflect.deleteProperty(navigator, 'userAgent')
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

it('follows a live display-mode change and leaves a user rebind alone', async () => {
  setPlatform('Win32')
  const moveTo = stubDisplayMode('standalone')
  const { $bindings, bindingsFor, setBinding } = await import('@/store/keybinds')
  const { keybindActionAllowedInEditableTarget } = await import('./actions')

  setBinding('view.closeTab', ['mod+shift+q'])
  moveTo('browser')

  expect(bindingsFor('session.new')).toEqual(['alt+n'])
  expect(bindingsFor('session.next')).toEqual(['alt+]'])
  expect(keybindActionAllowedInEditableTarget('session.new', 'alt+n')).toBe(true)
  expect(bindingsFor('view.closeTab')).toEqual(['mod+shift+q'])

  moveTo('standalone')

  expect(bindingsFor('session.new')).toEqual(['mod+n'])
  expect($bindings.get()['view.closeTab']).toEqual(['mod+shift+q'])
})

it('keeps a rebind that equals the other window kind\'s default across flips and a reload', async () => {
  setPlatform('Win32')
  const moveTo = stubDisplayMode('standalone')
  const { bindingsFor, setBinding } = await import('@/store/keybinds')

  setBinding('view.closeTab', ['alt+w'])
  moveTo('browser')
  moveTo('standalone')

  expect(bindingsFor('view.closeTab')).toEqual(['alt+w'])

  vi.resetModules()
  const reloaded = await import('@/store/keybinds')

  expect(reloaded.bindingsFor('view.closeTab')).toEqual(['alt+w'])
})

it('skips a default move onto a chord the user gave another action', async () => {
  setPlatform('Win32')
  const moveTo = stubDisplayMode('standalone')
  const { $comboIndex, bindingsFor, conflictsFor, setBinding } = await import('@/store/keybinds')

  setBinding('session.focusSearch', ['alt+n'])
  moveTo('browser')

  expect($comboIndex.get().get('alt+n')).toEqual(['session.focusSearch'])
  expect(bindingsFor('session.new')).toEqual(['mod+n'])
  expect(conflictsFor('session.focusSearch', 'alt+n')).toEqual([])
  expect(bindingsFor('session.next')).toEqual(['alt+]'])

  vi.resetModules()
  const reloaded = await import('@/store/keybinds')

  expect(reloaded.$comboIndex.get().get('alt+n')).toEqual(['session.focusSearch'])
})

it('a non-Chromium web-app window keeps the browser-tab defaults', async () => {
  setPlatform('Win32')
  setUserAgent(FIREFOX_UA)
  stubDisplayMode('standalone')
  const { bindingsFor } = await import('@/store/keybinds')

  for (const shortcut of shortcuts) {
    expect(bindingsFor(shortcut.id)).toEqual([shortcut.tab])
  }
})

it('a user Alt binding still fires from the composer inside an app window', async () => {
  setPlatform('Win32')
  const moveTo = stubDisplayMode('browser')
  const { setBinding } = await import('@/store/keybinds')
  const { actionAllowedInInput } = await import('./combo')

  setBinding('view.closeTab', ['alt+shift+q'])
  moveTo('standalone')

  expect(actionAllowedInInput('view.closeTab', 'alt+shift+q')).toBe(true)
})

it('reports an app-window Ctrl default as a conflict for the same chord spelled mod', async () => {
  setPlatform('Win32')
  stubDisplayMode('standalone')
  const { conflictsFor } = await import('@/store/keybinds')

  expect(conflictsFor('session.focusSearch', 'mod+tab')).toEqual(['session.next'])
})

it('factory defaults spelled ctrl and mod do not warn against each other', async () => {
  setPlatform('Win32')
  stubDisplayMode('standalone')
  const { conflictsFor, bindingsFor } = await import('@/store/keybinds')
  const { allKeybindActions } = await import('@/lib/keybinds/actions')

  const warned = allKeybindActions().filter(action =>
    bindingsFor(action.id).some(combo => conflictsFor(action.id, combo).length > 0)
  )

  expect(warned.map(action => action.id).filter(id => /^(profile\.switch|session\.slot)\./.test(id))).toEqual([])
})
