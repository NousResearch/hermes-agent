import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { $switcherOpen, closeSwitcher } from '@/store/session-switcher'

import {
  composerFocusBlockedBySurface,
  composerFocusKeysAllowed,
  isActivateOnEnterTarget,
  typeToFocusChar
} from './composer-focus-keys'

function keydown(init: KeyboardEventInit & { target?: EventTarget }): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init })

  if (init.target) {
    Object.defineProperty(event, 'target', { value: init.target })
  }

  return event
}

describe('isActivateOnEnterTarget', () => {
  it('ignores body / null', () => {
    expect(isActivateOnEnterTarget(document.body)).toBe(false)
    expect(isActivateOnEnterTarget(null)).toBe(false)
  })

  it('detects buttons and walks to a wrapping activator', () => {
    const button = document.createElement('button')
    const wrap = document.createElement('div')
    wrap.setAttribute('role', 'button')
    const child = document.createElement('span')
    wrap.append(child)
    document.body.append(button, wrap)

    expect(isActivateOnEnterTarget(button)).toBe(true)
    expect(isActivateOnEnterTarget(child)).toBe(true)
    expect(isActivateOnEnterTarget(document.createElement('div'))).toBe(false)
  })
})

describe('composerFocusBlockedBySurface', () => {
  beforeEach(() => {
    $workspaceIsPage.set(false)
    closeSwitcher()
    document.body.replaceChildren()
  })

  afterEach(() => {
    $workspaceIsPage.set(false)
    closeSwitcher()
    document.body.replaceChildren()
  })

  it('is clear on empty chat chrome', () => {
    expect(composerFocusBlockedBySurface()).toBe(false)
  })

  it('blocks dialogs, the session switcher, and full pages', () => {
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.append(dialog)
    expect(composerFocusBlockedBySurface()).toBe(true)

    document.body.replaceChildren()
    $switcherOpen.set(true)
    expect(composerFocusBlockedBySurface()).toBe(true)

    closeSwitcher()
    $workspaceIsPage.set(true)
    expect(composerFocusBlockedBySurface()).toBe(true)
  })

  it('blocks while an overlay covers the chat (composer sits behind it)', () => {
    const overlay = document.createElement('div')
    overlay.setAttribute('data-overlay-surface', '')
    document.body.append(overlay)

    expect(composerFocusBlockedBySurface()).toBe(true)
  })

  it('ignores a clarify card — it yields only its own keys, per-key', () => {
    const card = document.createElement('div')
    card.setAttribute('data-clarify-choices', '2')
    document.body.append(card)

    expect(composerFocusBlockedBySurface()).toBe(false)
  })

  it('blocks when focus is inside a terminal', () => {
    const term = document.createElement('div')
    term.setAttribute('data-terminal', '')
    const inner = document.createElement('div')
    term.append(inner)
    document.body.append(term)
    Object.defineProperty(document, 'activeElement', { configurable: true, get: () => inner })

    expect(composerFocusBlockedBySurface()).toBe(true)

    Object.defineProperty(document, 'activeElement', {
      configurable: true,
      get: () => document.body
    })
  })
})

describe('typeToFocusChar', () => {
  it('returns printables (case/symbols via event.key)', () => {
    expect(typeToFocusChar(keydown({ key: 'a', code: 'KeyA' }))).toBe('a')
    expect(typeToFocusChar(keydown({ key: 'A', code: 'KeyA', shiftKey: true }))).toBe('A')
    expect(typeToFocusChar(keydown({ key: '?', code: 'Slash', shiftKey: true }))).toBe('?')
    expect(typeToFocusChar(keydown({ key: ' ', code: 'Space' }))).toBe(' ')
  })

  it('rejects non-printables and modified chords', () => {
    expect(typeToFocusChar(keydown({ key: 'Enter', code: 'Enter' }))).toBeNull()
    expect(typeToFocusChar(keydown({ key: 'a', code: 'KeyA', metaKey: true }))).toBeNull()
    expect(typeToFocusChar(keydown({ key: 'a', code: 'KeyA', isComposing: true }))).toBeNull()
  })
})

describe('composerFocusKeysAllowed', () => {
  beforeEach(() => {
    $workspaceIsPage.set(false)
    closeSwitcher()
    document.body.replaceChildren()
    vi.spyOn(document, 'activeElement', 'get').mockReturnValue(document.body)
  })

  afterEach(() => {
    $workspaceIsPage.set(false)
    closeSwitcher()
    document.body.replaceChildren()
    vi.restoreAllMocks()
  })

  it('passes rebound chords; allows soft keys on the transcript', () => {
    expect(composerFocusKeysAllowed(keydown({ key: 'i', code: 'KeyI', metaKey: true }), 'mod+i')).toBe(true)
    expect(composerFocusKeysAllowed(keydown({ key: '/', code: 'Slash', target: document.body }), '/')).toBe(true)
    expect(composerFocusKeysAllowed(keydown({ key: 'Enter', code: 'Enter', target: document.body }), 'enter')).toBe(
      true
    )
    expect(composerFocusKeysAllowed(keydown({ key: 'h', code: 'KeyH', target: document.body }), 'type')).toBe(true)
  })

  it('refuses editables; refuses Enter on buttons but allows / and typing', () => {
    const input = document.createElement('input')
    const button = document.createElement('button')
    document.body.append(input, button)

    expect(composerFocusKeysAllowed(keydown({ key: 'a', code: 'KeyA', target: input }), 'type')).toBe(false)
    expect(composerFocusKeysAllowed(keydown({ key: 'Enter', code: 'Enter', target: button }), 'enter')).toBe(false)
    expect(composerFocusKeysAllowed(keydown({ key: '/', code: 'Slash', target: button }), '/')).toBe(true)
    expect(composerFocusKeysAllowed(keydown({ key: 'a', code: 'KeyA', target: button }), 'type')).toBe(true)
  })

  it('refuses when a dialog is open', () => {
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.append(dialog)

    expect(composerFocusKeysAllowed(keydown({ key: 'a', code: 'KeyA', target: document.body }), 'type')).toBe(false)
  })

  it('yields only the keys a live clarify card actually binds', () => {
    const card = document.createElement('div')
    // Two choices → rows A/B plus the "Other" row C; 1/2/3 are the digit twins.
    card.setAttribute('data-clarify-choices', '2')
    document.body.append(card)

    const allowed = (key: string, combo: string) =>
      composerFocusKeysAllowed(keydown({ key, target: document.body }), combo)

    // The card's own shortcuts win over type-to-focus.
    expect(allowed('a', 'type')).toBe(false)
    expect(allowed('B', 'type')).toBe(false)
    expect(allowed('c', 'type')).toBe(false)
    expect(allowed('1', 'type')).toBe(false)
    expect(allowed('3', 'type')).toBe(false)
    expect(allowed('Enter', 'enter')).toBe(false)

    // Everything past the last row is not the card's — typing a real message
    // instead of picking an option must still reach the composer.
    expect(allowed('d', 'type')).toBe(true)
    expect(allowed('z', 'type')).toBe(true)
    expect(allowed('4', 'type')).toBe(true)
    expect(allowed('?', 'type')).toBe(true)
    expect(allowed(' ', 'type')).toBe(true)
  })

  it('leaves every key alone for a clarify card in a background tab', () => {
    const tab = document.createElement('div')
    tab.setAttribute('data-pane-hidden', '')
    const card = document.createElement('div')
    card.setAttribute('data-clarify-choices', '2')
    tab.append(card)
    document.body.append(tab)

    expect(composerFocusKeysAllowed(keydown({ key: 'a', target: document.body }), 'type')).toBe(true)
    expect(composerFocusKeysAllowed(keydown({ key: 'Enter', target: document.body }), 'enter')).toBe(true)
  })
})
