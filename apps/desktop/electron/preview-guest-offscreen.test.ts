import { describe, expect, it, vi } from 'vitest'

import { commandFocusedGuest, notePreviewGuestHidden } from './preview-guest-offscreen'

let nextId = 1

function fakeContents(type: 'webview' | 'window', hostWebContents?: unknown) {
  const history = {
    canGoBack: vi.fn(() => true),
    canGoForward: vi.fn(() => true),
    goBack: vi.fn(),
    goForward: vi.fn()
  }

  return {
    id: nextId++,
    hostWebContents,
    navigationHistory: history,
    reload: vi.fn(),
    getType: () => type,
    isDestroyed: () => false,
    once: vi.fn()
  }
}

type Fake = ReturnType<typeof fakeContents>
const asContents = (fake: Fake) => fake as never

describe('focused preview guest gestures', () => {
  it('navigates the focused guest its host reports on screen', () => {
    const host = fakeContents('window')
    const guest = fakeContents('webview', host)

    notePreviewGuestHidden(asContents(host), asContents(guest), false)

    expect(commandFocusedGuest('back', asContents(guest))).toBe(true)
    expect(guest.navigationHistory.goBack).toHaveBeenCalledOnce()
  })

  // A hidden session's kept Browser is still Chromium's focused webContents
  // after the switch: a mouse back on the session now on screen must not walk
  // that page's history (or reload it) — it falls through to the renderer.
  it('leaves a focused guest its host reported hidden untouched', () => {
    const host = fakeContents('window')
    const guest = fakeContents('webview', host)

    notePreviewGuestHidden(asContents(host), asContents(guest), true)

    expect(commandFocusedGuest('back', asContents(guest))).toBe(false)
    expect(commandFocusedGuest('forward', asContents(guest))).toBe(false)
    expect(commandFocusedGuest('reload', asContents(guest))).toBe(false)
    expect(guest.navigationHistory.goBack).not.toHaveBeenCalled()
    expect(guest.navigationHistory.goForward).not.toHaveBeenCalled()
    expect(guest.reload).not.toHaveBeenCalled()

    // Shown again: the user is back in it.
    notePreviewGuestHidden(asContents(host), asContents(guest), false)
    expect(commandFocusedGuest('back', asContents(guest))).toBe(true)
    expect(guest.navigationHistory.goBack).toHaveBeenCalledOnce()
  })

  it('takes visibility only from the renderer that embeds the guest', () => {
    const host = fakeContents('window')
    const stranger = fakeContents('window')
    const guest = fakeContents('webview', host)

    notePreviewGuestHidden(asContents(stranger), asContents(guest), true)

    expect(commandFocusedGuest('back', asContents(guest))).toBe(true)
  })

  it('leaves app chrome focus to the renderer', () => {
    expect(commandFocusedGuest('back', asContents(fakeContents('window')))).toBe(false)
    expect(commandFocusedGuest('back', null)).toBe(false)
  })
})
