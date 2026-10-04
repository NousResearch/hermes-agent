import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  GUEST_EXTERNAL_CHANNEL,
  type GuestClickEvent,
  installGuestExternalHandoff,
  installGuestInteractionHandoff
} from './preview-guest-preload'

// Preview-pane guest bridge (#112941): the preload forwards a user's click on a
// `_blank` anchor to the host and nothing else.
function rig() {
  const sent: { channel: string; args: unknown[] }[] = []
  const listeners: { type: string; listener: (event: GuestClickEvent) => void; capture?: boolean }[] = []

  installGuestExternalHandoff({
    addEventListener: (type, listener, capture) => listeners.push({ capture, listener, type }),
    sendToHost: (channel, ...args) => sent.push({ args, channel })
  })

  return { click: (event: GuestClickEvent) => listeners[0].listener(event), listeners, sent }
}

const blankAnchor = (href: string) => ({
  closest: (selector: string) => (selector === 'a[target="_blank"]' ? { href } : null)
})

describe('guest interaction handoff', () => {
  function interactionRig() {
    const sent: { channel: string; args: unknown[] }[] = []
    const listeners = new Map<string, (event: { isTrusted?: boolean; target?: unknown }) => void>()
    installGuestInteractionHandoff({
      addEventListener: (type, listener, capture) => {
        assert.equal(capture, true)
        listeners.set(type, listener)
      },
      sendToHost: (channel: string, ...args: unknown[]) => sent.push({ channel, args })
    })

    return {
      emit: (type: string, event: { isTrusted?: boolean; target?: unknown }) => listeners.get(type)?.(event),
      sent
    }
  }

  test('trusted body pointer and keyboard input notify without guest data or target assumptions', () => {
    const { emit, sent } = interactionRig()
    // A #text target has no closest(). Neither text nor keys belong in IPC.
    emit('pointerdown', { isTrusted: true, target: { nodeType: 3, textContent: 'private page text' } })
    emit('keydown', { isTrusted: true, target: null })
    assert.deepEqual(sent, [
      { channel: 'preview-guest-interaction', args: [] },
      { channel: 'preview-guest-interaction', args: [] }
    ])
  })

  test('synthetic input and focus or passive page events never express selection intent', () => {
    const { emit, sent } = interactionRig()

    for (const type of ['pointerdown', 'keydown']) {
      emit(type, { isTrusted: false })
      emit(type, {})
    }

    for (const type of ['focus', 'focusin', 'pointermove', 'load']) {
      emit(type, { isTrusted: true })
    }

    assert.deepEqual(sent, [])
  })
})

describe('installGuestExternalHandoff', () => {
  test('a trusted click inside a _blank anchor is forwarded once, in the capture phase', () => {
    const { click, listeners, sent } = rig()

    assert.deepEqual(
      listeners.map(entry => [entry.type, entry.capture]),
      [['click', true]]
    )

    click({ button: 0, isTrusted: true, target: blankAnchor('https://www.google.com/search?q=traceback') })

    assert.deepEqual(sent, [{ args: ['https://www.google.com/search?q=traceback'], channel: GUEST_EXTERNAL_CHANNEL }])
  })

  test('synthetic clicks, non-primary buttons and non-_blank targets are dropped', () => {
    const { click, sent } = rig()

    // Page script dispatching its own click event must not reach the OS browser.
    click({ button: 0, isTrusted: false, target: blankAnchor('https://evil.example/') })
    click({ button: 1, isTrusted: true, target: blankAnchor('https://evil.example/') })
    click({ button: 0, isTrusted: true, target: { closest: () => null } })
    click({ button: 0, isTrusted: true, target: blankAnchor('') })
    click({ button: 0, isTrusted: true, target: null })

    assert.deepEqual(sent, [])
  })
})
