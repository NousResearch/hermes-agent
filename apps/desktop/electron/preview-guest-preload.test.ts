import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { GUEST_EXTERNAL_CHANNEL, type GuestClickEvent, installGuestExternalHandoff } from './preview-guest-preload'

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
