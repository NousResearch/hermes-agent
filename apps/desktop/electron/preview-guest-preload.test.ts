import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { GUEST_EXTERNAL_CHANNEL, installGuestExternalHandoff } from './preview-guest-preload'

function rig() {
  const sent: { channel: string; args: unknown[] }[] = []
  const listeners: { type: string; listener: (event: unknown) => void; capture?: boolean }[] = []

  const host = {
    addEventListener: (type: 'click', listener: (event: unknown) => void, capture?: boolean) =>
      listeners.push({ type, listener, capture }),
    sendToHost: (channel: string, ...args: unknown[]) => sent.push({ args, channel })
  }

  installGuestExternalHandoff(host)

  return { click: (target: unknown) => listeners[0].listener({ target }), listeners, sent }
}

describe('installGuestExternalHandoff', () => {
  test('registers one capture-phase click listener', () => {
    const { listeners } = rig()

    assert.equal(listeners.length, 1)
    assert.equal(listeners[0].type, 'click')
    assert.equal(listeners[0].capture, true)
  })

  test('forwards a clicked _blank anchor as the channel message', () => {
    const { click, sent } = rig()

    click({
      closest: (selector: string) =>
        selector === 'a[target="_blank"]' ? { href: 'https://www.google.com/search?q=traceback' } : null
    })

    assert.deepEqual(sent, [{ args: ['https://www.google.com/search?q=traceback'], channel: GUEST_EXTERNAL_CHANNEL }])
  })

  test('climbs from an inner element through the shared DOM', () => {
    const { click, sent } = rig()

    // The listener resolves the enclosing anchor itself via `closest`.
    click({
      closest: (selector: string) => (selector === 'a[target="_blank"]' ? { href: 'https://chatgpt.com/?q=why' } : null)
    })

    assert.equal(sent.length, 1)
    assert.equal(sent[0].args[0], 'https://chatgpt.com/?q=why')
  })

  test('ignores clicks that resolve to no _blank anchor', () => {
    const { click, sent } = rig()

    click({ closest: () => null })
    click({ closest: (selector: string) => (selector === 'a[target="_blank"]' ? { href: '' } : null) })
    click(null)
    click({})

    assert.deepEqual(sent, [])
  })
})
