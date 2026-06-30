import { describe, expect, it } from 'vitest'

import { clearGuestNonce, GUEST_SENTINEL_PREFIX, issueGuestNonce, parseGuestSentinel } from './browser-guest-bus'

const line = (nonce: string, payload: unknown) => `${GUEST_SENTINEL_PREFIX}${nonce}:${JSON.stringify(payload)}`

describe('browser-guest-bus sentinel parser', () => {
  it('parses a valid nonce-bound picked sentinel', () => {
    const nonce = issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'picked', ref: '@e0' }))).toEqual({
      kind: 'picked',
      ref: '@e0',
      tabId: 'tab-1'
    })
  })

  it('carries the ok flag for style-probe acks', () => {
    const nonce = issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'style-probe', ok: true, ref: 'p1' }))).toEqual({
      kind: 'style-probe',
      ok: true,
      ref: 'p1',
      tabId: 'tab-1'
    })
  })

  it('rejects a forged sentinel with the wrong nonce (spoof containment)', () => {
    issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line('forged', { kind: 'picked', ref: 'x' }))).toBeNull()
  })

  it('rejects a sentinel for a tab with no issued nonce', () => {
    clearGuestNonce('tab-2')

    expect(parseGuestSentinel('tab-2', line('whatever', { kind: 'picked', ref: 'x' }))).toBeNull()
  })

  it('treats a normal console line as not-a-sentinel (never dropped)', () => {
    issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', 'Uncaught TypeError: boom')).toBeNull()
    expect(parseGuestSentinel('tab-1', `${GUEST_SENTINEL_PREFIX} no separator here`)).toBeNull()
  })

  it('rejects oversized payloads', () => {
    const nonce = issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'picked', ref: 'x'.repeat(20000) }))).toBeNull()
  })

  it('rejects malformed JSON', () => {
    const nonce = issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', `${GUEST_SENTINEL_PREFIX}${nonce}:{not json`)).toBeNull()
  })

  it('rejects unknown kinds and missing refs', () => {
    const nonce = issueGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'evil', ref: 'x' }))).toBeNull()
    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'picked' }))).toBeNull()
  })

  it('clearGuestNonce invalidates subsequent sentinels', () => {
    const nonce = issueGuestNonce('tab-1')

    clearGuestNonce('tab-1')

    expect(parseGuestSentinel('tab-1', line(nonce, { kind: 'picked', ref: 'x' }))).toBeNull()
  })
})
