import assert from 'node:assert/strict'

import { test } from 'vitest'

import { parseDeepLink } from './deep-link-url'

const SCHEMES = ['hermes']

function parse(url: string, schemes: readonly string[] = SCHEMES) {
  const reasons: string[] = []

  return { payload: parseDeepLink(url, schemes, reason => reasons.push(reason)), reasons }
}

test('a malformed percent escape in the path is refused, never thrown, and the link is not logged', () => {
  for (const url of ['hermes://open/%E0%A4%A', 'hermes://blueprint/%', 'hermes://skill/install%zz?token=secret']) {
    const { payload, reasons } = parse(url)

    assert.equal(payload, null, url)
    assert.equal(reasons.length, 1, url)
    assert.ok(!reasons[0].includes('%') && !reasons[0].includes('secret'), url)
  }
})

test('the malformed-path rejection never echoes the host, path or query', () => {
  for (const url of [
    'hermes://private-token-label/%',
    'hermes://sk-live-abc123/%E0%A4%A?token=secret-value',
    'hermes://open/%zz?user=alice%40example.com'
  ]) {
    const { payload, reasons } = parse(url)
    const parsed = new URL(url)

    assert.equal(payload, null, url)
    assert.deepEqual(reasons, ['ignoring deep link with a malformed path'], url)
    assert.ok(!reasons[0].includes(parsed.hostname), url)

    for (const value of parsed.searchParams.values()) {
      assert.ok(!reasons[0].includes(value), url)
    }
  }
})

test('a payload re-dispatched the way main.ts queues it parses back to itself', () => {
  for (const url of [
    'hermes://blueprint/daily%20brief?slot=morning&tz=UTC',
    'hermes://open/a%2Fb%3Fc',
    'hermes://close-preview'
  ]) {
    const first = parse(url).payload

    assert.ok(first, url)
    const { kind, name, params } = first
    const query = Object.keys(params).length ? `?${new URLSearchParams(params).toString()}` : ''

    assert.deepEqual(parse(`hermes://${kind}/${encodeURIComponent(name)}${query}`).payload, first)
  }
})

test('only the accepted schemes parse', () => {
  assert.equal(parse('hermes-dev://open/x').payload, null)
  assert.ok(parse('hermes-dev://open/x', ['hermes-dev', 'hermes']).payload)
  assert.equal(parse('not a url').payload, null)
})
