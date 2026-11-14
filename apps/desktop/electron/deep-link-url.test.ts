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
