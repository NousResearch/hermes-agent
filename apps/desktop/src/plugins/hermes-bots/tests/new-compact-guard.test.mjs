import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The /new -> /compact guard protects a bot's canonical forever-chat from being
// forked by /new. Canonical identity is the NAME — the profile's session titled
// "Bot Chat" — reported by the gateway as canonical_session on every roster row.
// The guard compares the on-screen session id against that registry row (durable
// id OR compression-lineage tip). No stored meta.chat pointer is consulted:
// pointers dangle; the registry row cannot.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Locate the /new reroute guard block.
const guardStart = source.indexOf('const slashNew =')
assert.notEqual(guardStart, -1, '/new guard block is missing')
const guardBlock = source.slice(guardStart, guardStart + 900)

test('the /new guard reads the canonical registry row, never a stored pointer', () => {
  assert.match(guardBlock, /canonical_session/)
  assert.doesNotMatch(guardBlock, /meta\?\.chat/)
  assert.doesNotMatch(guardBlock, /chat_pin/)
})

test('the guard matches both the durable registry id and the lineage tip', () => {
  assert.match(guardBlock, /canonical\?\.id/)
  assert.match(guardBlock, /canonical\?\.resolved_id/)
})
