import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// Proactive reclaim refresh (polish on top of #92928/#92901): when the
// gateway reaps the runtime behind the OPEN bot chat (session.reclaimed —
// idle TTL, LRU, or the mass WS-orphan reap that killed every background
// bot's handle at once in the Aug 23 incident), the plugin re-resumes the
// canonical chat immediately instead of letting the user's next send eat a
// stale-id error + recovery round-trip.
//
// Source-shape contracts:
// - the listener subscribes via host.onEvent('session.reclaimed'), feature-
//   detected, and is disposed with the other listeners;
// - the match is on the STORED id against BOTH claim identities;
// - a stale generation or missing claim/bot is a no-op;
// - a failed re-resume is swallowed (next-send recovery ladder remains).

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function reclaimBlock() {
  const start = pluginSource.indexOf("host.onEvent('session.reclaimed'")
  assert.ok(start > 0, 'reclaim listener exists')
  return pluginSource.slice(start - 400, start + 1800)
}

test('listener is feature-detected and disposed', () => {
  const block = reclaimBlock()
  assert.match(block, /typeof host\.onEvent === 'function'/)
  assert.match(pluginSource, /stopReclaimSync\?\.\(\)/)
})

test('match is stored-id against both claim identities', () => {
  const block = reclaimBlock()
  assert.match(block, /payload\.stored_session_id/)
  assert.match(block, /\[claim\.openedSessionId, claim\.openedRegistryId\]\.filter\(Boolean\)/)
  assert.match(block, /owned\.includes\(stored\)/)
})

test('re-resume guards the open generation and refreshes both claim ids', () => {
  const block = reclaimBlock()
  assert.match(block, /generation !== botOpenGeneration/)
  assert.match(block, /openedRegistryId: opened\.registryId/)
  assert.match(block, /openedSessionId: opened\.openedId/)
})

test('failed re-resume is swallowed — next-send recovery stays the backstop', () => {
  const block = reclaimBlock()
  assert.match(block, /\.catch\(\(\) => \{/)
  assert.match(block, /next send recovers via the ladder/)
})
