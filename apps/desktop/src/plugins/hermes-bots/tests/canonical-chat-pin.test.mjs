import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// #24's guarantee (preserve a valid canonical pin instead of clobbering it from
// a bounded session list) is now enforced inside openBotCanonicalChat(), which
// #28 introduced to unify the create / recover / replace lifecycle. This test
// pins that guarantee against the current structure: a valid pinned id present
// in the session list must be opened as-is (never replaced), and only an
// ACTUALLY-missing pin triggers recovery.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const start = source.indexOf('async function openBotCanonicalChat(')
assert.notEqual(start, -1, 'openBotCanonicalChat declaration is missing')
const end = source.indexOf('\nfunction ', start)
const fnSource = source.slice(start, end === -1 ? undefined : end)

test('regression: a pinned canonical chat found in the list is opened, not replaced', () => {
  // The pin is opened directly with its own id under the bot's profile.
  assert.match(fnSource, /host\.openSession\(id, \{ profile: name \}\)/)
  // Replacement (rows[0].id) only happens when the pin is NOT in the list...
  assert.match(fnSource, /if \(!rows\.some\(session => session\.id === id\)\)/)
  assert.match(fnSource, /id = rows\[0\]\.id/)
  // ...and an empty list clears the stale pin before recreating (the #28 fix),
  // rather than opening a known-bad id.
  assert.match(fnSource, /if \(!rows\.length\)/)
  assert.match(fnSource, /return createCanonicalChat\(name\)/)
})
