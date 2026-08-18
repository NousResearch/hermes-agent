import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The /new -> /compact guard protects a bot's canonical forever-chat from being
// forked by /new. It compares the current session id against the bot's stored
// canonical id. That id is persisted as meta.chat everywhere (createCanonicalChat
// saveBotMeta(name,{chat:sid}), openBotCanonicalChat, BotRow). A regression read
// it as meta.chat_pin — a key that is never written — so pinnedId was always null
// and the guard never fired: /new silently forked the forever-chat.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Locate the /new reroute guard block.
const guardStart = source.indexOf('const slashNew =')
assert.notEqual(guardStart, -1, '/new guard block is missing')
const guardBlock = source.slice(guardStart, guardStart + 600)

test('regression: /new guard reads the canonical id from meta.chat, not meta.chat_pin', () => {
  assert.match(guardBlock, /const pinnedId = meta\?\.chat \|\| null/)
  assert.doesNotMatch(guardBlock, /chat_pin/)
})

test('regression: canonical id is persisted as meta.chat (the key the guard reads)', () => {
  // The writer and the guard must agree on the key, or the guard never fires.
  assert.match(source, /saveBotMeta\([^)]*\{\s*chat:\s*sid\s*\}/)
})
