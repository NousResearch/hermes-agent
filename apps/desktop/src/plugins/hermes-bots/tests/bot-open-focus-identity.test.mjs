import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// First-click home bounce (community report, Aug 2026): clicking a bot whose
// canonical Bot Chat had been COMPRESSED landed on the Bots home instead of
// the chat; only a second click got through. Root cause: openRosterBot
// claimed the center with the durable registry id, but the session-focus
// edge that the open itself fired reports the compression-lineage TIP —
// releaseStaleOpenBotChat compared tip !== registry id, called the claim
// stale, released it, and the home reasserted over the freshly opened chat.
// The second click "worked" only because the tip was already focused, so no
// new focus edge fired to sabotage it.
//
// Contract pinned here (source-shape):
// - openBotCanonicalChat returns BOTH identities (registryId + openedId);
// - the $openBotChat claim stores openedSessionId alongside openedRegistryId;
// - releaseStaleOpenBotChat keeps the claim when the focused id matches
//   EITHER identity, and still releases on a genuinely foreign session.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('canonical open returns both the registry row and the opened tip', () => {
  const fn = pluginSource.slice(
    pluginSource.indexOf('async function openBotCanonicalChat'),
    pluginSource.indexOf('async function prepareBotSource')
  )
  assert.match(fn, /registryId: String\(existing\.id\)/)
  assert.match(fn, /openedId: String\(openedId\)/)
})

test('the open claim carries the opened session id', () => {
  const fn = pluginSource.slice(
    pluginSource.indexOf('async function openRosterBot'),
    pluginSource.indexOf('function displayName')
  )
  assert.match(fn, /openedSessionId: opened\.openedId/)
})

test('focus on either owned identity keeps the claim; foreign focus releases', () => {
  const start = pluginSource.indexOf('function releaseStaleOpenBotChat')
  const body = pluginSource.slice(start, pluginSource.indexOf('\n}', start) + 2)
  // executable check: evaluate the function against a stub store
  let stored = null
  const $openBotChat = {
    get: () => stored,
    set: value => {
      stored = value
    }
  }
  const release = new Function('$openBotChat', `${body}; return releaseStaleOpenBotChat`)($openBotChat)

  // compressed chat: claim carries registry 'reg-1' and tip 'tip-9'
  stored = { key: 'k', openedRegistryId: 'reg-1', openedSessionId: 'tip-9' }
  release('tip-9') // the focus edge the open itself fires
  assert.ok(stored, 'tip focus must NOT release the claim (first-click bounce)')

  release('reg-1')
  assert.ok(stored, 'registry-id focus must not release either')

  release('other-session')
  assert.equal(stored, null, 'a genuinely foreign session releases the claim')

  // legacy draft claim (no ids): any focused session releases
  stored = { key: 'k', openedRegistryId: '' }
  release('any')
  assert.equal(stored, null)
})
