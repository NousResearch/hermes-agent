import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Needs-attention badge (#93091 item 3): background bot failures whose class
// is attention-worthy (auth, quota, missing config, blocked) badge the roster
// tile; transient failures (rate limit, server error, timeout) never do; the
// bot's next good turn clears the badge. Display-only state — hooks live at
// the relay deliver reply and the group member turn boundary, and hidden bots
// keep their entry (hiding is roster-display-only).

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Evaluate just the delimited needs-attention helper section in a bare vm
 *  context (same slice-and-eval approach as the other pure-helper tests). */
function loadHelpers() {
  const start = pluginSource.indexOf('const BOT_ATTENTION_CLASSES')
  const end = pluginSource.indexOf('/** Last good cron list')
  assert.ok(start > -1 && end > start, 'needs-attention helper section exists')

  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = { atom }
  vm.createContext(context)

  return vm.runInContext(
    `${pluginSource.slice(start, end)}
;({ BOT_ATTENTION_CLASSES, BOT_ATTENTION_HINTS, attentionReasonFromError, $botAttention, noteBotAttention, clearBotAttention })`,
    context
  )
}

test('classifier: reason codes pass through; matches the #93091 item-1 enum', () => {
  const { BOT_ATTENTION_CLASSES, attentionReasonFromError } = loadHelpers()

  assert.deepEqual(
    [...BOT_ATTENTION_CLASSES].sort(),
    ['agent_blocked', 'missing_config', 'provider_auth_or_access', 'provider_quota_limit']
  )

  for (const code of BOT_ATTENTION_CLASSES) {
    assert.equal(attentionReasonFromError(code), code)
  }
})

test('classifier: raw-text fallback maps current gateway error strings', () => {
  const { attentionReasonFromError } = loadHelpers()

  // The anthropic 401 shape observed on current main.
  assert.equal(
    attentionReasonFromError(
      'Error code: 401 - {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'
    ),
    'provider_auth_or_access'
  )
  assert.equal(
    attentionReasonFromError('No LLM provider configured. Run hermes model to pick one.'),
    'missing_config'
  )
  assert.equal(attentionReasonFromError('No access token found for profile'), 'missing_config')
  assert.equal(attentionReasonFromError('Your account is out of funds'), 'provider_quota_limit')
  assert.equal(attentionReasonFromError('quota exceeded for this billing period'), 'provider_quota_limit')
  assert.equal(attentionReasonFromError('agent is blocked awaiting approval'), 'agent_blocked')
})

test('classifier: transient classes never badge', () => {
  const { attentionReasonFromError } = loadHelpers()

  for (const text of [
    'Rate limit exceeded, retry shortly',
    'Error code: 429 - too many requests',
    '500 Internal Server Error',
    'upstream 503 service unavailable',
    'the model is overloaded, try again',
    'request timed out after 180s',
    'temporarily unavailable',
    '',
    null,
    undefined
  ]) {
    assert.equal(attentionReasonFromError(text), null, `expected null for ${JSON.stringify(text)}`)
  }
})

test('lifecycle: set on classified failure, latest wins, cleared on success', () => {
  const { $botAttention, noteBotAttention, clearBotAttention } = loadHelpers()

  // Transient failure sets nothing.
  noteBotAttention('radar', 'Rate limit exceeded')
  assert.deepEqual(Object.keys($botAttention.get()), [])

  // Classified failure badges the bot with reason + timestamp + snippet.
  noteBotAttention('radar', 'Error code: 401 authentication_error')
  const first = $botAttention.get().radar
  assert.equal(first.reason, 'provider_auth_or_access')
  assert.ok(first.at > 0)
  assert.match(first.message, /401/)

  // A later failure for the same bot merges — latest wins.
  noteBotAttention('radar', 'No LLM provider configured')
  assert.equal($botAttention.get().radar.reason, 'missing_config')

  // Other bots are independent.
  noteBotAttention('dixie', 'quota exceeded')
  assert.equal($botAttention.get().dixie.reason, 'provider_quota_limit')

  // The next good turn clears exactly that bot.
  clearBotAttention('radar')
  assert.equal($botAttention.get().radar, undefined)
  assert.equal($botAttention.get().dixie.reason, 'provider_quota_limit')

  // Clearing an unbadged bot is a no-op.
  clearBotAttention('radar')
  clearBotAttention('')
  assert.equal($botAttention.get().dixie.reason, 'provider_quota_limit')
})

test('hooks: relay delivery and group member turns note/clear attention', () => {
  // Relay drain: deliver success clears, deliver failure notes.
  const drain = pluginSource.slice(
    pluginSource.indexOf('async function drainRelayOutboxes'),
    pluginSource.indexOf('function startBotRelay')
  )
  assert.match(drain, /clearBotAttention\(attentionKey\)/)
  // #93091: the drain prefers the typed reason from bot_relay.deliver's
  // error.data over free-text re-parsing, and forwards it to the reply.
  assert.match(drain, /noteBotAttention\(attentionKey, reason \|\| error\?\.message \|\| error\)/)
  assert.match(drain, /\.\.\.\(reason \? \{ reason \} : \{\}\)/)

  // Group member turn boundary: failure notes under the member key; a real
  // reply clears it. Typed gateway reasons take precedence over text parsing.
  assert.match(
    pluginSource,
    /noteBotAttention\(groupMemberKey\(member\), reason \|\| error\?\.message \|\| error\)/
  )
  assert.match(pluginSource, /clearBotAttention\(groupMemberKey\(member\)\)/)
})

test('render: roster row shows an amber warning glyph with a per-class hint', () => {
  assert.match(pluginSource, /BOT_ATTENTION_HINTS\[attention\.reason\]/)
  assert.match(pluginSource, /name: 'warning'/)
  // Lookup covers all three key shapes so relay- and group-recorded state
  // renders, including local/unannotated rows keyed by the ACTIVE connection.
  assert.match(pluginSource, /attentionByKey\[botSelectionKey\(bot\)\] \|\|/)
  assert.match(pluginSource, /attentionByKey\[botRosterKey\(bot\)\] \|\|/)
  assert.match(pluginSource, /attentionByKey\[`\$\{bot\?\.connectionId \|\| activeConnectionId\}::\$\{bot\?\.name \|\| 'default'\}`\]/)
  // Hint copy for every class.
  assert.match(pluginSource, /Sign in again for this profile/)
  assert.match(pluginSource, /Quota or balance exhausted/)
  assert.match(pluginSource, /Provider not configured — run hermes model/)
  assert.match(pluginSource, /Bot is blocked — see its last message/)
})
