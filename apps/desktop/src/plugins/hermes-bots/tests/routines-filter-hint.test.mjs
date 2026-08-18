import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// UX improvement: when a bot's cron store HAS jobs but none are namespaced
// `[bot:<name>]` for the active bot, the Routines pane now explains why it's
// empty instead of showing the generic empty state with no hint.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { listen: () => undefined } } }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__api = { routineFilterHint };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

test('hint: null when the active bot already has tagged jobs', () => {
  const all = [{ name: '[bot:ops] Morning', job_id: '1' }]
  assert.equal(load().__api.routineFilterHint(all, all), null)
})

test('hint: null when the store is genuinely empty', () => {
  assert.equal(load().__api.routineFilterHint([], []), null)
  assert.equal(load().__api.routineFilterHint(undefined, []), null)
})

test('hint: explains hidden jobs when store has jobs but none match the bot', () => {
  const all = [
    { name: '[bot:research] Digest', job_id: '2' },
    { name: 'untagged job', job_id: '3' }
  ]
  const hint = load().__api.routineFilterHint(all, [])
  assert.ok(hint)
  assert.match(hint, /tagged for this bot/)
})
