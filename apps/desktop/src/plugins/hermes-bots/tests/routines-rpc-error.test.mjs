import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// loadRoutines -> requestForBot. A rejected JSON-RPC value that is not an
// Error (or an Error whose `name` is not a string) used to crash React 19's
// formatter: `(e.name || '').trim` is not a function. The Routines pane then
// died instead of showing "Could not load cronjobs" (#94471).

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(request, requestProfile) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const host = { request, state: { profile: { listen: () => undefined } } }
  if (requestProfile) {
    host.requestProfile = requestProfile
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__routines = { asRpcError, loadRoutines, requestForBot };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

function react19Format(error) {
  return `${(error.name || '').trim()}: ${error.message}`
}

test('regression: a non-Error cron.manage rejection becomes a string-named Error', async () => {
  const runtime = load(async () => {
    const failure = { name: 32000, message: 'cron.manage failed', code: -32603 }
    throw failure
  })

  await assert.rejects(
    () => runtime.__routines.loadRoutines('research'),
    error => {
      assert.equal(typeof error.name, 'string')
      assert.doesNotThrow(() => react19Format(error))
      assert.match(String(error.message), /cron.manage failed/)
      assert.equal(error.cause.name, 32000)
      return true
    }
  )
})

test('regression: Error with a numeric name is coerced before React 19 format', () => {
  const weird = new Error('down')
  Object.defineProperty(weird, 'name', { value: 13, configurable: true })
  const coerced = load(async () => {}).__routines.asRpcError(weird, 'fallback')
  assert.equal(typeof coerced.name, 'string')
  assert.doesNotThrow(() => react19Format(coerced))
  assert.match(String(coerced.message), /down/)
})

test('regression: a frozen non-string Error name is copied, not mutated in place', () => {
  const weird = new Error('down')
  Object.defineProperty(weird, 'name', { value: 13, configurable: false, writable: false })
  const coerced = load(async () => {}).__routines.asRpcError(weird, 'fallback')
  assert.notEqual(coerced, weird)
  assert.equal(typeof coerced.name, 'string')
  assert.doesNotThrow(() => react19Format(coerced))
  assert.match(coerced.message, /down/)
})

test('regression: a real Error passes through unchanged', () => {
  const original = new Error('gateway rejected the pause')
  const coerced = load(async () => {}).__routines.asRpcError(original, 'fallback')
  assert.equal(coerced, original)
  assert.equal(coerced.name, 'Error')
})
