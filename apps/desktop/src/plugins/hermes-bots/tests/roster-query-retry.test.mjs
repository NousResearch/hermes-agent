import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// The Bots pane renders a spinner while useRoster() is isLoading and has no
// snapshot. React Query treats `retry: true` as infinite retries, which keeps
// isLoading true forever — the sidebar stays empty with no error/retry card.
// Bounded retries plus the existing 5s refetch recover SSH/sleep drops.

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
    host: {
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        gateway: { get: () => 'open', listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined }
      },
      request: () => Promise.resolve({ profiles: [] })
    },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__roster = { ROSTER_QUERY_RETRY };
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__roster
}

test('roster query retries are bounded so a stalled profiles.list cannot pin the spinner', () => {
  const { ROSTER_QUERY_RETRY } = load()
  assert.equal(typeof ROSTER_QUERY_RETRY, 'number')
  assert.ok(ROSTER_QUERY_RETRY >= 0)
  assert.ok(ROSTER_QUERY_RETRY < Number.POSITIVE_INFINITY)
  assert.notEqual(ROSTER_QUERY_RETRY, true)
  assert.match(pluginSource, /retry:\s*ROSTER_QUERY_RETRY/)
})
