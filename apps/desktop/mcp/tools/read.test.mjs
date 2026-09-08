import test from 'node:test'
import assert from 'node:assert/strict'
import { createReadTools } from './read.mjs'

// Capture the expression evalBounded would run, return a fake result.
function fakeConnect(result) {
  const calls = []
  const connect = async () => ({
    eval: async (expression) => {
      calls.push(expression)
      return typeof result === 'function' ? result(expression) : result
    }
  })
  return { connect, calls }
}

const deps = (result) => {
  const { connect, calls } = fakeConnect(result)
  const t = createReadTools({ connect, MAX_TEXT: 80, MAX_NODES: 20, MAX_EVAL: 4000, port: 9222, allowAct: false })
  return { ...t, calls }
}

test('BUG #6 (RED): query with limit:0 falls back to the cap instead of returning 0 nodes', async () => {
  const { query, calls } = deps(() => '[]')
  await query({ selector: 'div', limit: 0 })
  assert.ok(calls[0].includes('20'), 'expression must slice to MAX_NODES (20), got: ' + calls[0].match(/slice\(0, (\d+)\)/)?.[1])
})

test('BUG #4 (RED): query whitespace regex must be a single-escaped \\s+ in the sent expression', async () => {
  const { query, calls } = deps(() => '[]')
  await query({ selector: 'div' })
  assert.ok(calls[0].includes('/\\s+/g'), 'expression must contain /\\s+/g exactly')
  assert.ok(!calls[0].includes('\\\\s+'), 'expression must NOT contain a double backslash')
})

test('BUG #9 (RED): empty/whitespace selector throws a friendly error before eval', async () => {
  const { inspect, query } = deps(() => '[]')
  await assert.rejects(() => inspect({ selector: '' }), /selector required/i)
  await assert.rejects(() => query({ selector: '   ' }), /selector required/i)
})

test('#10: evalBounded truncation reports the omitted length', async () => {
  const long = 'x'.repeat(5000)
  const { evalBounded } = deps(long)
  const out = await evalBounded('1')
  assert.ok(out.startsWith('x'.repeat(4000).slice(0, 10)), 'starts with the kept prefix')
  assert.ok(/\[\+1000 chars truncated\]/.test(out), 'must report how many chars were cut, got: ' + out.slice(-40))
})

test('A2 (RED): evalBounded returns "null" when the renderer eval is undefined', async () => {
  const { evalBounded } = deps(undefined)
  assert.equal(await evalBounded('void 0'), 'null')
})

// --- P2: status applies the same --match filter as the CDP client ---

// deps with an injectable fetch returning a fixed /json/list.
const depsWithFetch = (list, match) => {
  const { connect } = fakeConnect('[]')
  const t = createReadTools({
    connect, MAX_TEXT: 80, MAX_NODES: 20, MAX_EVAL: 4000, MAX_CONSOLE: 50,
    port: 9222, allowAct: false, match, SELECTORS: {},
    fetchImpl: async () => ({ json: async () => list })
  })
  return t
}

const PAGES = [
  { type: 'page', url: 'http://127.0.0.1:5174/#/', title: 'Hermes dev', webSocketDebuggerUrl: 'ws://a' },
  { type: 'page', url: 'http://localhost:3000/other', title: 'Other page', webSocketDebuggerUrl: 'ws://b' }
]

test('P2 (RED): status applies the match filter — only matching targets count', async () => {
  const t = depsWithFetch(PAGES, '5174')
  const s = await t.status()
  assert.equal(s.cdpAlive, true)
  assert.equal(s.targets.length, 1, 'non-matching target must be filtered out')
  assert.match(s.targets[0].url, /5174/)
})

test('P2 (RED): zero matching targets → cdpAlive:false even though pages exist', async () => {
  const t = depsWithFetch(PAGES, '9999')
  const s = await t.status()
  assert.equal(s.cdpAlive, false, 'unreachable-matching renderer must not be reported alive')
  assert.equal(s.mode, 'unavailable')
})

test('P2: status payload exposes the match value for self-diagnosis', async () => {
  const t = depsWithFetch(PAGES, '5174')
  const s = await t.status()
  assert.equal(s.match, '5174')
})
