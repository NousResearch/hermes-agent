import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #93129: a bot told to stop must STAY stopped. The hold helpers are pure and
// vm-sliced out of plugin.js exactly like group-turn-races.test.mjs does for
// the #93127 helpers. NOTE: vm-realm arrays/objects fail strict deepEqual
// against host literals (different realm prototypes) — normalize with spread
// or JSON round-trip before comparing.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadHelpers() {
  const start = pluginSource.indexOf('// --- member-hold helpers (#93129)')
  const end = pluginSource.indexOf('// --- end member-hold helpers ---', start)
  assert.notEqual(start, -1, 'plugin carries the member-hold helper block')
  assert.notEqual(end, -1, 'member-hold helper block has a stable end marker')
  const context = {}
  vm.runInNewContext(
    `${pluginSource.slice(start, end)}
globalThis.classifyGroupHoldDirective = classifyGroupHoldDirective
globalThis.applyGroupHoldDirective = applyGroupHoldDirective
globalThis.heldMemberWatermarkAdvance = heldMemberWatermarkAdvance`,
    context
  )
  return context
}

// ── stop detection ───────────────────────────────────────────────────────────

test('explicit stop with a mention holds the mentioned member', () => {
  const { classifyGroupHoldDirective } = loadHelpers()
  for (const text of ['stop @impl', '@impl stop', '@impl please halt', 'pause @impl for now']) {
    const action = classifyGroupHoldDirective(text, ['impl'], false)
    assert.deepEqual([...action.hold], ['impl'], `"${text}" should hold`)
    assert.deepEqual([...action.release], [])
  }
})

test('stop word without any mention holds nobody', () => {
  const { classifyGroupHoldDirective } = loadHelpers()
  const action = classifyGroupHoldDirective('stop', [], false)
  assert.deepEqual([...action.hold], [])
})

test('conservative choice: "don\'t stop @x" still holds (documented trade-off)', () => {
  const { classifyGroupHoldDirective } = loadHelpers()
  const action = classifyGroupHoldDirective("don't stop @impl", ['impl'], false)
  assert.deepEqual([...action.hold], ['impl'])
})

test('"stopped" as part of another word does not trigger a hold', () => {
  const { classifyGroupHoldDirective } = loadHelpers()
  // \b(stop|halt|pause)\b — "stopped" is a different token
  const action = classifyGroupHoldDirective('@impl unstoppable work ahead', ['impl'], false)
  assert.deepEqual([...action.hold], [])
  // a plain non-stop mention releases instead (direct address overrides hold)
  assert.deepEqual([...action.release], ['impl'])
})

// ── hold lifecycle ───────────────────────────────────────────────────────────

test('stop sets a hold; resume for the same member clears it', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const stamp = { at: 1000, byMessageId: 'm1', thread: 't1' }
  const held = applyGroupHoldDirective({}, { mentioned: ['impl'], everyone: false }, 'stop @impl', stamp)
  assert.ok(held.impl)
  assert.equal(held.impl.at, 1000)
  const released = applyGroupHoldDirective(held, { mentioned: ['impl'], everyone: false }, '@impl resume', stamp)
  assert.equal(released.impl, undefined)
})

test('a direct non-stop mention of a held member releases the hold', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const held = { impl: { at: 1, byMessageId: null, thread: null } }
  const next = applyGroupHoldDirective(held, { mentioned: ['impl'], everyone: false }, '@impl what is your status?', {})
  assert.equal(next.impl, undefined)
})

test('@all resume releases every hold', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const held = { impl: { at: 1 }, docs: { at: 2 } }
  const next = applyGroupHoldDirective(held, { mentioned: [], everyone: true }, '@all resume', {})
  assert.deepEqual(JSON.parse(JSON.stringify(next)), {})
})

test('@all stop holds every member — symmetric with @all resume', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const next = applyGroupHoldDirective(
    {},
    { mentioned: [], everyone: true },
    '@all stop',
    { at: 5 },
    ['impl', 'docs']
  )
  assert.ok(next.impl)
  assert.ok(next.docs)
  assert.equal(next.impl.at, 5)
})

test('an unrelated room message leaves holds untouched (same object back)', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const held = { impl: { at: 1 } }
  const next = applyGroupHoldDirective(held, { mentioned: [], everyone: false }, 'receipt round complete', {})
  assert.equal(next, held)
})

test('holding one member does not disturb another\'s hold', () => {
  const { applyGroupHoldDirective } = loadHelpers()
  const held = { impl: { at: 1 } }
  const next = applyGroupHoldDirective(held, { mentioned: ['docs'], everyone: false }, 'stop @docs', { at: 2 })
  assert.ok(next.impl)
  assert.ok(next.docs)
})

// ── skip must not spin ───────────────────────────────────────────────────────

test('held skip consumes the delta exactly once', () => {
  const { heldMemberWatermarkAdvance } = loadHelpers()
  // fresh delta → advance to log length
  assert.equal(heldMemberWatermarkAdvance(3, 7), 7)
  // already consumed → no write, no spin
  assert.equal(heldMemberWatermarkAdvance(7, 7), null)
  assert.equal(heldMemberWatermarkAdvance(9, 7), null)
  // unset watermark treated as 0
  assert.equal(heldMemberWatermarkAdvance(undefined, 2), 2)
})
