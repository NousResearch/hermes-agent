import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #93127: duplicate room delivery. Two raceable paths existed:
//   1. a member turn mid-flight when the room epoch bumps still commits its
//      reply + watermark when it returns (the stale loop only noticed
//      supersession at the NEXT member boundary), and
//   2. the stale loop and the fresh loop can both append the same reply.
// The fix re-checks the epoch after the turn returns (shouldCommitMemberTurn)
// and drops adjacent byte-identical member echoes (isDuplicateGroupAppend).

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadHelpers() {
  const start = pluginSource.indexOf('// --- room-turn decision helpers (#93127)')
  const end = pluginSource.indexOf('// --- end room-turn decision helpers ---', start)
  assert.notEqual(start, -1, 'plugin carries the room-turn helper block')
  assert.notEqual(end, -1, 'room-turn helper block has a stable end marker')
  const context = {}
  vm.runInNewContext(
    `${pluginSource.slice(start, end)}
globalThis.shouldCommitMemberTurn = shouldCommitMemberTurn
globalThis.isDuplicateGroupAppend = isDuplicateGroupAppend`,
    context
  )
  return context
}

const memberEntry = (name, text, thread = 't1', at = Date.now(), source) => ({
  id: 'x',
  at,
  from: { kind: 'member', name, ...(source ? { source } : {}) },
  text,
  thread
})

test('a superseded turn is discarded — epoch moved on while the turn ran', () => {
  const { shouldCommitMemberTurn } = loadHelpers()
  assert.equal(shouldCommitMemberTurn(3, 4), false)
  assert.equal(shouldCommitMemberTurn(3, 7), false)
  // explicit same-thread supersession
  assert.equal(shouldCommitMemberTurn(3, 4, true), false)
})

test('a current turn commits — epoch unchanged since dispatch', () => {
  const { shouldCommitMemberTurn } = loadHelpers()
  assert.equal(shouldCommitMemberTurn(3, 3), true)
  assert.equal(shouldCommitMemberTurn(0, 0), true)
})

test('a cross-thread epoch bump does NOT discard finished work (no fresh loop re-drives it)', () => {
  const { shouldCommitMemberTurn } = loadHelpers()
  // Epoch moved, but no newer user entry landed in THIS thread: the
  // superseding send lives in another thread whose loop filters this one
  // out — dropping the reply would lose completed work forever.
  assert.equal(shouldCommitMemberTurn(3, 4, false), true)
})

test('adjacent identical member reply is a duplicate append', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const last = memberEntry('impl', 'Stopped. Standing by.')
  const dup = isDuplicateGroupAppend(last, { kind: 'member', name: 'impl' }, 'Stopped. Standing by.', 't1')
  assert.equal(dup, true)
})

test('identical text from a DIFFERENT member is not a duplicate', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const last = memberEntry('impl', 'Confirmed.')
  assert.equal(isDuplicateGroupAppend(last, { kind: 'member', name: 'reviewer' }, 'Confirmed.', 't1'), false)
})

test('identical text from the same member on another SOURCE is not a duplicate', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const last = memberEntry('impl', 'Confirmed.', 't1', Date.now(), 'laptop')
  assert.equal(isDuplicateGroupAppend(last, { kind: 'member', name: 'impl' }, 'Confirmed.', 't1'), false)
})

test('same text after an intervening entry is not a duplicate — only the LAST entry is checked', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  // The helper only ever receives the immediately-preceding entry; an
  // intervening entry means lastEntry is the interloper, not the echo.
  const intervening = memberEntry('reviewer', 'ack')
  assert.equal(isDuplicateGroupAppend(intervening, { kind: 'member', name: 'impl' }, 'Confirmed.', 't1'), false)
})

test('identical text in a DIFFERENT thread is not a duplicate', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const last = memberEntry('impl', 'Confirmed.', 'thread-a')
  assert.equal(isDuplicateGroupAppend(last, { kind: 'member', name: 'impl' }, 'Confirmed.', 'thread-b'), false)
})

test('identical text outside the recency window is not a duplicate', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const hourAgo = Date.now() - 60 * 60 * 1000
  const last = memberEntry('impl', 'Done.', 't1', hourAgo)
  assert.equal(isDuplicateGroupAppend(last, { kind: 'member', name: 'impl' }, 'Done.', 't1'), false)
})

test('user entries are never deduped', () => {
  const { isDuplicateGroupAppend } = loadHelpers()
  const last = { id: 'x', at: Date.now(), from: { kind: 'user', name: 'You' }, text: 'stop', thread: 't1' }
  assert.equal(isDuplicateGroupAppend(last, { kind: 'user', name: 'You' }, 'stop', 't1'), false)
})
