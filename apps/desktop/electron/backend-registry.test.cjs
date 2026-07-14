'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  parseRegistry,
  stringifyRegistry,
  upsertEntry,
  removePids,
  reapablePids,
  backendCommandMatches
} = require('./backend-registry.cjs')

test('parseRegistry reads the wrapped shape we write', () => {
  const text = stringifyRegistry([{ pid: 42, command: 'hermes serve', startedAt: 100 }])
  assert.deepEqual(parseRegistry(text), [{ pid: 42, command: 'hermes serve', startedAt: 100 }])
})

test('parseRegistry accepts a bare array for back-compat', () => {
  const text = JSON.stringify([{ pid: 7, command: 'hermes dashboard --no-open', startedAt: 5 }])
  assert.deepEqual(parseRegistry(text), [
    { pid: 7, command: 'hermes dashboard --no-open', startedAt: 5 }
  ])
})

test('parseRegistry returns [] on garbage / partial writes rather than throwing', () => {
  assert.deepEqual(parseRegistry('{"backends":[{"pid":1'), [])
  assert.deepEqual(parseRegistry(''), [])
  assert.deepEqual(parseRegistry(null), [])
  assert.deepEqual(parseRegistry('not json'), [])
})

test('parseRegistry drops entries without a valid positive-integer pid', () => {
  const text = JSON.stringify({
    backends: [
      { pid: 0 },
      { pid: -3 },
      { pid: 1.5 },
      { pid: 'x' },
      { command: 'no pid' },
      null,
      { pid: 9, command: 'ok' }
    ]
  })
  assert.deepEqual(parseRegistry(text), [{ pid: 9, command: 'ok', startedAt: 0 }])
})

test('parseRegistry de-duplicates by pid, keeping the first occurrence', () => {
  const text = JSON.stringify({
    backends: [
      { pid: 5, command: 'first' },
      { pid: 5, command: 'second' }
    ]
  })
  assert.deepEqual(parseRegistry(text), [{ pid: 5, command: 'first', startedAt: 0 }])
})

test('parseRegistry defaults missing command/startedAt', () => {
  assert.deepEqual(parseRegistry(JSON.stringify({ backends: [{ pid: 3 }] })), [
    { pid: 3, command: '', startedAt: 0 }
  ])
})

test('upsertEntry adds a new pid without mutating the input', () => {
  const before = [{ pid: 1, command: 'a', startedAt: 0 }]
  const after = upsertEntry(before, { pid: 2, command: 'b', startedAt: 9 })
  assert.deepEqual(before, [{ pid: 1, command: 'a', startedAt: 0 }])
  assert.deepEqual(after, [
    { pid: 1, command: 'a', startedAt: 0 },
    { pid: 2, command: 'b', startedAt: 9 }
  ])
})

test('upsertEntry replaces an existing pid in place of a duplicate', () => {
  const after = upsertEntry([{ pid: 1, command: 'old', startedAt: 1 }], {
    pid: 1,
    command: 'new',
    startedAt: 2
  })
  assert.deepEqual(after, [{ pid: 1, command: 'new', startedAt: 2 }])
})

test('upsertEntry ignores an invalid pid', () => {
  assert.deepEqual(upsertEntry([{ pid: 1, command: 'a', startedAt: 0 }], { pid: 0 }), [
    { pid: 1, command: 'a', startedAt: 0 }
  ])
})

test('removePids drops the listed pids only', () => {
  const entries = [
    { pid: 1, command: 'a', startedAt: 0 },
    { pid: 2, command: 'b', startedAt: 0 },
    { pid: 3, command: 'c', startedAt: 0 }
  ]
  assert.deepEqual(removePids(entries, [2]), [
    { pid: 1, command: 'a', startedAt: 0 },
    { pid: 3, command: 'c', startedAt: 0 }
  ])
})

test('reapablePids returns every recorded pid except our own', () => {
  const entries = [{ pid: 10 }, { pid: 20 }, { pid: 30 }]
  assert.deepEqual(reapablePids(entries, 20), [10, 30])
})

test('reapablePids de-duplicates and skips invalid pids', () => {
  const entries = [{ pid: 10 }, { pid: 10 }, { pid: 0 }, { pid: -1 }, { pid: 40 }]
  assert.deepEqual(reapablePids(entries, 999), [10, 40])
})

test('backendCommandMatches recognises hermes serve / dashboard forms', () => {
  assert.ok(backendCommandMatches('/Users/x/.hermes/venv/bin/hermes serve --host 127.0.0.1 --port 0'))
  assert.ok(backendCommandMatches('hermes dashboard --no-open --host 127.0.0.1 --port 9120'))
  assert.ok(backendCommandMatches('python -m hermes_cli.main serve --port 0'))
  assert.ok(backendCommandMatches('python3 /opt/hermes_cli/main.py dashboard'))
})

test('backendCommandMatches rejects unrelated processes (PID reuse guard)', () => {
  assert.ok(!backendCommandMatches('/Applications/Safari.app/Contents/MacOS/Safari'))
  assert.ok(!backendCommandMatches('python -m hermes_cli.main serverless_thing')) // "serve" only as a prefix
  assert.ok(!backendCommandMatches('hermes chat')) // hermes, but not a backend
  assert.ok(!backendCommandMatches('vim dashboard.py')) // dashboard, but not hermes
  assert.ok(!backendCommandMatches(''))
  assert.ok(!backendCommandMatches(null))
})
