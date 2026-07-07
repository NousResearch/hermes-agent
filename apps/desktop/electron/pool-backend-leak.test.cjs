'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

function readElectronFile(name) {
  return fs.readFileSync(path.join(__dirname, name), 'utf8').replace(/\r\n/g, '\n')
}

// ---------------------------------------------------------------------------
// Pool backend leak guards (2026-07-05 orphaned "--profile ezra serve"
// incident): every path that drops a profile backend from backendPool must
// either verify ownership or kill the child. A delete-without-kill leaks the
// process — the Map is the only registry the idle reaper and the shutdown
// kill-all ever consult.
// ---------------------------------------------------------------------------

test('spawnPoolBackend exit/error handlers only delete their own pool entry', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function spawnPoolBackend(')
  assert.notEqual(fnStart, -1, 'spawnPoolBackend function not found')
  const fnBody = source.slice(fnStart, source.indexOf('\nfunction stopPoolBackend('))

  // No unconditional delete may remain in the child event handlers.
  assert.ok(
    !/^\s*backendPool\.delete\(profile\)/m.test(fnBody),
    'spawnPoolBackend still deletes the pool entry unconditionally'
  )
  const guardedDeletes = (
    fnBody.match(/if \(backendPool\.get\(profile\) === entry\) backendPool\.delete\(profile\)/g) || []
  ).length
  assert.ok(
    guardedDeletes >= 2,
    `expected ownership-guarded deletes in both exit and error handlers, found ${guardedDeletes}`
  )
})

test('ensureBackend kills the child when the spawn promise rejects', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function ensureBackend(')
  assert.notEqual(fnStart, -1, 'ensureBackend function not found')
  const fnBody = source.slice(fnStart, fnStart + 1600)

  const catchStart = fnBody.indexOf('.catch(error => {')
  assert.notEqual(catchStart, -1, 'spawn failure catch not found in ensureBackend')
  const catchBody = fnBody.slice(catchStart, fnBody.indexOf('throw error', catchStart))
  assert.ok(
    catchBody.includes('stopBackendChild(entry.process)'),
    'spawn failure path must stop the already-spawned child'
  )
  assert.ok(
    catchBody.includes('waitForBackendExit(entry.process)'),
    'spawn failure path must escalate to SIGKILL via waitForBackendExit'
  )
})

test('stopPoolBackend escalates beyond SIGTERM for de-registered children', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('function stopPoolBackend(')
  assert.notEqual(fnStart, -1, 'stopPoolBackend function not found')
  const fnBody = source.slice(fnStart, fnStart + 700)
  assert.ok(
    fnBody.includes('waitForBackendExit(entry.process)'),
    'stopPoolBackend must schedule SIGKILL escalation after the Map delete'
  )
})

test('boot runs the stray pool backend sweep', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('function sweepStrayPoolBackends(')
  assert.notEqual(fnStart, -1, 'sweepStrayPoolBackends function not found')
  const fnBody = source.slice(fnStart, fnStart + 2400)

  // The sweep must stay narrowly scoped: orphans only (PPID 1), our exact
  // spawn shape, and never a gateway.
  assert.ok(fnBody.includes('ppid !== 1'), 'sweep must only target reparented (PPID 1) processes')
  assert.ok(
    fnBody.includes("'--host 127.0.0.1 --port 0'") || fnBody.includes('"--host 127.0.0.1 --port 0"'),
    'sweep must require the ephemeral-port spawn signature'
  )
  assert.ok(/gateway/.test(fnBody), 'sweep must explicitly skip gateway processes')

  const whenReady = source.slice(source.indexOf('app.whenReady().then('))
  assert.ok(
    whenReady.includes('sweepStrayPoolBackends()'),
    'app.whenReady must invoke the stray backend sweep'
  )
})
