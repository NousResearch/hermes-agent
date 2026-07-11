import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const source = fs
  .readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), 'main.ts'), 'utf8')
  .replace(/\r\n/g, '\n')

test('pool backend teardown waits for bounded exit before dropping lifecycle state', () => {
  const start = source.indexOf('async function stopPoolBackend(profile)')
  assert.notEqual(start, -1)
  const snippet = source.slice(start, start + 1200)

  assert.match(snippet, /poolStops\.get\(profile\)/)
  assert.match(snippet, /stopBackendChild\(entry\.process\)/)
  assert.match(snippet, /await waitForBackendExit\(entry\.process\)/)
  assert.match(snippet, /backendPool\.delete\(profile\)/)
  assert.ok(
    snippet.indexOf('await waitForBackendExit(entry.process)') < snippet.indexOf('backendPool.delete(profile)'),
    'pool state must remain owned until the child exits or is escalated'
  )
})

test('repeated pool cleanup reuses one in-flight teardown', () => {
  const start = source.indexOf('async function stopPoolBackend(profile)')
  const snippet = source.slice(start, start + 1200)

  assert.match(snippet, /if \(stopping\) \{\s*return stopping\s*\}/)
  assert.match(snippet, /poolStops\.set\(profile, stop\)/)
  assert.match(snippet, /poolStops\.delete\(profile\)/)

  const ensure = source.slice(source.indexOf('async function ensureBackend(profile)'), source.indexOf('async function ensureBackend(profile)') + 700)
  assert.match(ensure, /const stopping = poolStops\.get\(key\)/)
  assert.match(ensure, /if \(stopping\) \{\s*await stopping\s*\}/)
})

test('desktop quit waits for primary and pool backend teardown exactly once', () => {
  const start = source.indexOf("app.on('before-quit'")
  assert.notEqual(start, -1)
  const snippet = source.slice(start, start + 1800)

  assert.match(snippet, /event\.preventDefault\(\)/)
  assert.match(snippet, /if \(backendQuitPromise\)/)
  assert.match(snippet, /waitForBackendExit\(dying\)/)
  assert.match(snippet, /stopAllPoolBackends\(\)/)
  assert.match(snippet, /backendQuitReady = true\s*app\.quit\(\)/)
})
